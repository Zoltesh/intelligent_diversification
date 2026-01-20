"""
Predictions using XGBoost
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl
import xgboost as xgb

from features.feature_store import FeatureStore
from pipeline.time_index import (
    WINDOW_SIZE,
    WEEKS_2025,
    build_test_indices,
    build_week_keys,
)

DEFAULT_XGB_PARAMS = {
    "max_depth": 3,
    "eta": 0.1,
    "objective": "reg:squarederror",
    "device": "cuda",
    "tree_method": "hist",
    "seed": 42,
}

DEFAULT_NUM_BOOST_ROUND = 100


def load_engineered_features(data_dir: Path) -> dict[str, pl.DataFrame]:
    """Load engineered feature CSVs keyed by ticker."""
    assets_dict: dict[str, pl.DataFrame] = {}
    for csv_path in sorted(data_dir.glob("*_feature_df.csv")):
        ticker = csv_path.name.split("_feature_df.csv")[0]
        assets_dict[ticker] = pl.read_csv(csv_path)
    if not assets_dict:
        raise ValueError(f"No feature CSVs found in {data_dir}")
    return assets_dict


def add_targets(assets_dict: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Add the 1-week forward return target for each asset."""
    updated: dict[str, pl.DataFrame] = {}
    for ticker, df in assets_dict.items():
        updated[ticker] = df.with_columns(
            (
                (pl.col("close").shift(-WINDOW_SIZE) / pl.col("close")) - 1
            ).alias("target_return_1w")
        )
    return updated


def get_feature_cols(df: pl.DataFrame) -> list[str]:
    """Return feature columns, excluding base OHLCV and target."""
    exclude = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "target_return_1w",
    }
    return [col for col in df.columns if col not in exclude]


def walk_forward_validation(
    assets_dict: dict[str, pl.DataFrame],
    xgb_params: dict[str, object] | None = None,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
) -> tuple[dict[str, list[float]], dict[str, dict[str, list[dict[str, float]]]]]:
    """Run weekly walk-forward training and prediction per asset."""
    assets_dict = add_targets(assets_dict)
    tickers = list(assets_dict.keys())
    reference_df = assets_dict[tickers[0]]
    test_indices = build_test_indices(reference_df)
    week_keys = build_week_keys(len(test_indices))
    feature_store = FeatureStore(Path(__file__).resolve().parents[1] / "features")
    weekly_features = {
        ticker.lower(): feature_store.resolve_weekly_features(
            ticker,
            assets_dict[ticker],
            week_keys,
            read_existing=True,
        )
        for ticker in tickers
    }

    results: dict[str, list[float]] = {ticker: [] for ticker in tickers}
    weekly_importances: dict[str, list[list[dict[str, float]]]] = {
        ticker: [] for ticker in tickers
    }
    importance_sums: dict[str, dict[str, float]] = {
        ticker: {} for ticker in tickers
    }
    importance_counts: dict[str, dict[str, int]] = {
        ticker: {} for ticker in tickers
    }

    # Precompute numpy arrays and column indices to avoid repeated conversions.
    precomputed: dict[str, dict[str, object]] = {}
    for ticker, df in assets_dict.items():
        columns = df.columns
        col_idx = {name: idx for idx, name in enumerate(columns)}
        data = df.to_numpy()
        target = np.asarray(
            df.get_column("target_return_1w").to_numpy(), dtype=float
        )
        row_idx = np.arange(df.height)
        precomputed[ticker] = {
            "columns": columns,
            "col_idx": col_idx,
            "data": data,
            "target": target,
            "row_idx": row_idx,
        }

    params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}

    for week_idx, current_idx in enumerate(test_indices, start=1):
        for ticker, df in assets_dict.items():
            ticker_key = ticker.lower()
            if ticker_key not in weekly_features:
                raise ValueError(f"Missing features for {ticker}")
            week_key = f"week{week_idx}"
            feature_cols = weekly_features[ticker_key].get(week_key)
            if not feature_cols:
                raise ValueError(f"No features for {ticker} {week_key}")
            columns = precomputed[ticker]["columns"]
            missing_cols = [col for col in feature_cols if col not in columns]
            if missing_cols:
                raise ValueError(
                    f"Missing feature columns for {ticker} {week_key}: {missing_cols}"
                )
            col_idx = precomputed[ticker]["col_idx"]
            feature_idx = [col_idx[col] for col in feature_cols]
            data = precomputed[ticker]["data"]
            target = precomputed[ticker]["target"]
            row_idx = precomputed[ticker]["row_idx"]

            train_mask = (
                (row_idx < current_idx)
                & ~np.isnan(target)
                & (row_idx + WINDOW_SIZE < current_idx)
            )
            if not train_mask.any():
                raise ValueError(
                    f"Empty training set for {ticker} at idx {current_idx}"
                )

            x_train = data[train_mask][:, feature_idx]
            y_train = target[train_mask]
            x_test = data[current_idx : current_idx + 1, feature_idx]

            dtrain = xgb.QuantileDMatrix(
                x_train, label=y_train, feature_names=feature_cols
            )

            booster = xgb.train(params, dtrain, num_boost_round=num_boost_round)
            # Use DMatrix for prediction to avoid device mismatch warnings.
            dtest = xgb.DMatrix(x_test, feature_names=feature_cols)
            pred = float(booster.predict(dtest)[0])
            results[ticker].append(pred)

            raw_importance = booster.get_score(importance_type="gain")
            sorted_importance = sorted(
                raw_importance.items(), key=lambda kv: kv[1], reverse=True
            )
            weekly_importances[ticker].append(
                [{"feature": k, "gain": float(v)} for k, v in sorted_importance]
            )
            for feature, gain in raw_importance.items():
                importance_sums[ticker][feature] = (
                    importance_sums[ticker].get(feature, 0.0) + float(gain)
                )
                importance_counts[ticker][feature] = (
                    importance_counts[ticker].get(feature, 0) + 1
                )

        print(f"Completed Week {week_idx}/{WEEKS_2025}")

    avg_importances: dict[str, list[dict[str, float]]] = {}
    for ticker in tickers:
        avg = {
            feature: importance_sums[ticker][feature]
            / importance_counts[ticker][feature]
            for feature in importance_sums[ticker]
        }
        avg_sorted = sorted(avg.items(), key=lambda kv: kv[1], reverse=True)
        avg_importances[ticker] = [
            {"feature": k, "avg_gain": float(v)} for k, v in avg_sorted
        ]

    return results, {
        "average_gain": avg_importances,
        "weekly_gain": weekly_importances,
    }


def save_predictions(results: dict[str, list[float]], output_path: Path) -> None:
    """Persist predictions to JSON."""
    output_path.write_text(json.dumps(results, indent=2))


def save_importances(
    importances: dict[str, dict[str, list[dict[str, float]]]],
    output_path: Path,
) -> None:
    """Persist feature importances to JSON."""
    output_path.write_text(json.dumps(importances, indent=2))


def run_predictions(
    *,
    xgb_params: dict[str, object] | None = None,
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND,
    data_dir: Path | None = None,
    results_dir: Path | None = None,
) -> tuple[dict[str, list[float]], dict[str, dict[str, list[dict[str, float]]]]]:
    """Run full prediction pipeline and write outputs."""
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = data_dir or (base_dir / "engineered_features")
    results_dir = results_dir or (Path(__file__).resolve().parent / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    assets_dict = load_engineered_features(data_dir)
    results, importances = walk_forward_validation(
        assets_dict,
        xgb_params=xgb_params,
        num_boost_round=num_boost_round,
    )
    output_path = results_dir / "weekly_predictions_2025.json"
    save_predictions(results, output_path)
    importance_path = results_dir / "weekly_feature_importances_2025.json"
    save_importances(importances, importance_path)
    return results, importances


def main() -> None:
    run_predictions()


if __name__ == "__main__":
    main()
