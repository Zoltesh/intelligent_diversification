"""
Predictions using XGBoost
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import xgboost as xgb

WINDOW_SIZE = 2016  # 12 intervals * 24 hours * 7 days
START_TIMESTAMP = 1735689600000  # Jan 1, 2025 00:00:00 UTC
WEEKS_2025 = 52


def load_engineered_features(data_dir: Path) -> dict[str, pl.DataFrame]:
    assets_dict: dict[str, pl.DataFrame] = {}
    for csv_path in sorted(data_dir.glob("*_feature_df.csv")):
        ticker = csv_path.name.split("_feature_df.csv")[0]
        assets_dict[ticker] = pl.read_csv(csv_path)
    if not assets_dict:
        raise ValueError(f"No feature CSVs found in {data_dir}")
    return assets_dict


def add_targets(assets_dict: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    updated: dict[str, pl.DataFrame] = {}
    for ticker, df in assets_dict.items():
        updated[ticker] = df.with_columns(
            (
                (pl.col("close").shift(-WINDOW_SIZE) / pl.col("close")) - 1
            ).alias("target_return_1w")
        )
    return updated


def get_feature_cols(df: pl.DataFrame) -> list[str]:
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


def build_test_indices(reference_df: pl.DataFrame) -> list[int]:
    week_ms = WINDOW_SIZE * 5 * 60 * 1000
    target_timestamps = [
        START_TIMESTAMP + (week_ms * i) for i in range(WEEKS_2025)
    ]

    reference = reference_df.select("timestamp").with_row_index("idx")
    reference = reference.sort("timestamp")
    targets = pl.DataFrame({"timestamp": target_timestamps}).sort("timestamp")

    forward = targets.join_asof(
        reference,
        on="timestamp",
        strategy="forward",
    )
    if forward["idx"].null_count() > 0:
        backward = targets.join_asof(
            reference,
            on="timestamp",
            strategy="backward",
        )
        indices = forward["idx"].fill_null(backward["idx"])
    else:
        indices = forward["idx"]

    if indices.null_count() > 0:
        raise ValueError(
            "Unable to map all weekly timestamps to indices"
        )

    return indices.cast(pl.Int64).to_list()


def walk_forward_validation(
    assets_dict: dict[str, pl.DataFrame]
) -> tuple[dict[str, list[float]], dict[str, dict[str, list[dict[str, float]]]]]:
    assets_dict = add_targets(assets_dict)
    tickers = list(assets_dict.keys())
    reference_df = assets_dict[tickers[0]]
    test_indices = build_test_indices(reference_df)

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

    for week_idx, current_idx in enumerate(test_indices, start=1):
        for ticker, df in assets_dict.items():
            feature_cols = get_feature_cols(df)
            train_df = df.slice(0, current_idx)
            train_clean = train_df.filter(
                pl.col("target_return_1w").is_not_null()
            )
            if train_clean.is_empty():
                raise ValueError(
                    f"Empty training set for {ticker} at idx {current_idx}"
                )

            x_train = train_clean.select(feature_cols).to_numpy()
            y_train = (
                train_clean.select("target_return_1w")
                .to_numpy()
                .reshape(-1)
            )

            test_row = df.slice(current_idx, 1)
            x_test = test_row.select(feature_cols).to_numpy()

            dtrain = xgb.DMatrix(
                x_train, label=y_train, feature_names=feature_cols
            )
            dtest = xgb.DMatrix(x_test, feature_names=feature_cols)

            params = {
                "max_depth": 3,
                "eta": 0.1,
                "objective": "reg:squarederror",
                "device": "cuda",
                "tree_method": "hist",
                "seed": 42,
            }
            booster = xgb.train(params, dtrain, num_boost_round=100)
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
    output_path.write_text(json.dumps(results, indent=2))


def save_importances(
    importances: dict[str, dict[str, list[dict[str, float]]]],
    output_path: Path,
) -> None:
    output_path.write_text(json.dumps(importances, indent=2))

def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "engineered_features"
    assets_dict = load_engineered_features(data_dir)
    results, importances = walk_forward_validation(assets_dict)
    output_path = Path(__file__).resolve().parents[1] / "weekly_predictions_2025.json"
    save_predictions(results, output_path)
    importance_path = (
        Path(__file__).resolve().parents[1]
        / "weekly_feature_importances_2025.json"
    )
    save_importances(importances, importance_path)


if __name__ == "__main__":
    main()
