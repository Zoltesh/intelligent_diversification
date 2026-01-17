from __future__ import annotations

import json
from pathlib import Path

import polars as pl

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


def load_predictions(pred_path: Path) -> dict[str, list[float]]:
    data = json.loads(pred_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Predictions JSON must be a dict")
    return {k: list(v) for k, v in data.items()}


def build_weekly_timestamps() -> list[int]:
    week_ms = WINDOW_SIZE * 5 * 60 * 1000
    return [START_TIMESTAMP + (week_ms * i) for i in range(WEEKS_2025)]


def map_closes_for_timestamps(
    df: pl.DataFrame, timestamps: list[int]
) -> list[float]:
    base = df.select(["timestamp", "close"]).sort("timestamp")
    targets = pl.DataFrame({"timestamp": timestamps}).sort("timestamp")

    forward = targets.join_asof(
        base, on="timestamp", strategy="forward"
    )
    if forward["close"].null_count() > 0:
        backward = targets.join_asof(
            base, on="timestamp", strategy="backward"
        )
        closes = forward["close"].fill_null(backward["close"])
    else:
        closes = forward["close"]

    if closes.null_count() > 0:
        raise ValueError("Unable to map all timestamps to closes")

    return [float(x) for x in closes.to_list()]


def compute_actual_returns(
    df: pl.DataFrame, weekly_timestamps: list[int]
) -> list[float]:
    week_ms = WINDOW_SIZE * 5 * 60 * 1000
    future_timestamps = [ts + week_ms for ts in weekly_timestamps]
    current_closes = map_closes_for_timestamps(df, weekly_timestamps)
    future_closes = map_closes_for_timestamps(df, future_timestamps)
    return [
        (future / current) - 1
        for current, future in zip(current_closes, future_closes)
    ]


def compare_predictions(
    assets_dict: dict[str, pl.DataFrame],
    predictions: dict[str, list[float]],
) -> dict[str, dict[str, list[float]]]:
    weekly_timestamps = build_weekly_timestamps()
    results: dict[str, dict[str, list[float]]] = {}

    for ticker, df in assets_dict.items():
        if ticker not in predictions:
            raise ValueError(f"Missing predictions for {ticker}")
        preds = predictions[ticker]
        if len(preds) != WEEKS_2025:
            raise ValueError(
                f"{ticker} predictions length {len(preds)} != {WEEKS_2025}"
            )

        actual = compute_actual_returns(df, weekly_timestamps)
        if len(actual) != WEEKS_2025:
            raise ValueError(
                f"{ticker} actual length {len(actual)} != {WEEKS_2025}"
            )

        error = [p - a for p, a in zip(preds, actual)]
        abs_error = [abs(e) for e in error]

        results[ticker] = {
            "predicted": preds,
            "actual": actual,
            "error": error,
            "abs_error": abs_error,
        }

    results["timestamps"] = weekly_timestamps
    return results


def save_results(results: dict[str, dict[str, list[float]]], out_path: Path) -> None:
    out_path.write_text(json.dumps(results, indent=2))


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    data_dir = base_dir / "engineered_features"
    pred_path = base_dir / "weekly_predictions_2025.json"
    out_path = base_dir / "weekly_price_validation_2025.json"

    assets_dict = load_engineered_features(data_dir)
    predictions = load_predictions(pred_path)
    results = compare_predictions(assets_dict, predictions)
    save_results(results, out_path)


if __name__ == "__main__":
    main()
