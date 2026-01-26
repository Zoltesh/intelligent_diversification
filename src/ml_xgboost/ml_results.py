from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def _pearson_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or not xs:
        return 0.0
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    cov = _mean((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = _mean((x - mean_x) ** 2 for x in xs)
    var_y = _mean((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def _round_metric(value: float, decimals: int = 4) -> float:
    return round(value, decimals)


def _compute_metrics(preds: list[float], actuals: list[float]) -> dict[str, float]:
    if len(preds) != len(actuals) or not preds:
        raise ValueError("Predictions and actuals must be same non-zero length")

    n = len(preds)
    errors = [p - a for p, a in zip(preds, actuals)]

    # Error metrics (use all data points)
    mae = _mean(abs(e) for e in errors)
    rmse = math.sqrt(_mean(e**2 for e in errors))
    bias = _mean(errors)
    corr = _pearson_corr(preds, actuals)

    # Directional metrics: exclude zero actuals, use p >= 0 for positive prediction
    # DirAcc = sign match rate (pred >= 0 and actual > 0) or (pred < 0 and actual < 0)
    correct = 0
    nonzero_count = 0
    for p, a in zip(preds, actuals):
        if a == 0:
            continue
        nonzero_count += 1
        if (p >= 0 and a > 0) or (p < 0 and a < 0):
            correct += 1
    dir_acc = correct / nonzero_count if nonzero_count else 0.0

    # PosPred%: fraction of predictions that are >= 0
    pos_pred_count = sum(1 for p in preds if p >= 0)
    pos_pred_pct = (pos_pred_count / n) * 100

    # PosPredHit: among positive predictions (p >= 0), what fraction had actual > 0
    pos_pred_actuals = [a for p, a in zip(preds, actuals) if p >= 0]
    pos_pred_hit = (
        sum(1 for a in pos_pred_actuals if a > 0) / len(pos_pred_actuals)
        if pos_pred_actuals
        else 0.0
    )

    # NegPredHit: among negative predictions (p < 0), what fraction had actual < 0
    neg_pred_actuals = [a for p, a in zip(preds, actuals) if p < 0]
    neg_pred_hit = (
        sum(1 for a in neg_pred_actuals if a < 0) / len(neg_pred_actuals)
        if neg_pred_actuals
        else 0.0
    )

    return {
        "N": n,
        "DirAcc": _round_metric(dir_acc, 4),
        "MAE": _round_metric(mae, 4),
        "RMSE": _round_metric(rmse, 4),
        "Bias": _round_metric(bias, 4),
        "Corr": _round_metric(corr, 4),
        "PosPred%": _round_metric(pos_pred_pct, 1),
        "PosPredHit": _round_metric(pos_pred_hit, 4),
        "NegPredHit": _round_metric(neg_pred_hit, 4),
    }


def _load_price_validation(results_dir: Path) -> dict[str, dict[str, list[float]]]:
    path = results_dir / "weekly_price_validation_2025.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing price validation file: {path}")
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Price validation JSON must be a dict")
    return data


def build_ml_results(
    *,
    results_dir: Path | None = None,
) -> dict[str, dict[str, float]]:
    results_dir = results_dir or (Path(__file__).resolve().parent / "results")
    raw = _load_price_validation(results_dir)

    metrics: dict[str, dict[str, float]] = {}
    for ticker, values in raw.items():
        if ticker == "timestamps":
            continue
        if not isinstance(values, dict):
            raise ValueError(f"Invalid results payload for {ticker}")
        preds = list(values.get("predicted", []))
        actuals = list(values.get("actual", []))
        metrics[ticker] = _compute_metrics(preds, actuals)

    return metrics


def save_ml_results(metrics: dict[str, dict[str, float]], out_path: Path) -> None:
    out_path.write_text(json.dumps(metrics, indent=2))


def main() -> None:
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics = build_ml_results(results_dir=results_dir)
    out_path = results_dir / "ml_results.json"
    save_ml_results(metrics, out_path)


if __name__ == "__main__":
    main()
