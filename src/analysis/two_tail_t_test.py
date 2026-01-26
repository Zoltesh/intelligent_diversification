from __future__ import annotations

from typing import Dict, List

import polars as pl
from scipy import stats


def _weekly_returns_df(results: Dict[str, object], label: str) -> pl.DataFrame:
    weekly_metrics = results.get("weekly_metrics")
    if not weekly_metrics:
        raise ValueError(f"{label} results missing weekly_metrics.")

    rows: List[Dict[str, object]] = []
    for week in weekly_metrics:
        timestamp = week.get("timestamp")
        portfolio = week.get("portfolio", {})
        net_return_pct = portfolio.get("net_return_pct")
        if timestamp is None or net_return_pct is None:
            raise ValueError(f"{label} weekly_metrics missing timestamp or net_return_pct.")
        rows.append(
            {
                "timestamp": int(timestamp),
                f"net_return_pct_{label}": net_return_pct,
            }
        )

    df = pl.DataFrame(rows)
    return df.with_columns(
        pl.col(f"net_return_pct_{label}").cast(pl.Float64).alias(f"net_return_pct_{label}")
    )


def run_two_tail_t_test(
    optimized_results: Dict[str, object],
    buy_and_hold_results: Dict[str, object],
    *,
    expected_weeks: int = 52,
) -> Dict[str, float | int]:
    optimized_df = _weekly_returns_df(optimized_results, "optimized")
    buy_hold_df = _weekly_returns_df(buy_and_hold_results, "buy_and_hold")

    joined = optimized_df.join(buy_hold_df, on="timestamp", how="inner").sort("timestamp")
    if joined.height != optimized_df.height or joined.height != buy_hold_df.height:
        raise ValueError(
            "Weekly timestamps do not match between optimized and buy-and-hold results."
        )
    if expected_weeks and joined.height != expected_weeks:
        raise ValueError(
            f"Expected {expected_weeks} weeks after alignment, got {joined.height}."
        )

    optimized_returns = joined.get_column("net_return_pct_optimized").to_list()
    buy_hold_returns = joined.get_column("net_return_pct_buy_and_hold").to_list()

    t_statistic, p_value = stats.ttest_rel(optimized_returns, buy_hold_returns)
    return {
        "t_statistic": float(t_statistic),
        "p_value": float(p_value),
        "n": len(optimized_returns),
    }
