"""
Wilcoxon Signed-Rank Test Module
Used when the normality assumption of the t-test is violated.
"""
from __future__ import annotations

from typing import Dict, List, Literal

import polars as pl
from scipy import stats


def _weekly_returns_df(results: Dict[str, object], label: str) -> pl.DataFrame:
    """
    Extracts weekly net returns from the simulation results dictionary.
    Identical logic to two_tail_t_test.py to ensure data consistency.
    """
    weekly_metrics = results.get("weekly_metrics")
    if not weekly_metrics:
        raise ValueError(f"{label} results missing weekly_metrics.")

    rows: List[Dict[str, object]] = []
    for week in weekly_metrics:
        timestamp = week.get("timestamp")
        portfolio = week.get("portfolio", {})
        net_return_pct = portfolio.get("net_return_pct")

        if timestamp is None or net_return_pct is None:
            raise ValueError(
                f"{label} weekly_metrics missing timestamp or net_return_pct."
            )
        rows.append(
            {
                "timestamp": int(timestamp),
                f"net_return_pct_{label}": net_return_pct,
            }
        )

    df = pl.DataFrame(rows)
    return df.with_columns(
        pl.col(f"net_return_pct_{label}")
        .cast(pl.Float64)
        .alias(f"net_return_pct_{label}")
    )


def run_wilcoxon_test(
    optimized_results: Dict[str, object],
    buy_and_hold_results: Dict[str, object],
    *,
    expected_weeks: int = 52,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
) -> Dict[str, float | int | str]:
    """
    Performs the non-parametric Wilcoxon Signed-Rank Test.
    
    Args:
        optimized_results: Simulation output for the XGBoost strategy.
        buy_and_hold_results: Simulation output for the Benchmark.
        expected_weeks: Validation check for sample size.
        alternative: "two-sided" checks for ANY difference.
                     "greater" checks if optimized > buy_and_hold.
    
    Returns:
        Dictionary containing statistic, p-value, and sample size.
    """
    # 1. Extract and Align Data
    optimized_df = _weekly_returns_df(optimized_results, "optimized")
    buy_hold_df = _weekly_returns_df(buy_and_hold_results, "buy_and_hold")

    joined = optimized_df.join(buy_hold_df, on="timestamp", how="inner").sort("timestamp")

    # 2. Validation
    if joined.height != optimized_df.height or joined.height != buy_hold_df.height:
        raise ValueError(
            "Weekly timestamps do not match between optimized and buy-and-hold results."
        )
    if expected_weeks and joined.height != expected_weeks:
        raise ValueError(
            f"Expected {expected_weeks} weeks after alignment, got {joined.height}."
        )

    # 3. Convert to Lists
    optimized_returns = joined.get_column("net_return_pct_optimized").to_list()
    buy_hold_returns = joined.get_column("net_return_pct_buy_and_hold").to_list()

    # 4. Run Wilcoxon Signed-Rank Test
    # This tests the null hypothesis that two related paired samples come from the same distribution.
    statistic, p_value = stats.wilcoxon(
        optimized_returns, 
        buy_hold_returns, 
        alternative=alternative
    )

    return {
        "test_type": "Wilcoxon Signed-Rank Test",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n": len(optimized_returns),
        "significance": "Significant" if p_value < 0.05 else "Not Significant",
    }