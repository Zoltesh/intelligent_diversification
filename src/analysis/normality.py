"""
Normality checks (Shapiro-Wilk & Kolmogorov-Smirnov)
Corrected to fit the KS test to the data's specific mean/std.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import polars as pl
from scipy import stats


def _weekly_returns_df(results: Dict[str, object], label: str) -> pl.DataFrame:
    """
    Extracts weekly net returns from the simulation results dictionary.
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


def run_shapiro_wilk(returns: List[float]) -> Dict[str, float]:
    """
    Performs the Shapiro-Wilk test for normality.
    """
    statistic, p_value = stats.shapiro(returns)
    return {
        "shapiro_stat": float(statistic),
        "shapiro_p_value": float(p_value),
    }


def run_kolmogorov_smirnov(returns: List[float]) -> Dict[str, float]:
    """
    Performs the Kolmogorov-Smirnov test for normality.
    
    Must pass the mean and std of the data to the test.
    Otherwise, it tests against N(0,1), which is incorrect for returns data.
    """
    mean_val = np.mean(returns)
    std_val = np.std(returns, ddof=1)
    
    # args=(mean, std) fits the normal distribution to the data
    statistic, p_value = stats.kstest(returns, "norm", args=(mean_val, std_val))
    
    return {
        "ks_stat": float(statistic),
        "ks_p_value": float(p_value),
    }


def assess_normality(
    results: Dict[str, object],
    label: str,
    *,
    expected_weeks: int = 52,
) -> Dict[str, float | int | str]:
    """
    Orchestrator that extracts data and runs BOTH normality tests
    as required by the Capstone Proposal.
    """
    # 1. Extract Data
    df = _weekly_returns_df(results, label)

    # 2. Validation
    if expected_weeks and df.height != expected_weeks:
        raise ValueError(
            f"Expected {expected_weeks} weeks for {label}, got {df.height}."
        )

    # 3. Get list of returns
    returns_list = df.get_column(f"net_return_pct_{label}").to_list()

    # 4. Run BOTH Tests
    shapiro_res = run_shapiro_wilk(returns_list)
    ks_res = run_kolmogorov_smirnov(returns_list)

    # 5. Combine Results
    return {
        "label": label,
        "n": len(returns_list),
        **shapiro_res,
        **ks_res,
    }