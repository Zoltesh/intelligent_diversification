"""
Strictly sequential, expanding-window portfolio simulation.
Bulletproof Version: Uses 'Soft Constraints' to prevent solver infeasibility.
"""

from __future__ import annotations
import cvxpy as cp
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
from pypfopt import EfficientFrontier, objective_functions

BASE_DIR = Path(__file__).resolve().parents[1]

# --- CONFIGURATION ---
START_TS_MS = 1735689600000  # 2025-01-01
WEEKS = 52
MS_PER_DAY = 86_400_000
MS_PER_WEEK = 7 * MS_PER_DAY

# CAP CONFIGURATION
# 0.35 = Max 35% per asset. Allow more concentration on high-conviction picks.
MAX_ASSET_WEIGHT = 0.35 
DEFAULT_L2_GAMMA = 0.01  # Deviation from equal weights
DEFAULT_TXN_COST_K = 0.00325  # Averaging Taker/Maker for Tier 2 on Coinbase Advanced
DEFAULT_TURNOVER_CAP = 0.35


def load_predictions(predictions_path: Path) -> Dict[str, List[float]]:
    with predictions_path.open("r") as f:
        predictions = json.load(f)
    return predictions


def load_wide_prices(
    data_dir: Path, symbols: List[str]
) -> Tuple[pl.DataFrame, List[str]]:
    long_frames: List[pl.DataFrame] = []
    missing_files: List[str] = []

    for symbol in symbols:
        csv_path = data_dir / f"{symbol}_feature_df.csv"
        if not csv_path.exists():
            missing_files.append(str(csv_path))
            continue
        df = pl.read_csv(csv_path, columns=["timestamp", "close"])
        long_frames.append(
            df.with_columns(pl.lit(symbol).alias("symbol")).select(
                ["timestamp", "symbol", "close"]
            )
        )

    if missing_files:
        raise FileNotFoundError(f"Missing engineered feature CSVs: {missing_files}")

    long_df = pl.concat(long_frames)
    
    wide_df = (
        long_df.pivot(
            values="close",
            index="timestamp",
            on="symbol",
            aggregate_function="first",
        )
        .sort("timestamp")
    )

    expected_cols = ["timestamp"] + symbols
    missing_cols = [c for c in symbols if c not in wide_df.columns]
    if missing_cols:
        raise ValueError(f"Missing symbols in price data: {missing_cols}")

    wide_df = wide_df.select(expected_cols)
    return wide_df, symbols


def infer_annualization_factor(history_df: pl.DataFrame) -> int:
    diffs = (
        history_df.select(pl.col("timestamp").diff().drop_nulls())
        .to_numpy()
        .flatten()
    )
    if diffs.size == 0:
        return 365 
    median_days = np.median(diffs) / MS_PER_DAY
    return 52 if median_days >= 6.5 else 365


def compute_covariance(history_df: pl.DataFrame, annualization_factor: int) -> np.ndarray:
    asset_cols = [c for c in history_df.columns if c != "timestamp"]
    prices_only = history_df.select(asset_cols)
    
    returns = prices_only.select(
        [((pl.col(c) / pl.col(c).shift(1)) - 1).alias(c) for c in asset_cols]
    ).drop_nulls()

    if returns.height < 2:
        return np.eye(len(asset_cols)) * 1e-6

    returns_matrix = returns.to_numpy()
    if returns_matrix.ndim == 1:
        returns_matrix = returns_matrix.reshape(-1, 1)
    cov_matrix = np.cov(returns_matrix, rowvar=False) * annualization_factor
    if np.isscalar(cov_matrix):
        cov_matrix = np.array([[cov_matrix]])
    cov_matrix = np.nan_to_num(cov_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return cov_matrix


def equal_weights(symbols: List[str]) -> Dict[str, float]:
    weight = 1.0 / len(symbols)
    return {symbol: weight for symbol in symbols}


def normalize_weights(weights: Dict[str, float], symbols: List[str]) -> Dict[str, float]:
    total = sum(weights.get(s, 0.0) for s in symbols)
    if total <= 0:
        return equal_weights(symbols)
    return {s: float(weights.get(s, 0.0)) / total for s in symbols}


def compute_investment_budget(weekly_preds: List[float]) -> float:
    """
    Compute investment budget based on prediction magnitude, not just count.
    Returns 1.0 (fully invested) unless predictions are strongly negative.
    """
    if not weekly_preds:
        return 0.0
    
    # Use magnitude-weighted approach
    positive_sum = sum(max(0, pred) for pred in weekly_preds)
    negative_sum = abs(sum(min(0, pred) for pred in weekly_preds))
    total_magnitude = positive_sum + negative_sum
    
    if total_magnitude == 0:
        return 0.0
    
    # Budget scales with positive vs negative magnitude
    # If positive_sum dominates, budget approaches 1.0
    # If negative_sum dominates, budget approaches 0.0
    budget = positive_sum / total_magnitude
    
    # Apply floor: minimum 60% invested unless predictions are very bearish
    return max(0.6, budget) if budget > 0.3 else budget


def apply_weight_cap(weights: Dict[str, float], cap: float) -> Dict[str, float]:
    """
    The 'Human' Logic:
    1. Check if any asset > 25%.
    2. If yes, cut it to 25%.
    3. Take the 'missing percent' and distribute it proportionally to everyone else.
    4. Repeat until stable (because adding to others might push THEM over 25%).
    """
    if not weights:
        return {}
        
    cleaned = normalize_weights(weights, list(weights.keys()))
    symbols = list(cleaned.keys())

    if cap <= 0:
        return equal_weights(symbols)

    min_feasible_cap = 1.0 / len(symbols)
    if cap * len(symbols) < 1.0:
        cap = min_feasible_cap

    remaining = cleaned.copy()
    remaining_budget = 1.0
    weights_out: Dict[str, float] = {}

    while remaining:
        remaining_total = sum(remaining.values())
        if remaining_total <= 0:
            return equal_weights(symbols)

        scaled = {
            k: (v / remaining_total) * remaining_budget for k, v in remaining.items()
        }
        over = {k: v for k, v in scaled.items() if v > cap}

        if not over:
            weights_out.update(scaled)
            break

        for k in over:
            weights_out[k] = cap
            remaining_budget -= cap
            remaining.pop(k, None)

        if remaining_budget <= 0:
            break

    return normalize_weights(weights_out, symbols)


def _build_frontier(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    prev_weights_array: np.ndarray,
    symbols: List[str],
    l2_gamma: float,
    txn_cost_k: float,
    turnover_cap: float | None,
) -> EfficientFrontier:
    ef = EfficientFrontier(expected_returns, cov_matrix, weight_bounds=(0.0, 1.0))
    ef.tickers = symbols
    ef.add_objective(objective_functions.L2_reg, gamma=l2_gamma)
    if txn_cost_k > 0:
        ef.add_objective(
            objective_functions.transaction_cost, w_prev=prev_weights_array, k=txn_cost_k
        )
    if turnover_cap is not None and turnover_cap > 0:
        ef.add_constraint(
            lambda w: cp.sum(cp.abs(w - prev_weights_array)) <= turnover_cap
        )
    return ef


def _solve_objective(ef: EfficientFrontier, objective: str) -> None:
    if objective == "max_sharpe":
        ef.max_sharpe(risk_free_rate=0.0)
    elif objective == "min_volatility":
        ef.min_volatility()
    elif objective == "max_quadratic_utility":
        ef.max_quadratic_utility(risk_aversion=0.5)
    else:
        raise ValueError(f"Unknown objective: {objective}")


def run_simulation(
    wide_df: pl.DataFrame,
    predictions: Dict[str, List[float]],
    *,
    objective: str = "max_sharpe",
    l2_gamma: float = DEFAULT_L2_GAMMA,
    txn_cost_k: float = DEFAULT_TXN_COST_K,
    turnover_cap: float = DEFAULT_TURNOVER_CAP,
) -> List[Dict[str, object]]:
    symbols = list(predictions.keys())
    history_df = wide_df.filter(pl.col("timestamp") < START_TS_MS)
    future_df = wide_df.filter(pl.col("timestamp") >= START_TS_MS)

    annualization_factor = infer_annualization_factor(history_df)
    
    results: List[Dict[str, object]] = []
    prev_weights: Dict[str, float] | None = None

    for week_idx in range(WEEKS):
        week_start = START_TS_MS + week_idx * MS_PER_WEEK

        # 1. Compute Data
        cov_matrix = compute_covariance(history_df, annualization_factor)
        raw_weekly_preds = [predictions[symbol][week_idx] for symbol in symbols]
        expected_returns = np.array(raw_weekly_preds, dtype=float) * 52
        investment_budget = compute_investment_budget(raw_weekly_preds)
        if prev_weights is None:
            prev_weights_array = np.zeros(len(symbols), dtype=float)
        else:
            prev_weights_array = np.array(
                [prev_weights.get(symbol, 0.0) for symbol in symbols], dtype=float
            )

        try:
            # --- STEP 1: SOLVER ---
            # Rely on L2_reg to discourage 100% allocation, but let the solver
            # output whatever it wants (e.g. 60%) to ensure it finds a solution.
            ef = _build_frontier(
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                prev_weights_array=prev_weights_array,
                symbols=symbols,
                l2_gamma=l2_gamma,
                txn_cost_k=txn_cost_k,
                turnover_cap=turnover_cap,
            )

            try:
                _solve_objective(ef, objective)
            except Exception as e:
                if "infeasible" in str(e).lower():
                    ef = _build_frontier(
                        expected_returns=expected_returns,
                        cov_matrix=cov_matrix,
                        prev_weights_array=prev_weights_array,
                        symbols=symbols,
                        l2_gamma=l2_gamma,
                        txn_cost_k=txn_cost_k,
                        turnover_cap=None,
                    )
                    _solve_objective(ef, objective)
                else:
                    raise
            
            cleaned_weights = ef.clean_weights(cutoff=0.01)
            
            # --- STEP 2: HUMAN-LIKE LOGIC ---
            weights = apply_weight_cap(cleaned_weights, cap=MAX_ASSET_WEIGHT)

        except Exception as e:
            # Fallback for Bear Markets (when all returns are negative)
            print(f"Week {week_idx} Max Sharpe failed ({e}). Switching to Defensive Mode.")
            try:
                ef_risk = _build_frontier(
                    expected_returns=expected_returns,
                    cov_matrix=cov_matrix,
                    prev_weights_array=prev_weights_array,
                    symbols=symbols,
                    l2_gamma=l2_gamma,
                    txn_cost_k=txn_cost_k,
                    turnover_cap=turnover_cap,
                )
                
                try:
                    ef_risk.min_volatility()
                except Exception as e2:
                    if "infeasible" in str(e2).lower():
                        ef_risk = _build_frontier(
                            expected_returns=expected_returns,
                            cov_matrix=cov_matrix,
                            prev_weights_array=prev_weights_array,
                            symbols=symbols,
                            l2_gamma=l2_gamma,
                            txn_cost_k=txn_cost_k,
                            turnover_cap=None,
                        )
                        ef_risk.min_volatility()
                    else:
                        raise
                
                cleaned_weights = ef_risk.clean_weights(cutoff=0.01)
                
                # Apply same capping logic to defensive portfolio
                weights = apply_weight_cap(cleaned_weights, cap=MAX_ASSET_WEIGHT)
                
            except Exception as e2:
                print(f"Week {week_idx} Critical Failure. Using Equal Weights.")
                if prev_weights:
                    weights = {symbol: float(prev_weights.get(symbol, 0.0)) for symbol in symbols}
                else:
                    weights = equal_weights(symbols)

        weights = {symbol: weight * investment_budget for symbol, weight in weights.items()}
        results.append({"timestamp": week_start, "weights": weights})
        prev_weights = weights

        # Expand Window
        week_slice = future_df.filter(
            (pl.col("timestamp") >= week_start)
            & (pl.col("timestamp") < week_start + MS_PER_WEEK)
        )
        if week_slice.height > 0:
            history_df = pl.concat([history_df, week_slice]).sort("timestamp")

    return results


def run_optimization(
    *,
    predictions_path: Path | None = None,
    data_dir: Path | None = None,
    results_dir: Path | None = None,
    weights_filename: str = "weekly_weights",
    objective: str = "max_sharpe",
    l2_gamma: float = DEFAULT_L2_GAMMA,
    txn_cost_k: float = DEFAULT_TXN_COST_K,
    turnover_cap: float = DEFAULT_TURNOVER_CAP,
) -> List[Dict[str, object]]:
    predictions_path = predictions_path or (
        BASE_DIR / "ml_xgboost" / "results" / "weekly_predictions_2025.json"
    )
    data_dir = data_dir or (BASE_DIR / "engineered_features")
    results_dir = results_dir or (Path(__file__).resolve().parent / "results")
    results_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(predictions_path)
    symbols = list(predictions.keys())

    if any(len(predictions[symbol]) != WEEKS for symbol in symbols):
        raise ValueError("Predictions must have 52 weeks.")

    wide_df, _ = load_wide_prices(data_dir, symbols)
    results = run_simulation(
        wide_df,
        predictions,
        objective=objective,
        l2_gamma=l2_gamma,
        txn_cost_k=txn_cost_k,
        turnover_cap=turnover_cap,
    )

    output_path = results_dir / f"{weights_filename}.json"
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    return results


def main() -> None:
    run_optimization()


if __name__ == "__main__":
    main()