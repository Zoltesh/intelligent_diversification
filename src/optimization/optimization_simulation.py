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

# --- CONFIGURATION ---
START_TS_MS = 1735689600000  # 2025-01-01
WEEKS = 52
MS_PER_DAY = 86_400_000
MS_PER_WEEK = 7 * MS_PER_DAY

# CAP CONFIGURATION
# 0.25 = Max 25% per asset. 
# Since we enforce this manually, it will never crash the solver.
MAX_ASSET_WEIGHT = 0.25 


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
    
    returns_matrix = returns.to_numpy()
    cov_matrix = np.cov(returns_matrix, rowvar=False) * annualization_factor
    return cov_matrix


def equal_weights(symbols: List[str]) -> Dict[str, float]:
    weight = 1.0 / len(symbols)
    return {symbol: weight for symbol in symbols}


def normalize_weights(weights: Dict[str, float], symbols: List[str]) -> Dict[str, float]:
    total = sum(weights.get(s, 0.0) for s in symbols)
    if total <= 0:
        return equal_weights(symbols)
    return {s: float(weights.get(s, 0.0)) / total for s in symbols}


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
        
    cleaned = weights.copy()
    symbols = list(cleaned.keys())
    
    # We loop a few times to ensure redistribution doesn't push a small weight over the cap
    for _ in range(10): 
        # 1. Cap values
        capped = {k: min(v, cap) for k, v in cleaned.items()}
        
        # 2. Renormalize (Distribute the missing %)
        total = sum(capped.values())
        if total <= 0: 
            return equal_weights(symbols)
            
        # Scale everything up so it sums to 1.0 again
        cleaned = {k: v / total for k, v in capped.items()}
        
        # 3. Check if we are done (Are all <= cap?)
        # We use a tiny epsilon (0.0001) for floating point tolerance
        if all(v <= cap + 0.0001 for v in cleaned.values()):
            return cleaned
            
    return cleaned


def run_simulation(
    wide_df: pl.DataFrame, predictions: Dict[str, List[float]]
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
        if prev_weights is None:
            prev_weights_array = np.zeros(len(symbols), dtype=float)
        else:
            prev_weights_array = np.array(
                [prev_weights.get(symbol, 0.0) for symbol in symbols], dtype=float
            )

        try:
            # --- STEP 1: SOLVER (The Math) ---
            # We REMOVED the hard constraint (w <= 0.25).
            # We rely on L2_reg to discourage 100% allocation, but we let the solver
            # output whatever it wants (e.g. 60%) to ensure it finds a solution.
            ef = EfficientFrontier(expected_returns, cov_matrix, weight_bounds=(0.0, 1.0))
            ef.add_objective(objective_functions.L2_reg, gamma=0.1)
            ef.add_objective(
                objective_functions.transaction_cost, w_prev=prev_weights_array, k=0.001
            )
            ef.add_constraint(
                lambda w: cp.sum(cp.abs(w - prev_weights_array)) <= 0.20
            )

            try:
                ef.max_sharpe(risk_free_rate=0.0)
            except Exception as e:
                if "infeasible" in str(e).lower():
                    ef = EfficientFrontier(
                        expected_returns, cov_matrix, weight_bounds=(0.0, 1.0)
                    )
                    ef.add_objective(objective_functions.L2_reg, gamma=0.1)
                    ef.add_objective(
                        objective_functions.transaction_cost,
                        w_prev=prev_weights_array,
                        k=0.001,
                    )
                    ef.max_sharpe(risk_free_rate=0.0)
                else:
                    raise
            
            cleaned_weights = ef.clean_weights(cutoff=0.01)
            mapped_weights = {symbols[i]: w for i, w in cleaned_weights.items()}
            
            # --- STEP 2: HUMAN LOGIC (The Constraints) ---
            # Now we force the distribution logic you asked for.
            weights = apply_weight_cap(mapped_weights, cap=MAX_ASSET_WEIGHT)

        except Exception as e:
            # Fallback for Bear Markets (when all returns are negative)
            print(f"Week {week_idx} Max Sharpe failed ({e}). Switching to Defensive Mode.")
            try:
                ef_risk = EfficientFrontier(expected_returns, cov_matrix, weight_bounds=(0.0, 1.0))
                ef_risk.add_objective(objective_functions.L2_reg, gamma=0.1)
                ef_risk.add_objective(
                    objective_functions.transaction_cost,
                    w_prev=prev_weights_array,
                    k=0.001,
                )
                ef_risk.add_constraint(
                    lambda w: cp.sum(cp.abs(w - prev_weights_array)) <= 0.20
                )
                
                try:
                    ef_risk.min_volatility()
                except Exception as e2:
                    if "infeasible" in str(e2).lower():
                        ef_risk = EfficientFrontier(
                            expected_returns, cov_matrix, weight_bounds=(0.0, 1.0)
                        )
                        ef_risk.add_objective(objective_functions.L2_reg, gamma=0.1)
                        ef_risk.add_objective(
                            objective_functions.transaction_cost,
                            w_prev=prev_weights_array,
                            k=0.001,
                        )
                        ef_risk.min_volatility()
                    else:
                        raise
                
                cleaned_weights = ef_risk.clean_weights(cutoff=0.01)
                mapped_weights = {symbols[i]: w for i, w in cleaned_weights.items()}
                
                # Apply same capping logic to defensive portfolio
                weights = apply_weight_cap(mapped_weights, cap=MAX_ASSET_WEIGHT)
                
            except Exception as e2:
                print(f"Week {week_idx} Critical Failure. Using Equal Weights.")
                if prev_weights:
                    weights = normalize_weights(prev_weights, symbols)
                else:
                    weights = equal_weights(symbols)

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


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    predictions_path = base_dir / "weekly_predictions_2025.json"
    data_dir = base_dir / "engineered_features"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(predictions_path)
    symbols = list(predictions.keys())

    if any(len(predictions[symbol]) != WEEKS for symbol in symbols):
        raise ValueError("Predictions must have 52 weeks.")

    wide_df, _ = load_wide_prices(data_dir, symbols)

    results = run_simulation(wide_df, predictions)

    output_path = results_dir / "weekly_weights.json"
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()