from __future__ import annotations

from pathlib import Path

import polars as pl

from features.feature_store import FeatureStore
from pipeline.time_index import WINDOW_SIZE, build_test_indices, build_week_keys
from utils.vif import calculate_vif_fast, stepwise_vif_prune


def _get_history_slice(
    df: pl.DataFrame,
    current_idx: int,
    weeks: int,
) -> pl.DataFrame:
    if weeks == -1:
        return df.slice(0, current_idx)

    history_bars = max(weeks, 0) * WINDOW_SIZE
    if history_bars <= 0:
        return df.slice(0, current_idx)

    history_bars = min(history_bars, current_idx)
    start_idx = current_idx - history_bars
    return df.slice(start_idx, history_bars)


def _vif_pruned_features(
    df: pl.DataFrame,
    max_vif: float,
    bse: bool,
) -> list[str]:
    if "target_return_1w" in df.columns:
        df = df.drop("target_return_1w")

    if df.is_empty():
        return []

    if not bse:
        vif_df = calculate_vif_fast(df)
        if vif_df.is_empty():
            return []
        return (
            vif_df.filter(pl.col("VIF") <= max_vif)
            .select("feature")
            .to_series()
            .to_list()
        )

    return stepwise_vif_prune(df, max_vif=max_vif)


def write_weekly_vif_pruned_features(
    assets_dict: dict[str, pl.DataFrame],
    weeks: int,
    max_vif: float = 10.0,
    bse: bool = False,
    feature_store: FeatureStore | None = None,
    read_features: bool = True,
) -> None:
    """
    Compute weekly VIF-pruned feature lists and update each asset's feature set JSON.

    Args:
        assets_dict: Mapping of asset symbol to feature DataFrame.
        weeks: Number of historical weeks to use for VIF. Use -1 for all history.
        max_vif: Maximum VIF threshold; features above this are excluded.
        bse: When true, apply backward stepwise elimination (iterative VIF).
        feature_store: Optional FeatureStore for reading/writing feature sets.
        read_features: When true, use the current feature set as input.
    """
    if not assets_dict:
        raise ValueError("assets_dict is empty")

    if feature_store is None:
        feature_store = FeatureStore(Path(__file__).resolve().parent)
    reference_df = next(iter(assets_dict.values()))
    reference_df = reference_df.sort("timestamp")
    test_indices = build_test_indices(reference_df)
    week_keys = build_week_keys(len(test_indices))

    for symbol, df in assets_dict.items():
        df = df.sort("timestamp")
        weekly_features: dict[str, list[str]] = {}
        resolved = feature_store.resolve_weekly_features(
            symbol,
            df,
            week_keys,
            read_existing=read_features,
        )

        for week_idx, current_idx in enumerate(test_indices, start=1):
            week_key = f"week{week_idx}"
            print(f"Processing {symbol} {week_key}/{len(test_indices)}")
            window_df = _get_history_slice(df, current_idx, weeks)
            feature_cols = resolved.get(week_key, [])
            if feature_cols:
                selected_cols = list(feature_cols)
                if "target_return_1w" in window_df.columns:
                    selected_cols.append("target_return_1w")
                window_df = window_df.select(selected_cols)

            if "target_return_1w" in window_df.columns:
                window_df = window_df.filter(
                    pl.col("target_return_1w").is_not_null()
                )

            features = _vif_pruned_features(
                window_df,
                max_vif=max_vif,
                bse=bse,
            )
            weekly_features[week_key] = features

        feature_store.save_feature_set(symbol, weekly_features)
