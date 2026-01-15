"""

Technical indicators implementation.
"""

import polars as pl
import polars_talib as ptl


# Map of timeframes to minutes
tf_map = {
    '5m': 5,
    '15m': 15,
    '30m': 30,
    '1h': 60
}

BASE_TIMEFRAME = "5m"


def _tf_factor(tf: str) -> int:
    minutes = tf_map.get(tf)
    if minutes is None:
        raise ValueError(f"Unsupported timeframe '{tf}'")
    base_minutes = tf_map[BASE_TIMEFRAME]
    if minutes % base_minutes != 0:
        raise ValueError(
            f"Timeframe '{tf}' must be a multiple of {BASE_TIMEFRAME}"
        )
    return minutes // base_minutes


def add_atr(df: pl.DataFrame, tf: str, period: int) -> pl.DataFrame:
    # 1. Validation
    factor = _tf_factor(tf)
    if period <= 0:
        raise ValueError("period must be a positive integer")

    effective_period = period * factor
    atr_expr = ptl.atr(
        pl.col("high"),
        pl.col("low"),
        pl.col("close"),
        timeperiod=effective_period)
    col_name = f"atr_{period}_{tf}"
    return df.with_columns(atr_expr.alias(col_name))


def add_rsi(df: pl.DataFrame, tf: str, period: int) -> pl.DataFrame:
    # 1. Validation
    factor = _tf_factor(tf)
    if period <= 0:
        raise ValueError("period must be a positive integer")

    effective_period = period * factor
    rsi_expr = ptl.rsi(pl.col("close"), timeperiod=effective_period)

    col_name = f"rsi_{period}_{tf}"
    return df.with_columns(rsi_expr.alias(col_name))


def add_willr(df: pl.DataFrame, tf: str, period: int) -> pl.DataFrame:
    # 1. Validation
    factor = _tf_factor(tf)
    if period <= 0:
        raise ValueError("period must be a positive integer")

    effective_period = period * factor
    willr_expr = ptl.willr(
        pl.col("high"),
        pl.col("low"),
        pl.col("close"),
        timeperiod=effective_period
    )

    col_name = f"willr_{period}_{tf}"
    return df.with_columns(willr_expr.alias(col_name))
