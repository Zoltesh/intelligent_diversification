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


def add_adx(df: pl.DataFrame, tf: str, period: int) -> pl.DataFrame:
    # 1. Validation
    factor = _tf_factor(tf)
    if period <= 0:
        raise ValueError("period must be a positive integer")

    effective_period = period * factor
    adx_expr = ptl.adx(
        pl.col("high"),
        pl.col("low"),
        pl.col("close"),
        timeperiod=effective_period
    )
    col_name = f"adx_{period}_{tf}"
    return df.with_columns(adx_expr.alias(col_name))


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
        timeperiod=effective_period
    )
    col_name = f"atr_{period}_{tf}"
    return df.with_columns(atr_expr.alias(col_name))


def add_bbands(
    df: pl.DataFrame,
    tf: str,
    period: int,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,
    matype: int = 0,
) -> pl.DataFrame:
    # 1. Validation
    factor = _tf_factor(tf)
    if period <= 0:
        raise ValueError("period must be a positive integer")
    if nbdevup <= 0 or nbdevdn <= 0:
        raise ValueError("nbdevup and nbdevdn must be positive")

    effective_period = period * factor
    bbands_expr = ptl.bbands(
        pl.col("close"),
        timeperiod=effective_period,
        nbdevup=nbdevup,
        nbdevdn=nbdevdn,
        matype=matype,
    )

    upper_col = f"bbands_upper_{period}_{tf}"
    middle_col = f"bbands_middle_{period}_{tf}"
    lower_col = f"bbands_lower_{period}_{tf}"
    return df.with_columns(
        bbands_expr.struct.field("upperband").alias(upper_col),
        bbands_expr.struct.field("middleband").alias(middle_col),
        bbands_expr.struct.field("lowerband").alias(lower_col),
    )


def add_macd(
    df: pl.DataFrame,
    tf: str,
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
) -> pl.DataFrame:
    # 1. Validation
    factor = _tf_factor(tf)
    if fastperiod <= 0 or slowperiod <= 0 or signalperiod <= 0:
        raise ValueError("periods must be positive integers")
    if fastperiod >= slowperiod:
        raise ValueError("fastperiod must be less than slowperiod")

    effective_fast = fastperiod * factor
    effective_slow = slowperiod * factor
    effective_signal = signalperiod * factor
    macd_expr = ptl.macd(
        pl.col("close"),
        fastperiod=effective_fast,
        slowperiod=effective_slow,
        signalperiod=effective_signal,
    )

    macd_col = f"macd_{fastperiod}_{slowperiod}_{signalperiod}_{tf}"
    signal_col = f"macd_signal_{fastperiod}_{slowperiod}_{signalperiod}_{tf}"
    hist_col = f"macd_hist_{fastperiod}_{slowperiod}_{signalperiod}_{tf}"
    return df.with_columns(
        macd_expr.struct.field("macd").alias(macd_col),
        macd_expr.struct.field("macdsignal").alias(signal_col),
        macd_expr.struct.field("macdhist").alias(hist_col),
    )


def add_obv(df: pl.DataFrame, tf: str) -> pl.DataFrame:
    # 1. Validation
    _tf_factor(tf)

    obv_expr = ptl.obv(
        pl.col("close"),
        pl.col("volume"),
    )
    col_name = f"obv_{tf}"
    return df.with_columns(obv_expr.alias(col_name))


def add_cci(df: pl.DataFrame, tf: str, period: int) -> pl.DataFrame:
    # 1. Validation
    factor = _tf_factor(tf)
    if period <= 0:
        raise ValueError("period must be a positive integer")

    effective_period = period * factor
    cci_expr = ptl.cci(
        pl.col("high"),
        pl.col("low"),
        pl.col("close"),
        timeperiod=effective_period,
    )
    col_name = f"cci_{period}_{tf}"
    return df.with_columns(cci_expr.alias(col_name))


def add_cmo(df: pl.DataFrame, tf: str, period: int) -> pl.DataFrame:
    # 1. Validation
    factor = _tf_factor(tf)
    if period <= 0:
        raise ValueError("period must be a positive integer")

    effective_period = period * factor
    cmo_expr = ptl.cmo(
        pl.col("close"),
        timeperiod=effective_period,
    )
    col_name = f"cmo_{period}_{tf}"
    return df.with_columns(cmo_expr.alias(col_name))


def add_mom(df: pl.DataFrame, tf: str, period: int) -> pl.DataFrame:
    # 1. Validation
    factor = _tf_factor(tf)
    if period <= 0:
        raise ValueError("period must be a positive integer")

    effective_period = period * factor
    mom_expr = ptl.mom(
        pl.col("close"),
        timeperiod=effective_period,
    )
    col_name = f"mom_{period}_{tf}"
    return df.with_columns(mom_expr.alias(col_name))


def add_mfi(df: pl.DataFrame, tf: str, period: int) -> pl.DataFrame:
    # 1. Validation
    factor = _tf_factor(tf)
    if period <= 0:
        raise ValueError("period must be a positive integer")

    effective_period = period * factor
    mfi_expr = ptl.mfi(
        pl.col("high"),
        pl.col("low"),
        pl.col("close"),
        pl.col("volume"),
        timeperiod=effective_period,
    )
    col_name = f"mfi_{period}_{tf}"
    return df.with_columns(mfi_expr.alias(col_name))


def add_trix(df: pl.DataFrame, tf: str, period: int) -> pl.DataFrame:
    # 1. Validation
    factor = _tf_factor(tf)
    if period <= 0:
        raise ValueError("period must be a positive integer")

    effective_period = period * factor
    trix_expr = ptl.trix(
        pl.col("close"),
        timeperiod=effective_period,
    )
    col_name = f"trix_{period}_{tf}"
    return df.with_columns(trix_expr.alias(col_name))


def add_wma(df: pl.DataFrame, tf: str, period: int) -> pl.DataFrame:
    # 1. Validation
    factor = _tf_factor(tf)
    if period <= 0:
        raise ValueError("period must be a positive integer")

    effective_period = period * factor
    wma_expr = ptl.wma(
        pl.col("close"),
        timeperiod=effective_period,
    )
    col_name = f"wma_{period}_{tf}"
    return df.with_columns(wma_expr.alias(col_name))


def add_roc(df: pl.DataFrame, tf: str, period: int) -> pl.DataFrame:
    # 1. Validation
    factor = _tf_factor(tf)
    if period <= 0:
        raise ValueError("period must be a positive integer")

    effective_period = period * factor
    roc_expr = ptl.roc(
        pl.col("close"),
        timeperiod=effective_period,
    )
    col_name = f"roc_{period}_{tf}"
    return df.with_columns(roc_expr.alias(col_name))


def add_rsi(df: pl.DataFrame, tf: str, period: int) -> pl.DataFrame:
    # 1. Validation
    factor = _tf_factor(tf)
    if period <= 0:
        raise ValueError("period must be a positive integer")

    effective_period = period * factor
    rsi_expr = ptl.rsi(
        pl.col("close"),
        timeperiod=effective_period,
        )

    col_name = f"rsi_{period}_{tf}"
    return df.with_columns(rsi_expr.alias(col_name))


def add_stoch(df: pl.DataFrame, tf: str, period: int) -> pl.DataFrame:
    # 1. Validation
    factor = _tf_factor(tf)
    if period <= 0:
        raise ValueError("period must be a positive integer")

    effective_period = period * factor
    stoch_expr = ptl.stoch(pl.col("high"), pl.col("low"), pl.col("close"), timeperiod=effective_period)
    col_name = f"stoch_{period}_{tf}"
    return df.with_columns(stoch_expr.alias(col_name))


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


def add_indicators(df: pl.DataFrame, tf: str, period: int) -> pl.DataFrame:
    factor = _tf_factor(tf)
    if period <= 0:
        raise ValueError("period must be a positive integer")

    effective_period = period * factor

    return df.with_columns(
        ptl.cci(
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            timeperiod=effective_period,
        ).alias(f"cci_{period}_{tf}"),
        ptl.cmo(
            pl.col("close"),
            timeperiod=effective_period,
        ).alias(f"cmo_{period}_{tf}"),
        ptl.mom(
            pl.col("close"),
            timeperiod=effective_period,
        ).alias(f"mom_{period}_{tf}"),
        ptl.mfi(
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            pl.col("volume"),
            timeperiod=effective_period,
        ).alias(f"mfi_{period}_{tf}"),
        ptl.trix(
            pl.col("close"),
            timeperiod=effective_period,
        ).alias(f"trix_{period}_{tf}"),
        ptl.wma(
            pl.col("close"),
            timeperiod=effective_period,
        ).alias(f"wma_{period}_{tf}"),
        ptl.roc(
            pl.col("close"),
            timeperiod=effective_period,
        ).alias(f"roc_{period}_{tf}"),
        ptl.adx(
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            timeperiod=effective_period,
        ).alias(f"adx_{period}_{tf}"),
        ptl.atr(
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            timeperiod=effective_period,
        ).alias(f"atr_{period}_{tf}"),
        ptl.rsi(
            pl.col("close"),
            timeperiod=effective_period,
        ).alias(f"rsi_{period}_{tf}"),
        ptl.willr(
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            timeperiod=effective_period,
        ).alias(f"willr_{period}_{tf}"),
    )
