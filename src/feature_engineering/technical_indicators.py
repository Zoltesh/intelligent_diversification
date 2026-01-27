"""

Technical indicators implementation.
"""

import json
from pathlib import Path

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
DEFAULT_CONFIG_PATH = Path(__file__).with_name("indicator_config.json")


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


def _validate_timeframes(timeframes: list[str]) -> None:
    if not timeframes:
        raise ValueError("timeframes must be a non-empty list")
    for tf in timeframes:
        _tf_factor(tf)


def load_indicator_config(config_path: Path | str | None = None) -> dict:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Indicator config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or "indicators" not in config:
        raise ValueError("Config must be a dict with an 'indicators' key")
    return config


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
    matype: int = 1,
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


def add_ppo(
    df: pl.DataFrame,
    tf: str,
    fastperiod: int = 12,
    slowperiod: int = 26,
    matype: int = 0,
) -> pl.DataFrame:
    # 1. Validation
    factor = _tf_factor(tf)
    if fastperiod <= 0 or slowperiod <= 0:
        raise ValueError("periods must be positive integers")
    if fastperiod >= slowperiod:
        raise ValueError("fastperiod must be less than slowperiod")

    effective_fast = fastperiod * factor
    effective_slow = slowperiod * factor
    ppo_expr = ptl.ppo(
        pl.col("close"),
        fastperiod=effective_fast,
        slowperiod=effective_slow,
        matype=matype,
    )
    col_name = f"ppo_{fastperiod}_{slowperiod}_{matype}_{tf}"
    return df.with_columns(ppo_expr.alias(col_name))


def add_aroon(df: pl.DataFrame, tf: str, period: int) -> pl.DataFrame:
    # 1. Validation
    factor = _tf_factor(tf)
    if period <= 0:
        raise ValueError("period must be a positive integer")

    effective_period = period * factor
    aroon_expr = ptl.aroon(
        pl.col("high"),
        pl.col("low"),
        timeperiod=effective_period,
    )
    down_col = f"aroon_down_{period}_{tf}"
    up_col = f"aroon_up_{period}_{tf}"
    return df.with_columns(
        aroon_expr.struct.field("aroondown").alias(down_col),
        aroon_expr.struct.field("aroonup").alias(up_col),
    )


def add_kama(df: pl.DataFrame, tf: str, period: int) -> pl.DataFrame:
    # 1. Validation
    factor = _tf_factor(tf)
    if period <= 0:
        raise ValueError("period must be a positive integer")

    effective_period = period * factor
    kama_expr = ptl.kama(
        pl.col("close"),
        timeperiod=effective_period,
    )
    col_name = f"kama_{period}_{tf}"
    return df.with_columns(kama_expr.alias(col_name))


def add_adosc(
    df: pl.DataFrame,
    tf: str,
    fastperiod: int = 3,
    slowperiod: int = 10,
) -> pl.DataFrame:
    # 1. Validation
    factor = _tf_factor(tf)
    if fastperiod <= 0 or slowperiod <= 0:
        raise ValueError("periods must be positive integers")
    if fastperiod >= slowperiod:
        raise ValueError("fastperiod must be less than slowperiod")

    effective_fast = fastperiod * factor
    effective_slow = slowperiod * factor
    adosc_expr = ptl.adosc(
        pl.col("high"),
        pl.col("low"),
        pl.col("close"),
        pl.col("volume"),
        fastperiod=effective_fast,
        slowperiod=effective_slow,
    )
    col_name = f"adosc_{fastperiod}_{slowperiod}_{tf}"
    return df.with_columns(adosc_expr.alias(col_name))


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


def add_indicators(
    df: pl.DataFrame,
    config_path: Path | str | None = None,
) -> pl.DataFrame:
    df = df.sort("timestamp")
    config = load_indicator_config(config_path)
    indicators = config.get("indicators", {})
    if not indicators:
        raise ValueError("Config has no indicators defined")

    registry = {
        "adx": ("period", add_adx),
        "atr": ("period", add_atr),
        "bbands": ("params", add_bbands),
        "macd": ("params", add_macd),
        "ppo": ("params", add_ppo),
        "obv": ("tf_only", add_obv),
        "aroon": ("period", add_aroon),
        "cci": ("period", add_cci),
        "cmo": ("period", add_cmo),
        "mom": ("period", add_mom),
        "mfi": ("period", add_mfi),
        "trix": ("period", add_trix),
        "kama": ("period", add_kama),
        "wma": ("period", add_wma),
        "roc": ("period", add_roc),
        "rsi": ("period", add_rsi),
        "stoch": ("period", add_stoch),
        "willr": ("period", add_willr),
        "adosc": ("params", add_adosc),
    }

    for name, settings in indicators.items():
        if name not in registry:
            raise ValueError(f"Unsupported indicator '{name}' in config")
        if not isinstance(settings, dict):
            raise ValueError(f"Indicator settings must be a dict: '{name}'")

        mode, func = registry[name]
        timeframes = settings.get("timeframes", [])
        _validate_timeframes(timeframes)

        if mode == "period":
            periods = settings.get("periods", [])
            if not periods:
                raise ValueError(f"Indicator '{name}' requires 'periods'")
            for tf in timeframes:
                for period in periods:
                    df = func(df=df, tf=tf, period=period)
        elif mode == "params":
            params_list = settings.get("params", [])
            if not params_list:
                raise ValueError(f"Indicator '{name}' requires 'params'")
            for tf in timeframes:
                for params in params_list:
                    if not isinstance(params, dict):
                        raise ValueError(
                            f"Indicator '{name}' params must be dicts"
                        )
                    df = func(df=df, tf=tf, **params)
        else:
            for tf in timeframes:
                df = func(df=df, tf=tf)

    return df
