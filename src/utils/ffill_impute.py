import polars as pl

VALUE_COLS = ["open", "high", "low", "close", "volume"]

def ffill_impute(lf: pl.LazyFrame, every: str = "5m") -> tuple[pl.LazyFrame, int]:
    base = (
        lf.select(["timestamp", *VALUE_COLS])
          .with_columns(
              pl.col("timestamp").cast(pl.Datetime("ms")),
              *[pl.col(c).cast(pl.Float64) for c in VALUE_COLS],
          )
          .unique(subset="timestamp", keep="first")
          .sort("timestamp")
    )

    # Derive start/end from the data (small collect)
    bounds = base.select(
        pl.col("timestamp").min().alias("start"),
        pl.col("timestamp").max().alias("end"),
    ).collect()
    start = bounds["start"][0]
    end = bounds["end"][0]

    full_index = pl.datetime_range(
        start, end, interval=every, time_unit="ms", eager=True
    )

    skeleton = pl.DataFrame({"timestamp": full_index}).lazy()

    joined = skeleton.join(base, on="timestamp", how="left")

    imputed_count = int(
        joined.select(pl.col("open").is_null().sum()).collect().item()
    )

    out = (
        joined.with_columns(
            pl.col(VALUE_COLS).fill_null(strategy="forward")  # includes volume now
        )
        .with_columns(pl.col("timestamp").dt.timestamp("ms").cast(pl.Int64()))
        .select(["timestamp", *VALUE_COLS])
    )

    return out, imputed_count