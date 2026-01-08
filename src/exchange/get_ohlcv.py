"""
Get ohlcv data, extended.
"""
import asyncio
import contextlib
import math
import os
import shutil
import tempfile
import ccxt.async_support as ccxt
import polars as pl


_TF_TO_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "6h": 360,
    "1d": 1440,
}


def _timeframe_to_minutes(timeframe: str) -> int:
    try:
        return _TF_TO_MINUTES[timeframe]
    except KeyError as e:
        raise ValueError(f"Unsupported timeframe: {timeframe}") from e


def _empty_ohlcv_lazyframe() -> pl.LazyFrame:
    return pl.LazyFrame(
        schema={
            "timestamp": int,
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
        }
    )


async def get_ohlcv(
    exchange: ccxt.coinbaseadvanced,
    symbol: str,
    timeframe: str,
    since_ms: int,
    end_ms: int
    ) -> pl.LazyFrame:

    tf_minutes = _timeframe_to_minutes(timeframe)
    tf_ms = tf_minutes * 60_000

    frames: list[pl.DataFrame] = []
    current_ms = since_ms
    max_limit = 300  # ccxt default

    # Simple progress estimate: (1440 / tf_minutes) * num_days / max_limit
    total_minutes = max(0, (end_ms - since_ms) // 60_000)
    num_days = total_minutes / 1440
    est_total_batches = max(1, math.ceil((1440 / tf_minutes) * num_days / max_limit))
    batch_num = 0

    try:
        # Fetch forward in batches, trimming anything outside [since_ms, end_ms).
        while current_ms < end_ms:
            remaining = (end_ms - current_ms + tf_ms - 1) // tf_ms  # candles needed (end-exclusive)
            limit = int(min(max_limit, remaining))
            if limit <= 0:
                break

            batch_num += 1
            print(f"{symbol} {timeframe}: {batch_num}/{est_total_batches}")

            ohlcv = await exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_ms,
                limit=limit,
            )
            if not ohlcv:
                break

            df = pl.DataFrame(
                ohlcv,
                schema=['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                orient='row',
            ).filter(pl.col("timestamp") < end_ms)

            if df.is_empty():
                break

            frames.append(df)

            last_ms = int(df["timestamp"][-1])
            next_ms = last_ms + tf_ms

            # Guard: if the exchange returns the same last candle again, stop.
            if next_ms <= current_ms:
                break

            current_ms = next_ms
    except Exception as e:
        print(e)
        
    if not frames:
        return _empty_ohlcv_lazyframe()

    return (
        pl.concat(frames, how="vertical")
        .unique(subset=["timestamp"], keep="first")
        .sort("timestamp")
        .lazy()
    )


async def fetch_and_write_ohlcv(
    exchange: ccxt.coinbaseadvanced,
    symbol: str,
    timeframe: str,
    since_ms: int,
    end_ms: int,
    output_path: str | None = None,
) -> str:
    """
    Fetch OHLCV in batches, write each batch to temp parquet files, then merge into a single parquet.
    Temp files are cleaned up at the end.
    """
    from src.file_ops.ohlcv import base_dir

    os.makedirs(base_dir, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(base_dir, f"{symbol}.parquet")

    tf_minutes = _timeframe_to_minutes(timeframe)
    tf_ms = tf_minutes * 60_000
    max_limit = 300

    total_minutes = max(0, (end_ms - since_ms) // 60_000)
    num_days = total_minutes / 1440
    est_total_batches = max(1, math.ceil((1440 / tf_minutes) * num_days / max_limit))

    tmp_dir = tempfile.mkdtemp(prefix=f"ohlcv_tmp_{symbol.replace('/', '-')}_{timeframe}_", dir=base_dir)
    q: asyncio.Queue[tuple[int, pl.DataFrame] | None] = asyncio.Queue(maxsize=2)

    async def _writer() -> list[str]:
        paths: list[str] = []
        while True:
            item = await q.get()
            if item is None:
                q.task_done()
                break

            batch_num, df = item
            batch_path = os.path.join(tmp_dir, f"batch_{batch_num:06d}.parquet")
            await asyncio.to_thread(df.write_parquet, batch_path)
            paths.append(batch_path)
            q.task_done()
        return paths

    writer_task = asyncio.create_task(_writer())
    batch_num = 0
    current_ms = since_ms

    try:
        while current_ms < end_ms:
            remaining = (end_ms - current_ms + tf_ms - 1) // tf_ms
            limit = int(min(max_limit, remaining))
            if limit <= 0:
                break

            batch_num += 1
            print(f"{symbol} {timeframe}: {batch_num}/{est_total_batches}")

            ohlcv = await exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_ms,
                limit=limit,
            )
            if not ohlcv:
                break

            df = pl.DataFrame(
                ohlcv,
                schema=["timestamp", "open", "high", "low", "close", "volume"],
                orient="row",
            ).filter(pl.col("timestamp") < end_ms)

            if df.is_empty():
                break

            # Enqueue for async writing; bounded queue prevents memory ballooning.
            await q.put((batch_num, df))

            last_ms = int(df["timestamp"][-1])
            next_ms = last_ms + tf_ms
            if next_ms <= current_ms:
                break
            current_ms = next_ms

        await q.put(None)
        batch_paths = await writer_task

        if not batch_paths:
            _empty_ohlcv_lazyframe().sink_parquet(output_path)
            return output_path

        (
            pl.scan_parquet(batch_paths)
            .unique(subset=["timestamp"], keep="first")
            .sort("timestamp")
            .sink_parquet(output_path)
        )
        return output_path
    finally:
        if not writer_task.done():
            writer_task.cancel()
            with contextlib.suppress(Exception):
                await writer_task
        shutil.rmtree(tmp_dir, ignore_errors=True)