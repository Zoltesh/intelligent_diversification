import os
import polars as pl

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
base_dir = os.path.join(project_root, "data", "ohlcv")

# Given ohlcv data, write to file
async def write_ohlcv(symbol: str, ohlcv: pl.LazyFrame):
    os.makedirs(base_dir, exist_ok=True)

    file_path = os.path.join(base_dir, f"{symbol}.parquet")
    ohlcv.sink_parquet(file_path)

async def read_ohlcv(symbol: str) -> pl.LazyFrame:
    file_path = os.path.join(base_dir, f"{symbol}.parquet")
    return pl.scan_parquet(file_path)