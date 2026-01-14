import asyncio
import ccxt.async_support as ccxt
import os
import polars as pl

from src.exchange.get_ohlcv import fetch_and_write_ohlcv
from src.file_ops.ohlcv import read_ohlcv

exchange = ccxt.coinbaseadvanced()
exchange.enableRateLimit = True

fetch_sem = asyncio.Semaphore(3)

SYMBOLS = ["ADA", "AVAX", "BCH", "BTC", "DOGE", "ETH", "LINK", "LTC", "SOL", "XRP"]
QUOTE = "USDC"

timeframe = '5m'
since = 1704067200000 # 2024-01-01 00:00:00
end = 1767225300000 # 2025-12-31 23:55:00

async def process_data(symbol: str):
    async with fetch_sem:
        await fetch_and_write_ohlcv(
        exchange=exchange,
        symbol=f"{symbol}-USDC",
        timeframe=timeframe,
        since_ms=since,
        end_ms=end,
    )

def drop_tail():
    # For each parquet, drop any records greater than timestamp == 1767164400000
    for file in os.listdir('src/data/'):
        lf = pl.scan_parquet(f'src/data/{file}')
        df = lf.collect()
        print(len(df))
        df_filtered = df.filter(~(pl.col("timestamp") > 1767164400000))
        print(f'{file}: {len(df_filtered)}')

        # Overwrite the file with the filtered data
        df_filtered.write_parquet(f'src/data/{file}')


async def main():
    coroutines = [process_data(symbol) for symbol in SYMBOLS]

    try:
        await asyncio.gather(*coroutines)
    finally:
        await exchange.close()

def read():
    # Also print total number of rows across all symbols
    total_rows = 0
    for symbol in SYMBOLS:
        lf = pl.scan_parquet(f"src/data/{symbol}-USDC.parquet")
        df = lf.collect()
        perc = (210_528 - len(df))/210_528 * 100
        print(f'{symbol} {len(df)} {perc:.2f}% {df['timestamp'].min()} {df['timestamp'].max()}')
        total_rows += df.shape[0]
    print(f"Total number of rows across all symbols: {total_rows}")


if __name__ == "__main__":
    read()