import asyncio
import ccxt.async_support as ccxt
import polars as pl

from src.exchange.get_ohlcv import fetch_and_write_ohlcv
from src.file_ops.ohlcv import read_ohlcv

exchange = ccxt.coinbaseadvanced()
exchange.enableRateLimit = True

fetch_sem = asyncio.Semaphore(3)

SYMBOLS = ["ADA", "AVAX", "BCH", "BTC", "DOGE", "ETH", "LINK", "LTC", "SOL", "XRP"]
QUOTE = "USDC"

timeframe = '5m'
since = 1704092400000
end = 1767250799000

async def process_data(symbol: str):
    async with fetch_sem:
        await fetch_and_write_ohlcv(
        exchange=exchange,
        symbol=f"{symbol}-USDC",
        timeframe=timeframe,
        since_ms=since,
        end_ms=end,
    )

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
        lf = pl.scan_parquet(f"data_cleaned/{symbol}-USDC.parquet")
        df = lf.collect()
        print(f"{symbol} {df.shape}")
        total_rows += df.shape[0]
    print(f"Total number of rows across all symbols: {total_rows}")

if __name__ == "__main__":
    read()