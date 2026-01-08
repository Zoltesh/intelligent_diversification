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

async def read():
    for symbol in SYMBOLS:
        lf = await read_ohlcv(f"{symbol}-USDC")
        df = lf.collect()
        # Print the shape and the start/end timestamps as datetimes using polars from_epoch
        first_dt = df.select(
            pl.from_epoch("timestamp", time_unit="ms")
        ).head(1).item(0, 0)

        last_dt = df.select(
            pl.from_epoch("timestamp", time_unit="ms")
        ).tail(1).item(0, 0)
        print(f"{symbol} {df.shape} from {first_dt} to {last_dt}")

if __name__ == "__main__":
    asyncio.run(read())