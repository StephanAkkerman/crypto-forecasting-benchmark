import numpy as np
import pandas as pd
from binance.client import Client

from config import coin_dir

# Initialize the Client
client = Client()


def fetchData(symbol="BTC", amount=1, timeframe="1d", as_csv=False, file_name=None):
    """
    Pandas DataFrame with the latest OHLCV data from Binance.

    Parameters
    --------------
    symbol : string, combine the coin you want to get with the pair. For instance "BTC" for BTC/USDT.
    amount : int, the amount of rows that should be returned divided by 1000. For instance 2 will return 1000 rows.
    timeframe : string, the timeframe according to the Binance API. For instance "4h" for the 4 hour candles.
    All the timeframe options are: '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'
    """
    # https://python-binance.readthedocs.io/en/latest/binance.html#binance.client.Client.get_klines

    # ms calculations based on: http://convertlive.com/nl/u/converteren/minuten/naar/milliseconden
    # 1m = 60000 ms
    if timeframe == "1m":
        diff = 60000
    elif timeframe == "3m":
        diff = 3 * 60000
    elif timeframe == "5m":
        diff = 5 * 60000
    elif timeframe == "15m":
        diff = 15 * 60000
    elif timeframe == "30m":
        diff = 30 * 60000

    # 1h = 3600000 ms
    elif timeframe == "1h":
        diff = 3600000
    elif timeframe == "2h":
        diff = 2 * 3600000
    elif timeframe == "4h":
        diff = 4 * 3600000
    elif timeframe == "6h":
        diff = 6 * 3600000
    elif timeframe == "8h":
        diff = 8 * 3600000
    elif timeframe == "12h":
        diff = 12 * 3600000

    # 1d = 86400000 ms
    elif timeframe == "1d":
        diff = 86400000
    elif timeframe == "3d":
        diff = 3 * 86400000
    elif timeframe == "1W":
        diff = 604800000
    elif timeframe == "1M":
        diff = 2629800000

    else:
        print(f"{timeframe} is an invalid timeframe")
        return

    # Add USDT to the symbol
    full_symbol = symbol + "USDT"

    # Get current time, by getting the latest candle
    candleList = client.get_klines(symbol=full_symbol, limit=1000, interval=timeframe)

    if amount > 1:
        end = candleList[-1][0]
        # Get the amount of data specified by amount parameter
        for _ in range(amount):
            # Make the list from oldest to newest
            candleList = (
                client.get_klines(
                    symbol=full_symbol, limit=1000, interval=timeframe, endTime=end
                )
                + candleList
            )

            # Calculate the end point by using the difference in ms per candle
            end = end - diff * 1000

    df = pd.DataFrame(candleList)

    # Rename columns
    # https://python-binance.readthedocs.io/en/latest/binance.html#binance.client.Client.get_klines
    new_columns = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]

    df.columns = new_columns

    # Convert time in ms to datetime
    df["date"] = pd.to_datetime(df["date"], unit="ms")

    # The default values are string, so convert these to numeric values
    df["open"] = pd.to_numeric(df["open"])
    df["high"] = pd.to_numeric(df["high"])
    df["low"] = pd.to_numeric(df["low"])
    df["close"] = pd.to_numeric(df["close"])
    df["volume"] = pd.to_numeric(df["volume"])
    df["usdt_vol"] = df["volume"] * df["close"]

    # Calculate log returns and volatility
    df["log returns"] = np.log(df["close"]).diff().dropna()
    df["volatility"] = df["log returns"].rolling(window=30).std() * np.sqrt(30)

    if as_csv:
        if file_name == None:
            file_name = full_symbol + "_" + timeframe + ".csv"

        df.to_csv(f"{coin_dir}/{symbol}/{file_name}", index=False)
        print(f"Succesfully saved {len(df)} rows to {file_name}")

    return df
