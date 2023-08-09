import numpy as np
import pandas as pd

from config import all_coins, timeframes, coin_dir
from data.binance_data import fetchData


def create_all_data():
    for coin in all_coins:
        for time in timeframes:
            fetchData(symbol=coin, amount=1, timeframe=time, as_csv=True)


def format_TOTAL():
    """
    Adds a date column to the TOTAL.csv file
    Removes the unnecessary data, so it's as long as the BTC data
    Then adds logarithmic returns and the volatility of it
    """

    for time, btc_time in [("1", "1m"), ("15", "15m"), ("240", "4h"), ("1D", "1d")]:
        df = pd.read_csv(f"data/coins/TOTAL/CRYPTOCAP_TOTAL, {time}.csv")

        df["date"] = pd.to_datetime(df["time"], unit="s")

        btc = pd.read_csv(f"data/coins/BTC/BTCUSDT_{btc_time}.csv")

        # Only keeps dates that are in both dataframes
        df = df[df["date"].isin(btc["date"])]
        df.reset_index(drop=True, inplace=True)

        # Make the date the index
        df.set_index("date", inplace=True)

        # Add the log returns
        df["close"] = df["close"].astype(float)
        df["log returns"] = np.log(df["close"]).diff().dropna()

        # Add volatility
        df["volatility"] = df["log returns"].rolling(window=30).std() * np.sqrt(30)

        # Add news columns to csv
        df.to_csv(f"{coin_dir}/TOTAL/TOTAL_{btc_time}.csv", index=True)

        # Make sure they are 1000 rows
        print(btc_time, len(df))
