import pandas as pd
from config import coin_dir


def read_csv(coin: str, timeframe: str, col_names: list = ["close"]):
    df = pd.read_csv(f"{coin_dir}/{coin}/{coin}USDT_{timeframe}.csv")

    # Set date as index
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)

    return df[col_names]
