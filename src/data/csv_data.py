import pandas as pd
from config import coin_dir


def read_csv(coin: str, timeframe: str, col_names: list = ["close"]) -> pd.DataFrame:
    """
    Reads the csv file of a coin and returns it as a pandas DataFrame

    Parameters
    ----------
    coin : str
        The name of the coin, e.g. "BTC"
    timeframe : str
        The timeframe of the data, e.g. "1m"
    col_names : list, optional
        The column name in the .csv file as list, by default ["close"]

    Returns
    -------
    pd.DataFrame
        The csv file as a pandas DataFrame, with "date" as index
    """

    df = pd.read_csv(f"{coin_dir}/{coin}/{coin}USDT_{timeframe}.csv")

    # Set date as index
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)

    return df[col_names]
