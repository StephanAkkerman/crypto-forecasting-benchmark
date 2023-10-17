import pandas as pd

import config


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

    df = pd.read_csv(f"{config.coin_dir}/{coin}/{coin}USDT_{timeframe}.csv")

    # Set date as index
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)

    return df[col_names]


def get_data(coin: str, time_frame: str, data_type: str = "log returns"):
    if data_type != "scaled":
        if data_type == "returns":
            return [read_csv(coin, time_frame, col_names=["close"]).diff().dropna()]
        else:
            return [read_csv(coin, time_frame, col_names=[data_type]).dropna()]
    else:
        data = []

        for period in range(config.n_periods):
            train = pd.read_csv(
                f"{config.model_output_dir}/{config.scaled_pred}/ARIMA/{coin}/{time_frame}/train_{period}.csv",
                index_col=0,
            )

            # Convert index column to datetime
            train.index = pd.to_datetime(train.index)

            test = pd.read_csv(
                f"{config.model_output_dir}/{config.scaled_pred}/ARIMA/{coin}/{time_frame}/test_{period}.csv",
                index_col=0,
            )
            test.index = pd.to_datetime(test.index)

            data.append(pd.concat([train, test], axis=0))
        return data
