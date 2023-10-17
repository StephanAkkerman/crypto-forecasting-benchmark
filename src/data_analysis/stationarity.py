from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

# Local imports
from config import all_coins, timeframes, statistics_dir
from data.csv_data import get_data


def stationarity_test(
    data_type: str = "log returns", file_name: str = "stationarity_test"
):
    """
    Performs the Augmented Dickey-Fuller and KPSS test on the data and saves the results to an Excel file.

    Parameters
    ----------
    data_type : str
        Options are: "close", "returns" "log returns", and "scaled", by default "log returns"
    file_name : str, optional
        The name for the file to be saved in /data/tests/, by default "adf_test"
    """
    results = pd.DataFrame()

    for coin in tqdm(all_coins):
        for time in timeframes:
            for df in get_data(coin, time, data_type):
                _, p_val, _, _, _, _ = adfuller(df)
                _, p_val2, _, _ = kpss(df)
                adf_dict = {
                    "Coin": coin,
                    "Time": time,
                    "adf p-val": p_val,
                    "kpss p-val": p_val2,
                }

                # Use concat instead of append to avoid the warning
                results = pd.concat(
                    [results, pd.DataFrame(adf_dict, index=[0])],
                    axis=0,
                    ignore_index=True,
                )

    # Write to Excel
    if file_name != "":
        results.to_excel(f"{statistics_dir}/{file_name}.xlsx")

    # Show the coins that are stationary
    adf_significant = results[results["adf p-val"] < 0.05]
    kpss_significant = results[results["kpss p-val"] > 0.05]
    print(adf_significant, len(adf_significant))
    print(kpss_significant, len(kpss_significant))


def plot_price(crypto: str = "BTC", timeframe: str = "1d"):
    """
    Shows a plot of the price and returns of the crypto.

    Parameters
    ----------
    crypto : str, optional
        The symbol of the cryptocurrency, by default "BTC"
    timeframe : str, optional
        The time frame to use, by default "1d"
    """

    df = get_data(crypto, timeframe, "close")
    df_diff = get_data(crypto, timeframe, "returns")

    _, axs = plt.subplots(2, 1, figsize=(12, 8))

    df["close"].plot(ax=axs[0], label="Price")
    df_diff["close"].plot(ax=axs[1], label="Returns")

    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")

    axs[0].set_ylabel("Price in USD")
    axs[1].set_ylabel("Returns in USD")

    axs[0].set_xlabel("")
    axs[1].set_xlabel("Date")

    plt.show()
