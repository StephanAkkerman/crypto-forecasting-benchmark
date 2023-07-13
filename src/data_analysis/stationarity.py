import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, kpss

# Local imports
from data.vars import all_coins, timeframes
from data.csv_data import read_csv

def stationarity_tests():
    """
    Performs the Augmented Dickey-Fuller and KPSS tests on the data and saves the results to an Excel file.
    """
    plot_price()
    
    for diff in [False, True]:
        adf_test(diff)
        kpss_test(diff)

def adf_test(diff : bool = False, file_name : str = "adf_test"):
    """
    Performs the Augmented Dickey-Fuller test on the data and saves the results to an Excel file.

    Parameters
    ----------
    diff : bool, optional
        If True then uses returns instead, by default False
    file_name : str, optional
        The name for the file to be saved in /data/tests/, by default "adf_test"
    """
    
    results = pd.DataFrame()

    for coin in all_coins:
        for time in timeframes:
            df = read_csv(coin, time)

            if diff:
                df = df.diff().dropna()
                file_name = f"{file_name}_diff"

            test_stat, p_val, num_lags, num_obs, crit_vals, _ = adfuller(df)
            first_crit = crit_vals["1%"]
            second_crit = crit_vals["5%"]
            third_crit = crit_vals["10%"]
            adf_dict = {
                "Coin": coin,
                "Time": time,
                "ADF Test Statistic": test_stat,
                "p-value": p_val,
                "Num Lags": num_lags,
                "Num Observations": num_obs,
                "1% Critical Value": round(first_crit, 2),
                "5% Critical Value": round(second_crit, 2),
                "10% Critical Value": round(third_crit, 2),
            }

            # Use concat instead of append to avoid the warning
            results = pd.concat(
                [results, pd.DataFrame(adf_dict, index=[0])], axis=0, ignore_index=True
            )

    # Write to Excel
    results.to_excel(f"data/tests/{file_name}.xlsx")

    # Show the coins that are stationary, p-value < 0.05
    print(results[results["p-value"] < 0.05])


def kpss_test(diff : bool = False, file_name : str = "kpss_test"):
    """
    Performs the KPSS test on the data and saves the results to an Excel file.

    Parameters
    ----------
    diff : bool, optional
        If True then uses returns instead, by default False
    file_name : str, optional
        The name for the file to be saved in /data/tests/, by default "kpss_test"
    """

    results = pd.DataFrame()

    for coin in all_coins:
        for time in timeframes:
            df = read_csv(coin, time)

            if diff:
                df = df.diff().dropna()
                file_name = f"{file_name}_diff"

            test_stat, p_val, num_lags, crit_vals = kpss(df)
            first_crit = crit_vals["1%"]
            second_crit = crit_vals["5%"]
            third_crit = crit_vals["10%"]
            fourth_crit = crit_vals["2.5%"]

            info = {
                "Coin": coin,
                "Time": time,
                "KPSS Test Statistic": test_stat,
                "p-value": p_val,
                "Num Lags": num_lags,
                "1% Critical Value": round(first_crit, 2),
                "2.5% Critical Value": round(fourth_crit, 2),
                "5% Critical Value": round(second_crit, 2),
                "10% Critical Value": round(third_crit, 2),
            }

            # Use concat instead of append to avoid the warning
            results = pd.concat(
                [results, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
            )

    results.to_excel(f"data/tests/{file_name}.xlsx")

    # Show the coins that are stationary, p-value < 0.05
    print(results[results["p-value"] > 0.05])


def plot_price(crypto : str = "BTC", timeframe : str = "1d"):
    """
    Shows a plot of the price and returns of the crypto.

    Parameters
    ----------
    crypto : str, optional
        The symbol of the cryptocurrency, by default "BTC"
    timeframe : str, optional
        The time frame to use, by default "1d"
    """
    
    
    df = read_csv(crypto, timeframe)
    df_diff = df.diff().dropna()

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
    plt.savefig("data/plots/price.png")