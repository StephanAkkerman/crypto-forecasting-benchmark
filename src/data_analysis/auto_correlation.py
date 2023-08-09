import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Local imports
from config import all_coins, timeframes, plots_dir, statistics_dir
from data.csv_data import read_csv


def auto_cor_tests():
    """
    Main function to perform all autocorrelation tests
    """
    plot_acf()
    plot_log_returns()

    for diff, log in [(False, False), (True, False), (True, True)]:
        print("Durbin-Watson: ", durbin_watson(diff, log))
        breusch_godfrey(diff, log)
        ljung_box(diff, log)


def durbin_watson(diff: bool = True, log: bool = True) -> int:
    """
    Generates the test statistic for Durbin-Watson test for autocorrelation

    Parameters
    ----------
    diff : bool, optional
        Use the returns of the data, by default True
    log : bool, optional
        Use the logarithmic (returns) of the data, by default True

    Returns
    -------
    int
        The number of times the test statistic is between 1.5 and 2.5 (no autocorrelation)
    """
    results = pd.DataFrame()

    for coin in all_coins:
        for time in timeframes:
            df = read_csv(coin, time)

            if log:
                df = np.log(df)

            if diff:
                df = df.diff().dropna()

            # Fit a regression model to the data
            X = sm.add_constant(df.iloc[:, :-1])
            y = df.iloc[:, -1]
            model = sm.OLS(y, X).fit()

            # Perform the Durbin-Watson test
            dw = sm.stats.stattools.durbin_watson(model.resid)

            info = {
                "Coin": coin,
                "Time": time,
                "Durbin-Watson": dw,
            }

            # Use concat instead of append to avoid the warning
            results = pd.concat(
                [results, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
            )

    # A value of 2.0 indicates no autocorrelation
    return len(
        results[(results["Durbin-Watson"] > 1.5) & (results["Durbin-Watson"] < 2.5)]
    )


def ljung_box(diff: bool = True, log: bool = True) -> None:
    """
    Performs the Ljung-Box test for autocorrelation on all datasets and saves it as an excel file

    Parameters
    ----------
    diff : bool, optional
        Use the returns of the data, by default True
    log : bool, optional
        Use the logarithmic (returns) of the data, by default True
    """
    results = pd.DataFrame()
    file_name = "Ljung-Box"

    for lag in range(1, 101):
        for coin in all_coins:
            for time in timeframes:
                df = read_csv(coin, time)

                if log:
                    df = np.log(df)
                    file_name = f"{file_name}_log"

                if diff:
                    df = df.diff().dropna()
                    file_name = f"{file_name}_diff"

                # Perform the Ljung-Box test with a lag of 20
                res = sm.stats.acorr_ljungbox(df.values.squeeze(), lags=lag)
                p_val = res["lb_pvalue"].tolist()[-1]

                info = {
                    "Coin": coin,
                    "Time": time,
                    # P-value > 0.05 indicates no autocorrelation
                    "Result": "Autocorrelated"
                    if p_val < 0.05
                    else "Not Autocorrelated",
                    "Lag": lag,
                }

                results = pd.concat(
                    [results, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
                )

    # Save as excel
    results.to_excel(f"{statistics_dir}/{file_name}.xlsx", index=False)


def breusch_godfrey(diff: bool = True, log: bool = True):
    """
    Performs the Breusch-Godfrey test for autocorrelation on all datasets and saves it as an excel file

    Parameters
    ----------
    diff : bool, optional
        Use the returns of the data, by default True
    log : bool, optional
        Use the logarithmic (returns) of the data, by default True
    """
    results = pd.DataFrame()
    file_name = "Breusch-Godfrey"

    for lag in range(1, 101):
        for coin in all_coins:
            for time in timeframes:
                df = read_csv(coin, time)

                if log:
                    df = np.log(df)
                    file_name = f"{file_name}_log"

                if diff:
                    df = df.diff().dropna()
                    file_name = f"{file_name}_diff"

                # Fit a regression model to the data
                X = sm.add_constant(df.iloc[:, :-1])
                y = df.iloc[:, -1]
                model = sm.OLS(y, X).fit()

                # Perform the Breusch-Godfrey test with 2 lags
                bg = sm.stats.diagnostic.acorr_breusch_godfrey(model, nlags=lag)

                info = {
                    "Coin": coin,
                    "Time": time,
                    "Result": "Autocorrelated"
                    if bg[1] < 0.05
                    else "Not Autocorrelated",
                    "Lag": lag,
                }

                results = pd.concat(
                    [results, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
                )

    # Save it as excel
    results.to_excel(f"{statistics_dir}/{file_name}.xlsx", index=False)


def plot_acf(crypto="BTC", timeframe="1d"):
    """
    Plots the autocorrelation function of a cryptocurrency.

    Parameters
    ----------
    crypto : str, optional
        The name of the cryptocurrency, by default "BTC"
    timeframe : str, optional
        The timeframe of the data, by default "1d"
    """

    df = read_csv(crypto, timeframe)

    df_diff = df.diff().dropna()

    _, axs = plt.subplots(2, 1, figsize=(8, 8))

    plot_acf(df, ax=axs[0], title="")
    plot_acf(df_diff, ax=axs[1], title="")

    # Set axs axis titles
    axs[0].set_xlabel("Lag")
    axs[1].set_xlabel("Lag")

    axs[0].set_ylabel("ACF")
    axs[1].set_ylabel("ACF")

    plt.show()
    plt.savefig("output/plots/acf.png")


def plot_log_returns(crypto="BTC", timeframe="1d"):
    """
    Displays the returns and logarithmic returns of a cryptocurrency.

    Parameters
    ----------
    crypto : str, optional
        The name of the cryptocurrency, by default "BTC"
    timeframe : str, optional
        The timeframe of the data, by default "1d"
    """
    returns = read_csv(crypto, timeframe).diff().dropna()
    log_returns = read_csv(crypto, timeframe, ["log returns"])

    _, axs = plt.subplots(2, 1, figsize=(12, 8))

    returns["close"].plot(ax=axs[0], label="Returns")
    log_returns["log returns"].plot(ax=axs[1], label="Logarithmic returns")

    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")

    axs[0].set_ylabel("Returns in USD")
    axs[1].set_ylabel("Logarithmic returns")

    axs[0].set_xlabel("")
    axs[1].set_xlabel("Date")

    plt.show()
    plt.savefig(f"{plots_dir}/log_returns.png")
