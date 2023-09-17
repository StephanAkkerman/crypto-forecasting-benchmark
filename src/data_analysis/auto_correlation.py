from tqdm import tqdm
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Local imports
from config import all_coins, timeframes, plots_dir, statistics_dir
from data.csv_data import read_csv


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


def autocorrelation_test(
    test_type: str = "Ljung-Box",
    diff: bool = False,
    log_returns: bool = True,
    to_csv: bool = True,
    to_excel: bool = False,
) -> None:
    """
    Performs either the Ljung-Box or Breusch-Godfrey test for autocorrelation on all datasets and saves it as an excel file

    Parameters
    ----------
    test_type : str
        The type of test to perform ('Ljung-Box' or 'Breusch-Godfrey')
    diff : bool, optional
        Use the returns of the data, by default True
    log_returns : bool, optional
        Use the logarithmic (returns) of the data, by default True
    to_csv : bool, optional
        Save the results to a CSV file, by default True
    to_excel : bool, optional
        Save the results to an Excel file, by default False
    """
    results = pd.DataFrame()
    file_name = test_type

    if log_returns:
        file_name = f"{file_name}_log_returns"
    elif diff:
        file_name = f"{file_name}_diff"

    for lag in tqdm(range(1, 101)):
        for coin in all_coins:
            for time in timeframes:
                if log_returns:
                    df = read_csv(coin, time, ["log returns"])
                    df = df.dropna()
                else:
                    df = read_csv(coin, time, ["close"])
                    if diff:
                        df = df.diff().dropna()

                if test_type == "Ljung-Box":
                    res = sm.stats.acorr_ljungbox(df.values.squeeze(), lags=lag)
                    p_val = res["lb_pvalue"].tolist()[-1]

                elif test_type == "Breusch-Godfrey":
                    X = sm.add_constant(df.iloc[:, :-1])
                    y = df.iloc[:, -1]
                    model = sm.OLS(y, X).fit()
                    res = sm.stats.diagnostic.acorr_breusch_godfrey(model, nlags=lag)
                    p_val = res[1]

                info = {
                    "Coin": coin,
                    "Time Frame": time,
                    "Result": "Autocorrelated"
                    if p_val < 0.05
                    else "Not Autocorrelated",
                    "Lag": lag,
                }

                results = pd.concat(
                    [results, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
                )

    if to_excel:
        results.to_excel(f"{statistics_dir}/{file_name}.xlsx", index=False)

    if to_csv:
        results.to_csv(f"{statistics_dir}/{file_name}.csv", index=False)

    print(results)


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
