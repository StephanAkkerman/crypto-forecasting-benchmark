from tqdm import tqdm
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Local imports
from config import all_coins, timeframes, plots_dir, statistics_dir
from data.csv_data import get_data


def durbin_watson(df) -> int:
    # Fit a regression model to the data
    X = sm.add_constant(df.iloc[:, :-1])
    y = df.iloc[:, -1]
    model = sm.OLS(y, X).fit()

    # Perform the Durbin-Watson test
    dw = sm.stats.stattools.durbin_watson(model.resid)

    if dw > 1.5 and dw < 2.5:
        return "Not Autocorrelated"
    return "Autocorrelated"


def ljung_box(df, lag):
    ljung_box_p_val = sm.stats.acorr_ljungbox(df.values.squeeze(), lags=lag)[
        "lb_pvalue"
    ].tolist()[-1]
    if ljung_box_p_val < 0.05:
        return "Autocorrelated"
    return "Not Autocorrelated"


def breusch_godfrey(df, lag):
    X = sm.add_constant(df.iloc[:, :-1])
    y = df.iloc[:, -1]
    breusch_god_p_val = sm.stats.diagnostic.acorr_breusch_godfrey(
        sm.OLS(y, X).fit(), nlags=lag
    )[1]

    if breusch_god_p_val < 0.05:
        return "Autocorrelated"
    return "Not Autocorrelated"


def autocorrelation_tests(
    data_type: str = "log returns",
    as_csv: bool = False,
) -> None:
    """
    Performs either the Ljung-Box or Breusch-Godfrey test for autocorrelation on all datasets and saves it as an excel file

    Parameters
    ----------
    test_type : str
        The type of test to perform ('Ljung-Box' or 'Breusch-Godfrey')
    file_name : str, optional
        The name for the file to be saved in /data/tests/, by default ""
    """
    results = pd.DataFrame()
    dw_results = pd.DataFrame()
    nr_lags = 100

    for coin in tqdm(all_coins):
        for time in timeframes:
            dfs = get_data(coin, time, data_type)
            for df in dfs:
                info = {
                    "Coin": coin,
                    "Time Frame": time,
                    "Durbin-Watson": durbin_watson(df),
                }
                dw_results = pd.concat(
                    [dw_results, pd.DataFrame(info, index=[0])],
                    axis=0,
                    ignore_index=True,
                )

                # Ljung-Box and Breusch-Godfrey tests
                for lag in range(1, nr_lags + 1):
                    info = {
                        "Coin": coin,
                        "Time Frame": time,
                        "Ljung-Box": ljung_box(df, lag),
                        "Breusch-Godfrey": breusch_godfrey(df, lag),
                        "Lag": lag,
                    }

                    results = pd.concat(
                        [results, pd.DataFrame(info, index=[0])],
                        axis=0,
                        ignore_index=True,
                    )

    if as_csv:
        results.to_csv(
            f"{statistics_dir}/auto_correlation_results_{data_type.replace(' ', '_')}.csv",
            index=False,
        )
        dw_results.to_csv(
            f"{statistics_dir}/durbin_watson_results_{data_type.replace(' ', '_')}.csv",
            index=False,
        )

    print("Durbin-Watson:\n", dw_results["Durbin-Watson"].value_counts())

    # Find majority for both tests
    for test_name in ["Ljung-Box", "Breusch-Godfrey"]:
        # Grouping the DataFrame by 'Coin' and 'Time Frame' and counting the occurrences of "Autocorrelated"
        grouped_df = (
            results.groupby(["Coin", "Time Frame", test_name])
            .size()
            .reset_index(name="Count")
        )

        # Create a pivot table to show counts of "Autocorrelated" and "Not Autocorrelated" side by side
        pivot_df = grouped_df.pivot_table(
            index=["Coin", "Time Frame"],
            columns=test_name,
            values="Count",
            fill_value=0,
        ).reset_index()

        # Calculate a column to determine if predominantly "Autocorrelated"
        pivot_df["Predominantly"] = np.where(
            pivot_df["Autocorrelated"] >= len(dfs) * nr_lags / 2,
            "Autocorrelated",
            "Not Autocorrelated",
        )

        autocorrelated = pivot_df[pivot_df["Predominantly"] == "Autocorrelated"]
        # print(autocorrelated[["Coin", "Time Frame", "Predominantly"]])
        print(f"{test_name} Autocorrelated:", len(autocorrelated))


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

    df = get_data(crypto, timeframe, data_type="returns")[0]

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
    returns = get_data(crypto, timeframe, data_type="returns")[0]
    log_returns = get_data(crypto, timeframe)[0]

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
