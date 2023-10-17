from tqdm import tqdm
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Local imports
from config import all_coins, timeframes, plots_dir, statistics_dir
from data.csv_data import get_data


def durbin_watson(data_type: str = "log returns") -> int:
    """
    Generates the test statistic for Durbin-Watson test for autocorrelation

    Parameters
    ----------
    data_type: str
        Options are: "close", "returns" "log returns", and "scaled", by default "log returns"

    Returns
    -------
    int
        The number of times the test statistic is between 1.5 and 2.5 (no autocorrelation)
    """
    results = pd.DataFrame()

    for coin in all_coins:
        for time in timeframes:
            for df in get_data(coin, time, data_type):
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
    print(
        len(
            results[(results["Durbin-Watson"] > 1.5) & (results["Durbin-Watson"] < 2.5)]
        )
    )


def autocorrelation_test(
    data_type: str = "log returns",
    file_name: str = "",
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
    nr_lags = 100

    for lag in tqdm(range(1, nr_lags + 1)):
        for coin in all_coins:
            for time in timeframes:
                dfs = get_data(coin, time, data_type)
                for df in dfs:
                    ljung_box_p_val = sm.stats.acorr_ljungbox(
                        df.values.squeeze(), lags=lag
                    )["lb_pvalue"].tolist()[-1]
                    if ljung_box_p_val < 0.05:
                        ljung_box_p_val = "Autocorrelated"
                    else:
                        ljung_box_p_val = "Not Autocorrelated"

                    X = sm.add_constant(df.iloc[:, :-1])
                    y = df.iloc[:, -1]
                    breusch_god_p_val = sm.stats.diagnostic.acorr_breusch_godfrey(
                        sm.OLS(y, X).fit(), nlags=lag
                    )[1]

                    if breusch_god_p_val < 0.05:
                        breusch_god_p_val = "Autocorrelated"
                    else:
                        breusch_god_p_val = "Not Autocorrelated"
                    info = {
                        "Coin": coin,
                        "Time Frame": time,
                        "Ljung-Box": ljung_box_p_val,
                        "Breusch-Godfrey": breusch_god_p_val,
                        "Lag": lag,
                    }

                    results = pd.concat(
                        [results, pd.DataFrame(info, index=[0])],
                        axis=0,
                        ignore_index=True,
                    )

    if file_name != "":
        results.to_csv(f"{statistics_dir}/{file_name}.csv", index=False)

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

        print(pivot_df)

        # Calculate a column to determine if predominantly "Autocorrelated"
        pivot_df["Predominantly"] = np.where(
            pivot_df["Autocorrelated"] >= len(dfs) * nr_lags / 2,
            "Autocorrelated",
            "Not Autocorrelated",
        )

        print(pivot_df["Predominantly"])

        autocorrelated = pivot_df[pivot_df["Predominantly"] == "Autocorrelated"]
        print(autocorrelated[["Coin", "Time Frame", "Predominantly"]])
        print("Autocorrelated:", len(autocorrelated))


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
