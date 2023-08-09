import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_goldfeldquandt, het_arch

# Local imports
from config import all_coins, timeframes, statistics_dir
from data.csv_data import read_csv


def uncon_het_tests(log_returns: bool = True):
    """
    Tests for uncoditional heteroskedasticity using Breusch-Pagan and Goldfeld-Quandt tests.
    Saves the results as an excel file.

    Parameters
    ----------
    log_returns : bool
        If True, use the logarithmic returns of the data, by default True
    """

    # Read the dataset
    results = pd.DataFrame()

    for test in [het_breuschpagan, het_goldfeldquandt]:
        for coin in all_coins:
            for time in timeframes:
                df = read_csv(coin, time)

                if log_returns:
                    df = np.log(df)
                    df = df.diff().dropna()

                df["date"] = df.index

                # Convert the date to unix timestamp
                df["ts"] = df.date.values.astype(np.int64) // 10**9

                # Add a constant term to the dataset
                df["const"] = 1

                # Define the dependent and independent variables
                y = df["close"]

                x = df[["ts"]]
                x = sm.add_constant(x)

                # Fit the regression model
                model = sm.OLS(y, x).fit()

                # Perform Breusch-Pagan test
                if test == het_breuschpagan:
                    _, p_value, _, _ = test(model.resid, model.model.exog)
                    test_name = "Breusch-Pagan"
                elif test == het_goldfeldquandt:
                    _, p_value, _ = test(model.resid, model.model.exog)
                    test_name = "Goldfeld-Quandt"

                # Set signifance level
                alpha = 0.05

                info = {
                    "Coin": coin,
                    "Time": time,
                    # "Breusch-Pagan test statistic": test_stat,
                    "Result": "heteroskedasticity"
                    if p_value < alpha
                    else "homoskedasticity",
                    "Test": test_name,
                }

                results = pd.concat(
                    [results, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
                )

    # Save as excel
    results.to_excel(
        f"{statistics_dir}/unconditional_heteroskedasticity.xlsx", index=False
    )


def con_het_test():
    """
    Perform the Engle's ARCH test for conditional heteroskedasticity on all datasets and saves it as an excel file
    """

    results = pd.DataFrame()

    for coin in all_coins:
        for time in timeframes:
            # Read the dataset
            returns = read_csv(coin, time, ["log returns"]).dropna()

            # Perform the Engle's ARCH test
            _, p_value, _, _ = het_arch(returns)

            info = {
                "Coin": coin,
                "Time": time,
                "p-value": p_value,
                "result": "heteroskedasticity"
                if p_value < 0.05
                else "homoskedasticity",
            }

            results = pd.concat(
                [results, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
            )
    # save as .xlsx
    results.to_excel(f"{statistics_dir}/cond_heteroskedasticity.xlsx")
