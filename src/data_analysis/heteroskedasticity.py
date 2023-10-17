from tqdm import tqdm

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_goldfeldquandt, het_arch

# Local imports
from config import all_coins, timeframes, statistics_dir
from data.csv_data import get_data


def uncon_het_tests(
    data_type: str = "log returns", to_excel: bool = False, to_csv: bool = True
):
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

    file_name = f"{statistics_dir}/unconditional_heteroskedasticity_{data_type}"

    for coin in tqdm(all_coins):
        for time in timeframes:
            breusch_p_values = []
            gold_p_values = []
            for df in get_data(coin, time, data_type):
                df["date"] = df.index

                # Convert the date to unix timestamp
                df["ts"] = df.date.values.astype(np.int64) // 10**9

                # Add a constant term to the dataset
                df["const"] = 1

                # Define the dependent and independent variables
                if data_type != "close":
                    y = df["log returns"]
                else:
                    y = df["close"]

                x = df[["ts"]]
                x = sm.add_constant(x)

                # Fit the regression model
                model = sm.OLS(y, x).fit()

                # Perform Breusch-Pagan test
                _, breusch_p_value, _, _ = het_breuschpagan(
                    model.resid, model.model.exog
                )
                breusch_p_values.append(breusch_p_value)

                _, gold_p_value, _ = het_goldfeldquandt(model.resid, model.model.exog)
                gold_p_values.append(gold_p_value)

            breusch_p_value = np.mean(breusch_p_values)
            gold_p_value = np.mean(gold_p_values)

            # Set signifance level
            alpha = 0.05

            info = {
                "Coin": coin,
                "Time Frame": time,
                "Breusch-Pagan": "heteroskedasticity"
                if breusch_p_value < alpha
                else "homoskedasticity",
                "Goldfeld-Quandt": "heteroskedasticity"
                if gold_p_value < alpha
                else "homoskedasticity",
            }

            results = pd.concat(
                [results, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
            )

    # Save as .xlsx
    if to_excel:
        results.to_excel(f"{file_name}.xlsx", index=False)
    if to_csv:
        results.to_csv(f"{file_name}.csv", index=False)

    # Print results
    print(results["Breusch-Pagan"].value_counts())
    print(results["Goldfeld-Quandt"].value_counts())


def con_het_test(
    data_type: str = "log returns", to_csv: bool = True, to_excel: bool = False
):
    """
    Perform the Engle's ARCH test for conditional heteroskedasticity on all datasets and saves it as an excel file
    """

    file_name = f"{statistics_dir}/cond_heteroskedasticity_{data_type}"

    results = pd.DataFrame()

    for coin in tqdm(all_coins):
        for time in timeframes:
            # Read the dataset
            pvals = []
            for df in get_data(coin, time, data_type):
                # Perform the Engle's ARCH test
                _, p_value, _, _ = het_arch(df)
                pvals.append(p_value)

            p_value = np.mean(pvals)

            info = {
                "Coin": coin,
                "Time Frame": time,
                "p-value": p_value,
                "result": "heteroskedasticity"
                if p_value < 0.05
                else "homoskedasticity",
            }

            results = pd.concat(
                [results, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
            )
        # save as .xlsx
    if to_csv:
        results.to_csv(f"{file_name}.csv", index=False)
    if to_excel:
        results.to_excel(f"{file_name}.xlsx")

    print(results["result"].value_counts())
