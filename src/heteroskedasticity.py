import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_goldfeldquandt, het_arch
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from data import all_coins, timeframes, read_csv


def complete_test(test, diff):
    # Read the dataset
    mk_df = pd.DataFrame()

    for test in [het_breuschpagan, het_goldfeldquandt]:
        for coin in all_coins:
            for time in timeframes:
                df = read_csv(coin, time)

                if diff:
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

                mk_df = pd.concat(
                    [mk_df, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
                )

    # Save as excel
    mk_df.to_excel(f"data/tests/unconditional_heteroskedasticity.xlsx", index=False)

def arch_test(coin, time):
    returns = read_csv(coin, time, ["log returns"]).dropna()

    # Fit GARCH model
    model = arch_model(returns, vol="Garch", p=1, q=1)
    results = model.fit()

    print(results.summary())


def het_test(diff):
    for test in [het_breuschpagan, het_goldfeldquandt]:
        complete_test(test, diff)


def cond_het_test():
    # Read the dataset
    results = pd.DataFrame()

    for coin in all_coins:
        for time in timeframes:
            returns = read_csv(coin, time, ["log returns"]).dropna()

            # Perform the Engle's ARCH test
            test_stat, p_value, f_stat, f_p_value = het_arch(returns)

            info = {
                "Coin": coin,
                "Time": time,
                # "Breusch-Pagan test statistic": test_stat,
                "p-value": p_value,
                "result": "heteroskedasticity"
                if p_value < 0.05
                else "homoskedasticity",
            }

            results = pd.concat(
                [results, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
            )

    print(results)
    # save as .xlsx
    results.to_excel("data/tests/cond_heteroskedasticity.xlsx")


if __name__ == "__main__":
    het_test(True)
    # arch_test("BTC", "1d")
    # cond_het_test()
