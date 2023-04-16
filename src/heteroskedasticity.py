import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_goldfeldquandt
from arch import arch_model
from statsmodels.tsa.arima.model import ARIMA
from data import all_coins, timeframes, read_csv


def complete_test(test, diff):
    # Read the dataset
    mk_df = pd.DataFrame()

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
            elif test == het_goldfeldquandt:
                _, p_value, _ = test(model.resid, model.model.exog)

            # Set signifance level
            alpha = 0.05

            info = {
                "Coin": coin,
                "Time": time,
                # "Breusch-Pagan test statistic": test_stat,
                "p-value": p_value,
            }

            mk_df = pd.concat(
                [mk_df, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
            )

    print(test)
    print(len(mk_df))
    print("Number of p-values < 0.05:")
    # The null hypothesis of homoskedasticity is rejected. There is evidence of heteroskedasticity.
    print(len(mk_df[mk_df["p-value"] < alpha]))

    # The null hypothesis of homoskedasticity cannot be rejected. There is no strong evidence of heteroskedasticity.
    print(mk_df[mk_df["p-value"] >= alpha])


def arch_test(coin, time):
    df = read_csv(coin, time, ["log returns"]).dropna()
    model = ARIMA(df, order=(1, 0, 0))
    result = model.fit()
    residuals = result.resid

    # Fit the ARCH model to the residuals
    arch_model_fit = arch_model(residuals, vol="Arch", p=1, q=0, mean="Zero").fit(
        disp="off"
    )
    print(arch_model_fit.summary())


def het_test(diff):
    for test in [het_breuschpagan, het_goldfeldquandt]:
        complete_test(test, diff)


if __name__ == "__main__":
    het_test(True)
    # arch_test("BTC", "1d")
