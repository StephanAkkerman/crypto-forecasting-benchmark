import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Local imports
from data import all_coins, timeframes, read_csv


def durbin_watson(diff, log):
    dw_df = pd.DataFrame()

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
            dw_df = pd.concat(
                [dw_df, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
            )

    # A value of 2.0 indicates no autocorrelation
    return len(dw_df[(dw_df["Durbin-Watson"] > 1.5) & (dw_df["Durbin-Watson"] < 2.5)])


def ljung_box(lags, diff, log):
    lb_df = pd.DataFrame()

    for coin in all_coins:
        for time in timeframes:
            df = read_csv(coin, time)

            if log:
                df = np.log(df)

            if diff:
                df = df.diff().dropna()

            # Perform the Ljung-Box test with a lag of 20
            res = sm.stats.acorr_ljungbox(df.values.squeeze(), lags=lags)
            lb_stat = res["lb_stat"].tolist()[-1]
            lb_p = res["lb_pvalue"].tolist()[-1]

            info = {
                "Coin": coin,
                "Time": time,
                "Ljung-Box": lb_stat,
                "p-value": lb_p,
            }

            lb_df = pd.concat(
                [lb_df, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
            )

    # P-value > 0.05 indicates no autocorrelation
    return len(lb_df[lb_df["p-value"] > 0.05])


def breusch_godfrey(lags, diff, log):
    bg_df = pd.DataFrame()

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

            # Perform the Breusch-Godfrey test with 2 lags
            bg = sm.stats.diagnostic.acorr_breusch_godfrey(model, nlags=lags)

            info = {
                "Coin": coin,
                "Time": time,
                "Breusch-Godfrey": bg[0],
                "p-value": bg[1],
            }

            bg_df = pd.concat(
                [bg_df, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
            )

    return len(bg_df[bg_df["p-value"] > 0.05])


def auto_cor_test(diff, log):
    print("Durbin-Watson: ", durbin_watson(diff, log))

    avg_bg = 0
    avg_lb = 0

    for i in range(1, 100):
        print("Lags: ", i)
        avg_bg += breusch_godfrey(i, diff, log)
        avg_lb += ljung_box(i, diff, log)

    print("P-value > 0.05")
    print("Average BG: ", avg_bg / 99)
    print("Average LB: ", avg_lb / 99)


def plot_acf_pacf(crypto, timeframe):
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


if __name__ == "__main__":
    # auto_cor_test(True, True)
    plot_acf_pacf("BTC", "1d")
