import numpy as np
import pandas as pd
from scipy.stats import jarque_bera
from hurst import compute_Hc
from data import all_coins, timeframes, read_csv


def jarque_bera_test():
    results = pd.DataFrame()

    for coin in all_coins:
        for time in timeframes:
            df = read_csv(coin, time, ["log returns"]).dropna()
            jb_stat, p_value = jarque_bera(df["log returns"].values.tolist())

            info = {
                "Coin": coin,
                "Time": time,
                "P-value": p_value,
            }

            results = pd.concat(
                [results, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
            )

    alpha = 0.05  # significance level
    # if p_value < alpha:
    #    print(f"Reject the null hypothesis: Logarithmic returns do not follow a normal distribution (p-value={p_value:.4f})")
    # else:
    #    print(f"Fail to reject the null hypothesis: Logarithmic returns may follow a normal distribution (p-value={p_value:.4f})")

    print(results[results["P-value"] < alpha][["Coin", "Time", "P-value"]])


def get_hurst_exponent(time_series, max_lag=20):
    """Returns the Hurst Exponent of the time series"""

    lags = range(2, max_lag)

    # variances of the lagged differences
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]

    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]


def rough_vol_test():
    # https://towardsdatascience.com/introduction-to-the-hurst-exponent-with-code-in-python-4da0414ca52e
    # https://quantdare.com/demystifying-the-hurst-exponent/

    for coin in all_coins:
        for time in timeframes:
            df = read_csv(coin, time)
            log_returns = df["close"].values.tolist()
            # Method can be either 'hosking', 'cholesky', or 'daviesharte'
            hurst_exps = []
            for lag in [20, 100, 300, 500, 999]:
                hurst_exp = get_hurst_exponent(log_returns, lag)
                hurst_exps.append(hurst_exp)
                print(f"Hurst exponent with {lag} lags: {hurst_exp:.4f}")

def calc_hurst():
    results = pd.DataFrame()
    for coin in all_coins:
        for time in timeframes:
            df = read_csv(coin, time)
            prices = df["close"].values.tolist()
            H, c, data = compute_Hc(prices, kind='price', simplified=False)

            if 0.45 < H < 0.55:
                hurst_result = "Brownian motion"
            elif H < 0.45:
                hurst_result = "Negatively correlated"
            elif H > 0.55:
                hurst_result = "Positively correlated"

            info = {
                "Coin": coin,
                "Time": time,
                "Hurst exponent": H,
                "Hurst result": hurst_result,
            }

            results = pd.concat(
                [results, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
            )

            
    results.to_excel("data/tests/hurst.xlsx", index=False)

calc_hurst()
