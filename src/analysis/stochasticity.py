import pandas as pd
from scipy.stats import jarque_bera
from hurst import compute_Hc

from data.vars import all_coins, timeframes
from data.csv_data import read_csv

def jarque_bera_test():
    """
    Performs the Jarque-Bera test for normality on the logarithmic returns of the data.
    """
    
    results = pd.DataFrame()

    for coin in all_coins:
        for time in timeframes:
            df = read_csv(coin, time, ["log returns"]).dropna()
            _, p_value = jarque_bera(df["log returns"].values.tolist())

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


def calc_hurst():
    """
    Calculates the Hurst exponent for the data and saves it to an Excel file.
    """
    
    results = pd.DataFrame()
    for coin in all_coins:
        for time in timeframes:
            df = read_csv(coin, time)
            prices = df["close"].values.tolist()
            H, _, _ = compute_Hc(prices, kind='price', simplified=False)

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