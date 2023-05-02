import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import scipy.stats
from scipy.signal import correlate
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests

from vars import all_coins, timeframes
from csv_data import read_csv


def corr_matrix():
    one_d = pd.DataFrame()
    one_m = pd.DataFrame()
    four_h = pd.DataFrame()
    fifteen_m = pd.DataFrame()

    for coin in all_coins:
        for time in timeframes:
            df = read_csv(coin, time)
            df = np.log(df).diff().dropna()

            if time == "1d":
                one_d = pd.concat([one_d, df], axis=1, ignore_index=True)
            elif time == "1m":
                one_m = pd.concat([one_m, df], axis=1, ignore_index=True)
            elif time == "4h":
                four_h = pd.concat([four_h, df], axis=1, ignore_index=True)
            elif time == "15m":
                fifteen_m = pd.concat([fifteen_m, df], axis=1, ignore_index=True)

    for df in [one_m, fifteen_m, four_h, one_d]:
        df.columns = all_coins

        if df is one_m:
            time = "1-minute"
        elif df is fifteen_m:
            time = "15-minute"
        elif df is four_h:
            time = "4-hour"
        elif df is one_d:
            time = "1-day"

        pearson_matrix = df.corr(method="pearson").round(1)
        spear_matrix = df.corr(method="spearman").round(1)

        # Plot correlation matrices side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        sn.heatmap(pearson_matrix, cbar=False, annot=True, ax=ax1).set(
            title=f"{time} Pearson Correlation Matrix"
        )
        sn.heatmap(spear_matrix, cbar=False, annot=True, ax=ax2).set(
            title=f"{time} Spearman Correlation Matrix"
        )
        plt.show()


def corr_test():
    for time in timeframes:
        time_df = pd.DataFrame()
        for coin in all_coins:
            first_coin = read_csv(coin, time)
            first_coin = np.log(first_coin).diff().dropna()

            other_coins = [c for c in all_coins if c != coin]

            for c in other_coins:
                other_coin = read_csv(c, time)
                other_coin = np.log(other_coin).diff().dropna()

                # Calculate Pearson's correlation coefficient and p-value
                # correlation_coefficient, p_value = scipy.stats.pearsonr(first_coin['close'].values, other_coin['close'].values)
                correlation_coefficient, p_value = scipy.stats.spearmanr(
                    first_coin["close"].values, other_coin["close"].values
                )

                time_df = pd.concat(
                    [
                        time_df,
                        pd.DataFrame(
                            {
                                "Coin": coin,
                                "Other coin": c,
                                "Correlation coefficient": correlation_coefficient,
                                "p-value": p_value,
                            },
                            index=[0],
                        ),
                    ],
                    axis=0,
                    ignore_index=True,
                )

        print(time)
        # Check for p_val < 0.05
        print(len(time_df["p-value"] < 0.05))


def cross_cor():
    for time in timeframes:
        # Compute cross-correlations
        cross_correlations = np.zeros((len(all_coins), len(all_coins)))
        cross_lags = np.zeros((len(all_coins), len(all_coins)))
        for i in range(len(all_coins)):
            for j in range(len(all_coins)):
                first_coin = read_csv(all_coins[i], time, ["log returns"]).dropna()
                other_coin = read_csv(all_coins[j], time, ["log returns"]).dropna()

                # Perform cross-correlation
                cross_corr = correlate(
                    first_coin, other_coin, mode="full", method="auto"
                )
                max_cross_corr = np.max(np.abs(cross_corr))
                max_corr_lag = np.argmax(cross_corr) - (len(first_coin) - 1)

                cross_correlations[i, j] = max_cross_corr
                cross_lags[i, j] = max_corr_lag

        # cross_correlations = normalize(cross_correlations, norm='max')
        cross_correlations /= np.max(cross_correlations)

        # Replace values at diagonal with 1
        np.fill_diagonal(cross_correlations, 1)

        # Round cross-correlations
        cross_correlations = np.round(cross_correlations, 1)

        # Plot the cross-correlation results in a heatmap
        plt.figure(figsize=(8, 6))
        # sns.heatmap(
        #    cross_correlations,
        #    annot=True,
        #    cbar=False,
        #    vmin=0,
        #    vmax=1,
        #    square=True,
        #    xticklabels=all_coins,
        #    yticklabels=all_coins,
        # )
        sns.heatmap(
            cross_lags,
            annot=True,
            cbar=False,
            square=True,
            xticklabels=all_coins,
            yticklabels=all_coins,
        )

        if time == "1m":
            time = "1-Minute"
        elif time == "15m":
            time = "15-Minute"
        elif time == "4h":
            time = "4-Hour"
        elif time == "1d":
            time = "1-Day"

        plt.title(time + " Cross-correlation Heatmap")
        plt.xlabel("")
        plt.ylabel("")
        plt.show()


def granger_caus():
    # Perform Granger causality tests on the cryptocurrency data

    max_lag = 5  # The maximum number of lags to test for
    for time in timeframes:
        results_df = pd.DataFrame(columns=all_coins, index=all_coins)
        for c1 in all_coins:
            for c2 in all_coins:
                if c1 != c2:
                    first_coin = read_csv(c1, time, ["log returns"]).dropna()
                    other_coin = read_csv(c2, time, ["log returns"]).dropna()

                    cryptos = pd.concat([first_coin, other_coin], axis=1)
                    cryptos = cryptos.dropna()

                    result = grangercausalitytests(cryptos, max_lag, verbose=False)

                    # Extract the minimum p-value from the Granger causality test results
                    min_p_value = min(
                        [
                            result[lag][0]["ssr_ftest"][1]
                            for lag in range(1, max_lag + 1)
                        ]
                    )

                    # Store the minimum p-value in the results DataFrame
                    results_df.loc[c1, c2] = min_p_value
                else:
                    results_df.loc[c1, c2] = 1

        # Cast all values to float
        results_df = results_df.astype(float)
        results_df = results_df.round(1)

        # Add x to columns
        results_df.columns = [c + "_x" for c in all_coins]

        # Add y to index
        results_df.index = [c + "_y" for c in all_coins]

        # https://www.machinelearningplus.com/time-series/granger-causality-test-in-python/
        # See for more info

        sns.heatmap(
            results_df,
            annot=True,
            cbar=False,
            square=True,
        )

        plt.title(time + " Granger-Causality Heatmap")
        plt.xlabel("")
        plt.ylabel("")
        plt.show()


if __name__ == "__main__":
    # corr_test()
    # corr_matrix()
    granger_caus()
