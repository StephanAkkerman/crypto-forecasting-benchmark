import numpy as np
import pandas as pd
import seaborn as sn
import scipy.stats
import matplotlib.pyplot as plt
from scipy.signal import correlate
from statsmodels.tsa.stattools import grangercausalitytests

import config
from data.csv_data import read_csv


def correlation_tests():
    """
    Generates all correlation plots and matrices
    """

    corr_matrices()
    corr_pval()
    cross_cor(True)
    cross_cor(False)
    granger_caus()


def corr_matrix(time_frame: str, corr_method: str = "pearson"):
    tf_df = pd.DataFrame()
    for coin in config.all_coins:
        df = read_csv(coin, time_frame, ["log returns"]).dropna()
        tf_df = pd.concat([tf_df, df], axis=1, ignore_index=True)

    return tf_df.corr(method=corr_method).round(1)


def corr_matrices():
    """
    Generates the correlation matrix for all coins and time frames
    Shows each time frame separately, with Pearson and Spearman correlation
    """

    for time_frame in config.timeframes:
        pearson_matrix = corr_matrix(time_frame, "pearson")
        spear_matrix = corr_matrix(time_frame, "spearman")

        # Plot correlation matrices side by side
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        sn.heatmap(pearson_matrix, cbar=False, annot=True, ax=ax1).set(
            title=f"{time_frame} Pearson Correlation Matrix"
        )
        sn.heatmap(spear_matrix, cbar=False, annot=True, ax=ax2).set(
            title=f"{time_frame} Spearman Correlation Matrix"
        )

        plt.show()


def corr_pval(pearson: bool = False):
    """
    Prints the coin pairs that have a p-value below 0.05 using the Spearman or Pearson correlation
    """

    for time in config.timeframes:
        time_df = pd.DataFrame()
        for coin in config.all_coins:
            first_coin = read_csv(coin, time)
            first_coin = np.log(first_coin).diff().dropna()

            other_coins = [c for c in config.all_coins if c != coin]

            for c in other_coins:
                other_coin = read_csv(c, time)
                other_coin = np.log(other_coin).diff().dropna()

                # Calculate Pearson's correlation coefficient and p-value
                if pearson:
                    correlation_coefficient, p_value = scipy.stats.pearsonr(
                        first_coin["close"].values, other_coin["close"].values
                    )
                else:
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


def cross_cor(show_lags: bool = False):
    """
    Displays cross-correlation between all coins and time frames as a heatmap

    Parameters
    ----------
    show_lags : bool, optional
        Shows the lags instead of cross-correlation, by default False
    """
    for time in config.timeframes:
        # Compute cross-correlations
        cross_correlations = np.zeros((len(config.all_coins), len(config.all_coins)))
        cross_lags = np.zeros((len(config.all_coins), len(config.all_coins)))
        for i in range(len(config.all_coins)):
            for j in range(len(config.all_coins)):
                first_coin = read_csv(
                    config.all_coins[i], time, ["log returns"]
                ).dropna()
                other_coin = read_csv(
                    config.all_coins[j], time, ["log returns"]
                ).dropna()

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
        if not show_lags:
            sn.heatmap(
                cross_correlations,
                annot=True,
                cbar=False,
                vmin=0,
                vmax=1,
                square=True,
                xticklabels=config.all_coins,
                yticklabels=config.all_coins,
            )
        else:
            sn.heatmap(
                cross_lags,
                annot=True,
                cbar=False,
                square=True,
                xticklabels=config.all_coins,
                yticklabels=config.all_coins,
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
    """
    Performs Granger causality tests on the cryptocurrency data, displayed as a heatmap
    """

    max_lag = 5  # The maximum number of lags to test for
    for time in config.timeframes:
        results_df = pd.DataFrame(columns=config.all_coins, index=config.all_coins)
        for c1 in config.all_coins:
            for c2 in config.all_coins:
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
        results_df.columns = [c + "_x" for c in config.all_coins]

        # Add y to index
        results_df.index = [c + "_y" for c in config.all_coins]

        # https://www.machinelearningplus.com/time-series/granger-causality-test-in-python/
        # See for more info

        sn.heatmap(
            results_df,
            annot=True,
            cbar=False,
            square=True,
        )

        plt.title(time + " Granger-Causality Heatmap")
        plt.xlabel("")
        plt.ylabel("")
        plt.show()
