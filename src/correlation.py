import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import scipy.stats
from scipy.signal import correlate
import seaborn as sns
from data import all_coins, timeframes, read_csv


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
        for i in range(len(all_coins)):
            for j in range(len(all_coins)):

                first_coin = read_csv(all_coins[i], time, ["log returns"]).dropna()
                other_coin = read_csv(all_coins[j], time, ["log returns"]).dropna()
                
                # Perform cross-correlation
                cross_corr = correlate(first_coin, other_coin, mode='full', method='auto')
                max_cross_corr = np.max(cross_corr)

                cross_correlations[i, j] = max_cross_corr

                #print(f"Cross-correlation between {all_coins[i]} and {all_coins[j]}: {max_cross_corr}")

        # Normalize cross-correlations
        cross_correlations /= np.max(cross_correlations)

        # Round cross-correlations
        cross_correlations = np.round(cross_correlations, 1)

        # Plot the cross-correlation results in a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cross_correlations, annot=True, cmap='coolwarm', vmin=0, vmax=1, square=True, xticklabels=all_coins, yticklabels=all_coins)
        
        if time == "1m":
            time = "1-Minute"
        elif time == "15m":
            time = "15-Minute"
        elif time == "4h":
            time = "4-Hour"
        elif time == "1d":
            time = "1-Day"
        
        plt.title(time + ' Cross-correlation Heatmap')
        plt.xlabel('Cryptocurrencies')
        plt.ylabel('Cryptocurrencies')
        plt.show()

def cross_cor_plot():
    # float lists for cross
    # correlation
    x = read_csv("BTC", "1d", ["log returns"]).dropna()
    y = read_csv("ETH", "1d", ["log returns"]).dropna()

    x = x["log returns"].values
    y = y["log returns"].values

    # Plot graph
    fig = plt.figure()
    ax1 = fig.add_subplot(211)

    # cross correlation using
    # xcorr() function
    ax1.xcorr(x, y, usevlines=True, maxlags=5, normed=True, lw=2)
    # adding grid to the graph
    ax1.grid(True)
    ax1.axhline(0, color="blue", lw=2)

    # show final plotted graph
    plt.show()


if __name__ == "__main__":
    # corr_test()
    # corr_matrix()
    cross_cor()
