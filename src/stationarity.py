import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt

# Local imports
from vars import all_coins, timeframes
from csv_data import read_csv


def write_adf_test(diff=False):
    file_name = "adf_test"
    adf_df = pd.DataFrame()

    for coin in all_coins:
        for time in timeframes:
            df = read_csv(coin, time)

            if diff:
                df = df.diff().dropna()
                file_name = "adf_test_diff"

            test_stat, p_val, num_lags, num_obs, crit_vals, _ = adfuller(df)
            first_crit = crit_vals["1%"]
            second_crit = crit_vals["5%"]
            third_crit = crit_vals["10%"]
            adf_dict = {
                "Coin": coin,
                "Time": time,
                "ADF Test Statistic": test_stat,
                "p-value": p_val,
                "Num Lags": num_lags,
                "Num Observations": num_obs,
                "1% Critical Value": round(first_crit, 2),
                "5% Critical Value": round(second_crit, 2),
                "10% Critical Value": round(third_crit, 2),
            }

            # Use concat instead of append to avoid the warning
            adf_df = pd.concat(
                [adf_df, pd.DataFrame(adf_dict, index=[0])], axis=0, ignore_index=True
            )

    # Write to .csv
    adf_df.to_csv(f"data/tests/{file_name}.csv")
    adf_df.to_excel(f"data/tests/{file_name}.xlsx")

    # Show the coins that are stationary, p-value < 0.05
    print(adf_df[adf_df["p-value"] < 0.05])


def write_kpss_test(diff=False):
    file_name = "kpss_test"

    kpss_df = pd.DataFrame()

    for coin in all_coins:
        for time in timeframes:
            df = read_csv(coin, time)

            if diff:
                df = df.diff().dropna()
                file_name = "kpss_test_diff"

            test_stat, p_val, num_lags, crit_vals = kpss(df)
            first_crit = crit_vals["1%"]
            second_crit = crit_vals["5%"]
            third_crit = crit_vals["10%"]
            fourth_crit = crit_vals["2.5%"]

            info = {
                "Coin": coin,
                "Time": time,
                "KPSS Test Statistic": test_stat,
                "p-value": p_val,
                "Num Lags": num_lags,
                "1% Critical Value": round(first_crit, 2),
                "2.5% Critical Value": round(fourth_crit, 2),
                "5% Critical Value": round(second_crit, 2),
                "10% Critical Value": round(third_crit, 2),
            }

            # Use concat instead of append to avoid the warning
            kpss_df = pd.concat(
                [kpss_df, pd.DataFrame(info, index=[0])], axis=0, ignore_index=True
            )

    # Write to .csv
    kpss_df.to_csv(f"data/tests/{file_name}.csv")
    kpss_df.to_excel(f"data/tests/{file_name}.xlsx")

    # Show the coins that are stationary, p-value < 0.05
    print(kpss_df[kpss_df["p-value"] > 0.05])


def plot_price(crypto, timeframe):
    df = read_csv(crypto, timeframe)
    df_diff = df.diff().dropna()

    _, axs = plt.subplots(2, 1, figsize=(12, 8))

    df["close"].plot(ax=axs[0], label="Price")
    df_diff["close"].plot(ax=axs[1], label="Returns")

    axs[0].legend(loc="upper right")
    axs[1].legend(loc="upper right")

    axs[0].set_ylabel("Price in USD")
    axs[1].set_ylabel("Returns in USD")

    axs[0].set_xlabel("")
    axs[1].set_xlabel("Date")

    plt.show()


plot_price("BTC", "1d")
# write_kpss_test(True)
# write_adf_test()
