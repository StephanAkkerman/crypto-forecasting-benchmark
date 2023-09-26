from collections import Counter

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sn
from scipy.stats import mannwhitneyu, kruskal

import config
from experiment.rmse import read_rmse_csv, assign_mcap_category, assign_mcap
from experiment.volatility import read_volatility_csv
from experiment.baseline import get_all_baseline_comparison
from data_analysis.correlation import corr_matrix


def high_auto_cor(test_type: str):
    # Make an analysis of the data
    df = pd.read_csv(f"{config.statistics_dir}/{test_type}_log_returns.csv")

    # Grouping the DataFrame by 'Coin' and 'Time Frame' and counting the occurrences of "Autocorrelated"
    grouped_df = (
        df.groupby(["Coin", "Time Frame", "Result"]).size().reset_index(name="Count")
    )

    # Create a pivot table to show counts of "Autocorrelated" and "Not Autocorrelated" side by side
    pivot_df = grouped_df.pivot_table(
        index=["Coin", "Time Frame"], columns="Result", values="Count", fill_value=0
    ).reset_index()

    # Calculate a column to determine if predominantly "Autocorrelated"
    pivot_df["Predominantly"] = np.where(
        pivot_df["Autocorrelated"] > 49, "Autocorrelated", "Not Autocorrelated"
    )

    return pivot_df[["Coin", "Time Frame", "Predominantly"]]


def merge_rmse(df, avg: bool = True, merge: bool = True, pred=config.log_returns_pred):
    # Add RMSE data to the DataFrame
    rmse_dfs = []
    for time_frame in config.timeframes:
        rmse_df = read_rmse_csv(
            pred=config.log_returns_pred, time_frame=time_frame, avg=avg
        )

        # Add timeframe to it
        rmse_df["Time Frame"] = time_frame
        rmse_df["Market Cap Category"] = rmse_df.index.map(assign_mcap_category)
        rmse_df["Market Cap"] = rmse_df.index.map(assign_mcap)

        # Name index coin
        rmse_df["Coin"] = rmse_df.index

        rmse_dfs.append(rmse_df)

    # Concatenate the DataFrames
    rmse_df = pd.concat(rmse_dfs, axis=0, ignore_index=True)

    # Add RMSE data to the DataFrame
    if merge:
        return pd.merge(df, rmse_df, how="inner", on=["Coin", "Time Frame"])
    return rmse_df


def merge_vol(df, merge: bool = True):
    vol_dfs = []

    for time_frame in config.timeframes:
        vol_df = read_volatility_csv(time_frame=time_frame)
        # Add timeframe to it
        vol_df["Time Frame"] = time_frame

        # Name index coin
        vol_df["Market Cap"] = vol_df.index.map(assign_mcap)
        vol_df["Market Cap Category"] = vol_df.index.map(assign_mcap_category)
        vol_df["Coin"] = vol_df.index

        vol_dfs.append(vol_df)

    # Concatenate the DataFrames
    vol_df = pd.concat(vol_dfs, axis=0, ignore_index=True)

    if merge:
        return pd.merge(df, vol_df, how="inner", on=["Coin", "Time Frame"])
    return vol_df


def mannwhiteny_test(
    df,
    group1_name: str,
    group2_name: str,
    result_column: str = "Result",
    alternative: str = "less",
):
    for model in config.all_models:
        # Separate the data into the two groups you wish to compare
        group1 = df[df[result_column] == group1_name][model].dropna()

        # This groupo should perform better (lower RMSE)
        group2 = df[df[result_column] == group2_name][model].dropna()

        if len(group1) == 0 or len(group2) == 0:
            print(f"Skipping {model}")
            continue

        # Perform the Mann-Whitney U test with 'less' as the alternative hypothesis
        U, pval = mannwhitneyu(group2, group1, alternative=alternative)

        print(f"Mann-Whitney U test for {model}: U-statistic={U}, p-value={pval}")


def auto_correlation(group_tf: bool = False):
    # Find the cryptocurrencies that show autocorrelation on the log returns
    ljung = high_auto_cor("Ljung-Box")
    breusch = high_auto_cor("Breusch-Godfrey")

    # Find the overlap between the two DataFrames and determine the final 'Result'
    df = pd.merge(
        ljung,
        breusch,
        how="outer",
        on=["Coin", "Time Frame"],
        suffixes=("_ljung", "_breusch"),
    )
    df["Result"] = np.where(
        (df["Predominantly_ljung"] == "Autocorrelated")
        & (df["Predominantly_breusch"] == "Autocorrelated"),
        "Autocorrelated",
        "Not Autocorrelated",
    )

    df = merge_rmse(df)
    if group_tf:
        for tf in config.timeframes:
            print(tf)
            tf_df = df[df["Time Frame"] == tf]
            mannwhiteny_test(tf_df, "Autocorrelated", "Not Autocorrelated")
    else:
        mannwhiteny_test(df, "Autocorrelated", "Not Autocorrelated")


def find_majority(row):
    # Count the frequency of each unique result in the row
    counter = Counter(row)
    # Find the most common result
    most_common_result, freq = counter.most_common(1)[0]
    return most_common_result


def trend(group_tf: bool = False):
    # trend_tests(as_csv=True)
    df = pd.read_csv(f"{config.statistics_dir}/trend_results_log_returns.csv")

    # Apply the function across the rows
    df["Result"] = df.apply(find_majority, axis=1)

    # Change Results to trend if its increasing or decreasing
    df["Result"] = df["Result"].str.replace("increasing", "trend")
    df["Result"] = df["Result"].str.replace("decreasing", "trend")

    # Add RMSE data to the DataFrame
    df = merge_rmse(df)

    if group_tf:
        for tf in config.timeframes:
            print(tf)
            tf_df = df[df["Time Frame"] == tf]
            mannwhiteny_test(tf_df, "trend", "no trend")
    else:
        mannwhiteny_test(df, "trend", "no trend")


def ols_test(df, x):
    results = {}
    agg_results = pd.DataFrame()
    for forecasting_model in config.all_models:
        # Prepare the independent variable 'Seasonal Strength' and add a constant term for the intercept
        X = sm.add_constant(df[x])

        # Prepare the dependent variable. This is for RandomForest. Repeat for other models.
        y = df[forecasting_model]

        # Perform linear regression
        model = sm.OLS(y, X).fit()

        results_df = pd.DataFrame(
            [
                {
                    # "Time Frame": time_frame,
                    "Model": forecasting_model,
                    "Intercept": model.params[0],
                    "Coef": model.params[1],
                    "R-squared": model.rsquared,
                    "P>|t|": model.pvalues[1],
                    "F-statistic": model.fvalue,
                }
            ]
        )
        # Append to the aggregated results DataFrame
        agg_results = pd.concat([agg_results, results_df])

        results[forecasting_model] = model.pvalues[1]

    print(agg_results)
    return results


def seasonality(group_tf: bool = False):
    # Get seasonality data
    # seasonal_strength_test(log_returns=True)

    # Read seasonality data
    df = pd.read_csv(f"{config.statistics_dir}/stl_seasonality_log_returns.csv")

    # Add RMSE data to the DataFrame
    df = merge_rmse(df)

    if group_tf:
        tf_results = []
        for tf in config.timeframes:
            tf_df = df[df["Time Frame"] == tf]
            tf_results.append(ols_test(tf_df, "Seasonal Strength"))
        print(pd.DataFrame(tf_results).T)
    else:
        ols_test(df, "Seasonal Strength")


def heteroskedasticity(group_tf: bool = False):
    cond_het(group_tf)
    uncon_het(group_tf)


def uncon_het(group_tf: bool = False):
    df = pd.read_csv(
        f"{config.statistics_dir}/unconditional_heteroskedasticity_log_returns.csv"
    )

    # First test using breusch-pagan
    df = merge_rmse(df)

    if group_tf:
        for tf in config.timeframes:
            print(tf)
            tf_df = df[df["Time Frame"] == tf]
            print("Results for Breusch-Pagan:")
            mannwhiteny_test(
                tf_df,
                "heteroskedasticity",
                "homoskedasticity",
                result_column="Breusch-Pagan",
            )

            print("\nResults for Goldfeld-Quandt:")
            mannwhiteny_test(
                tf_df,
                "heteroskedasticity",
                "homoskedasticity",
                result_column="Goldfeld-Quandt",
            )
    else:
        print("Results for Breusch-Pagan:")
        mannwhiteny_test(
            df, "heteroskedasticity", "homoskedasticity", result_column="Breusch-Pagan"
        )

        print("\nResults for Goldfeld-Quandt:")
        mannwhiteny_test(
            df,
            "heteroskedasticity",
            "homoskedasticity",
            result_column="Goldfeld-Quandt",
        )


def cond_het(group_tf: bool = False):
    df = pd.read_csv(f"{config.statistics_dir}/cond_heteroskedasticity_log_returns.csv")

    # Add RMSE data to the DataFrame
    df = merge_rmse(df)

    if group_tf:
        for tf in config.timeframes:
            print(tf)
            tf_df = df[df["Time Frame"] == tf]
            mannwhiteny_test(
                tf_df, "heteroskedasticity", "homoskedasticity", result_column="result"
            )
    else:
        mannwhiteny_test(
            df, "heteroskedasticity", "homoskedasticity", result_column="result"
        )


def correlation(time_frame: str = "1d", corr_method: str = "pearson"):
    df = merge_rmse(None, merge=False)

    # group by time frame
    df = df[df["Time Frame"] == time_frame]

    # drop market cap
    df = df.drop(columns=["Market Cap", "Time Frame", "Market Cap Category"])

    # Make Coin index
    df = df.set_index("Coin", drop=True)

    df = df.T

    rmse_corr = df.corr(method=corr_method)
    price_corr = corr_matrix(time_frame, corr_method)
    
    agg_results = pd.DataFrame()
    
    for i in range(len(rmse_corr)):
        # Get the ith row from each correlation matrix
        X = rmse_corr.iloc[i].values
        y = price_corr.iloc[i].values
        
        # Add a constant term for the intercept
        X = sm.add_constant(X)

        # Create the OLS model and fit it to the data
        model = sm.OLS(y, X).fit()

        results = pd.DataFrame(
                [
                    {
                        # "Time Frame": time_frame,
                        "Coin": rmse_corr.index[i],
                        "Intercept": model.params[0],
                        "Coef": model.params[1],
                        "R-squared": model.rsquared,
                        "P>|t|": model.pvalues[1],
                        "F-statistic": model.fvalue,
                    }
                ]
            )

        # Append to the aggregated results DataFrame
        agg_results = pd.concat([agg_results, results])

    print(agg_results)

def stochasticity(group_tf: bool = False):
    # calc_hurst()

    df = pd.read_csv(f"{config.statistics_dir}/hurst_log_returns.csv")

    # Add RMSE data to the DataFrame
    df = merge_rmse(df)

    if group_tf:
        for tf in config.timeframes:
            tf_df = df[df["Time Frame"] == tf]
            print(tf)
            mannwhiteny_test(tf_df, "Brownian motion", "Positively correlated")
    else:
        mannwhiteny_test(df, "Brownian motion", "Positively correlated")


def volatility():
    df = merge_rmse(merge_vol(None, merge=False), avg=False)

    # Initialize an empty DataFrame to store aggregated results
    agg_results = pd.DataFrame()

    for forecasting_model in config.all_models:
        # Loop over the 5 periods
        for period in range(config.n_periods):
            # Get the index that corresponds to the period
            train = [lst[period] for lst in df["train_volatility"]]
            test = [lst[period] for lst in df["test_volatility"]]
            y = [lst[period] for lst in df[forecasting_model]]

            # Convert to dataframe
            X = pd.DataFrame({"Train": train, "Test": test})
            y = pd.Series(y, name="target")

            # Add constant term for intercept
            X = sm.add_constant(X)

            # Fit regression model
            model = sm.OLS(y, X).fit()

            results = pd.DataFrame(
                [
                    {
                        # "Time Frame": time_frame,
                        "Model": forecasting_model,
                        "Period": period,
                        "Intercept": model.params[0],
                        "Train_Coef": model.params[1],
                        "Test_Coef": model.params[2],
                        "R-squared": model.rsquared,
                        "P>|t|_Train": model.pvalues[1],
                        "P>|t|_Test": model.pvalues[2],
                        "F-statistic": model.fvalue,
                    }
                ]
            )

            # Append to the aggregated results DataFrame
            agg_results = pd.concat([agg_results, results])

    # Group by 'Model' and calculate the mean for each group
    average_results = agg_results.groupby("Model").mean().reset_index()

    print(average_results)


def mcap(use_cat: bool = False):
    df = merge_rmse(None, merge=False)

    # For categories
    if use_cat:
        for model in config.all_models:
            # Separate the data into the two groups you wish to compare
            small = df[df["Market Cap Category"] == "Small"][model].dropna()

            # This groupo should perform better (lower RMSE)
            mid = df[df["Market Cap Category"] == "Mid"][model].dropna()

            large = df[df["Market Cap Category"] == "Large"][model].dropna()

            # Perform the Mann-Whitney U test with 'less' as the alternative hypothesis
            U, pval = kruskal(small, mid, large)

            print(f"Kruskal-Wallis test for {model}: U-statistic={U}, p-value={pval}")
    else:
        agg_results = pd.DataFrame()
        for forecasting_model in config.all_models:
            # Prepare the independent variable 'Seasonal Strength' and add a constant term for the intercept
            X = sm.add_constant(df["Market Cap"])

            # Prepare the dependent variable. This is for RandomForest. Repeat for other models.
            y = df[forecasting_model]

            # Perform linear regression
            model = sm.OLS(y, X).fit()

            results = pd.DataFrame(
                [
                    {
                        "Model": forecasting_model,
                        "Intercept": model.params[0],
                        "Market Cap_Coef": model.params[1],
                        "R-squared": model.rsquared,
                        "P>|t|_Market Cap": model.pvalues[1],
                        "F-statistic": model.fvalue,
                    }
                ]
            )

            # Append to the aggregated results DataFrame
            agg_results = pd.concat([agg_results, results])
        print(agg_results)


def volatility_mcap(use_cat=False):
    df = merge_rmse(None, merge=False)

    # For categories
    if use_cat:
        # for period in range(config.n_periods):
        # Separate the data into the two groups you wish to compare
        small = df[df["Market Cap Category"] == "Small"]["period_volatility"].dropna()

        # This groupo should perform better (lower RMSE)
        mid = df[df["Market Cap Category"] == "Mid"]["period_volatility"].dropna()

        large = df[df["Market Cap Category"] == "Large"]["period_volatility"].dropna()

        # Perform the Mann-Whitney U test with 'less' as the alternative hypothesis
        U, pval = kruskal(small, mid, large)

        print(f"Kruskal-Wallis test: U-statistic={U}, p-value={pval}")
    else:
        # Initialize an empty DataFrame to store aggregated results
        agg_results = pd.DataFrame()

        # Loop over the 5 periods
        for period in range(config.n_periods):
            # Get the index that corresponds to the period
            X = df["Market Cap"]

            y = [lst[period] for lst in df["period_volatility"]]
            y = pd.Series(y, name="target")

            # Add constant term for intercept
            X = sm.add_constant(X)

            # Fit regression model
            model = sm.OLS(y, X).fit()

            results = pd.DataFrame(
                [
                    {
                        "Period": period,
                        "Intercept": model.params[0],
                        "Volatility_Coef": model.params[1],
                        "R-squared": model.rsquared,
                        "P>|t|_Volatility": model.pvalues[1],
                        "F-statistic": model.fvalue,
                    }
                ]
            )

            # Append to the aggregated results DataFrame
            agg_results = pd.concat([agg_results, results])

        print(agg_results)


def time_frames(pred: str = config.log_returns_pred):
    dfs = get_all_baseline_comparison(pred=pred)

    # Add time frame to it
    for i, time_frame in enumerate(config.timeframes):
        dfs[i]["Time Frame"] = time_frame

    # Merge dfs
    df = pd.concat(dfs)

    for model in config.all_models:
        if model == "ARIMA":
            continue

        # Separate the data into the two groups you wish to compare
        tf_1m = df[df["Time Frame"] == "1m"][model].dropna()

        # This groupo should perform better (lower RMSE)
        tf_15m = df[df["Time Frame"] == "15m"][model].dropna()

        tf_4h = df[df["Time Frame"] == "4h"][model].dropna()

        tf_1d = df[df["Time Frame"] == "1d"][model].dropna()

        # Perform the Mann-Whitney U test with 'less' as the alternative hypothesis
        U, pval = kruskal(tf_1m, tf_15m, tf_4h, tf_1d)

        print(f"Kruskal-Wallis test for {model}: U-statistic={U}, p-value={pval}")


def coin_correlation(show_heatmap=True, time_frame="1d"):
    df = merge_rmse(None, merge=False)
    df = df[df["Time Frame"] == time_frame]
    df = df[["Coin"] + config.all_models]
    # Set coin as column
    df = df.set_index("Coin", drop=True)

    if show_heatmap:
        # Plot heatmap
        sn.heatmap(df.corr(), annot=True, cmap="coolwarm", vmin=0, vmax=1)
        plt.show()

    # Perform linear regression
    agg_results = pd.DataFrame()
    for coin in config.all_coins:
        X = df.loc[coin]

        for coin2 in config.all_coins:
            if coin == coin2:
                continue
            y = df.loc[coin2]

            # Add constant term for intercept
            X = sm.add_constant(X)

            # Fit regression model
            model = sm.OLS(y, X).fit()

            results = pd.DataFrame(
                [
                    {
                        "Coins": f"{coin}-{coin2}",
                        "Intercept": model.params[0],
                        "Coef": model.params[1],
                        "R-squared": model.rsquared,
                        "P>|t|": model.pvalues[1],
                        "F-statistic": model.fvalue,
                    }
                ]
            )

            # Append to the aggregated results DataFrame
            agg_results = pd.concat([agg_results, results])

    print(agg_results)


def extended_performance():
    df = merge_rmse(None, avg=False, merge=False, pred=config.extended_pred)

    for time_frame in config.timeframes:
        tf_df = df[df["Time Frame"] == time_frame]
        print(time_frame)
        for model in config.ml_models:
            periods = []
            for p in range(config.n_periods):
                periods.append(tf_df[model].str[p].to_list())

            # Perform kruskal test
            U, pval = kruskal(*periods)

            print(f"Kruskal-Wallis test for {model}: U-statistic={U}, p-value={pval}")
