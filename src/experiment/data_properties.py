from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import statsmodels.api as sm
from scipy.stats import kruskal, mannwhitneyu

import config
from data_analysis.correlation import corr_matrix
from experiment.baseline import get_all_baseline_comparison
from experiment.rmse import assign_mcap, assign_mcap_category, read_rmse_csv
from experiment.volatility import read_volatility_csv


def high_auto_cor(test_type: str = "Ljung-Box", data_type: str = "log_returns"):
    # Make an analysis of the data
    df = pd.read_csv(f"{config.statistics_dir}/auto_correlation_results_{data_type}.csv")
    
    # Grouping the DataFrame by 'Coin' and 'Time Frame' and counting the occurrences of "Autocorrelated"
    grouped_df = (
        df.groupby(["Coin", "Time Frame", test_type]).size().reset_index(name="Count")
    )

    # Create a pivot table to show counts of "Autocorrelated" and "Not Autocorrelated" side by side
    pivot_df = grouped_df.pivot_table(
        index=["Coin", "Time Frame"], columns=test_type, values="Count", fill_value=0
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
        rmse_df = read_rmse_csv(pred=pred, time_frame=time_frame, avg=avg)

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


def merge_vol(df, merge: bool = True, avg: bool = False):
    vol_dfs = []
    
    for time_frame in config.timeframes:
        vol_df = read_volatility_csv(time_frame=time_frame)
        # Add timeframe to it
        vol_df["Time Frame"] = time_frame

        # Name index coin
        vol_df["Market Cap"] = vol_df.index.map(assign_mcap)
        vol_df["Market Cap Category"] = vol_df.index.map(assign_mcap_category)
        vol_df["Coin"] = vol_df.index

        if avg:
            vol_df["period_volatility"] = vol_df["period_volatility"].apply(np.mean)

        vol_dfs.append(vol_df)

    # Concatenate the DataFrames
    vol_df = pd.concat(vol_dfs, axis=0, ignore_index=True)

    if merge:
        return pd.merge(df, vol_df, how="inner", on=["Coin", "Time Frame"])
    return vol_df


def mannwhiteny_test(
    df: pd.DataFrame,
    group1_name: str,
    group2_name: str,
    use_RMSE: bool,
    result_column: str = "Result",
    alternative: str = "less",
):
    """
    Perform a Mann-Whitney U test on the two groups.
    The first group being the group we expect have a greater RMSE (group1_name).
    The second group being the group we expect have a lower RMSE (group2_name).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame consisting of the data
    group1_name : str
        The name of the values that is the first group
    group2_name : str
        The name of the values that is the second group
    result_column : str, optional
        The name of the column to use for getting the values, by default "Result"
    alternative : str, optional
        The hypothesis of mann whitney u test, by default "less"
    """
    if use_RMSE:
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

            if pval < 0.05:
                # Add red color
                pval = f"\033[91m{pval}\033[0m"

            print(f"Mann-Whitney U test for {model}: U-statistic={U}, p-value={pval}")

    else:
        group1 = df[df[result_column] == group1_name]["period_volatility"].dropna()
        group2 = df[df[result_column] == group2_name]["period_volatility"].dropna()

        if len(group1) == 0 or len(group2) == 0:
            return

        # Perform the Mann-Whitney U test with 'less' as the alternative hypothesis
        U, pval = mannwhitneyu(group2, group1, alternative=alternative)

        if pval < 0.05:
            # Add red color
            pval = f"\033[91m{pval}\033[0m"

        print(f"Mann-Whitney U test: U-statistic={U}, p-value={pval}")


def auto_correlation(
    group_tf: bool = False, use_RMSE: bool = True, alternative: str = "less"
):
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
    df = merge_vol(df, avg=True)

    if group_tf:
        for tf in config.timeframes:
            print(tf)
            tf_df = df[df["Time Frame"] == tf]
            mannwhiteny_test(
                tf_df,
                "Autocorrelated",
                "Not Autocorrelated",
                use_RMSE=use_RMSE,
                alternative=alternative,
            )
    else:
        mannwhiteny_test(
            df,
            "Autocorrelated",
            "Not Autocorrelated",
            use_RMSE=use_RMSE,
            alternative=alternative,
        )


def find_majority(row):
    # Count the frequency of each unique result in the row
    counter = Counter(row)
    # Find the most common result
    most_common_result, _ = counter.most_common(1)[0]
    return most_common_result


def trend(group_tf: bool = False, use_RMSE: bool = True, alternative: str = "less"):
    # trend_tests(as_csv=True)
    df = pd.read_csv(f"{config.statistics_dir}/trend_results_log_returns.csv")

    # Apply the function across the rows
    df["Result"] = df.apply(find_majority, axis=1)

    # Change Results to trend if its increasing or decreasing
    df["Result"] = df["Result"].str.replace("increasing", "trend")
    df["Result"] = df["Result"].str.replace("decreasing", "trend")

    # Add RMSE data to the DataFrame
    df = merge_rmse(df)
    df = merge_vol(df, avg=True)

    if group_tf:
        for tf in config.timeframes:
            print(tf)
            tf_df = df[df["Time Frame"] == tf]
            # print(tf_df[tf_df["Result"] == "trend"])
            mannwhiteny_test(
                tf_df,
                "trend",
                "no trend",
                use_RMSE=use_RMSE,
                alternative=alternative,
            )
    else:
        mannwhiteny_test(
            df,
            "trend",
            "no trend",
            use_RMSE=use_RMSE,
            alternative=alternative,
        )


def ols_test(df, x, use_RMSE: bool):
    if use_RMSE:
        results = {}
        agg_results = pd.DataFrame()
        for forecasting_model in config.all_models:
            # Prepare the independent variable 'Seasonal Strength' and add a constant term for the intercept
            X = sm.add_constant(df[x])

            # Prepare the dependent variable, this is the model's RMSE.
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
    else:
        # Prepare the independent variable 'Seasonal Strength' and add a constant term for the intercept
        X = sm.add_constant(df[x])

        # Prepare the dependent variable, this is the model's RMSE.
        y = df["period_volatility"]

        # Perform linear regression
        model = sm.OLS(y, X).fit()

        results_df = pd.DataFrame(
            [
                {
                    "Intercept": model.params[0],
                    "Coef": model.params[1],
                    "R-squared": model.rsquared,
                    "P>|t|": model.pvalues[1],
                    "F-statistic": model.fvalue,
                }
            ]
        )
        print(results_df)
        return results_df


def seasonality(group_tf: bool = False, use_RMSE: bool = True):
    # Get seasonality data
    # seasonal_strength_test(log_returns=True)

    # Read seasonality data
    df = pd.read_csv(f"{config.statistics_dir}/stl_seasonality_log_returns.csv")

    # Add RMSE data to the DataFrame
    df = merge_rmse(df)
    df = merge_vol(df, avg=True)

    if group_tf:
        tf_results = []
        for tf in config.timeframes:
            tf_df = df[df["Time Frame"] == tf]
            tf_results.append(ols_test(tf_df, "Seasonal Strength", use_RMSE))
        if use_RMSE:
            df = pd.DataFrame(tf_results).T
            df.columns = config.timeframes
            print(df)
    else:
        ols_test(df, "Seasonal Strength", use_RMSE)


def heteroskedasticity(group_tf: bool = False, use_RMSE: bool = True):
    cond_het(group_tf)
    uncon_het(group_tf)


def uncon_het(group_tf: bool = False, use_RMSE: bool = True, alternative: str = "less"):
    df = pd.read_csv(
        f"{config.statistics_dir}/unconditional_heteroskedasticity_log_returns.csv"
    )

    # First test using breusch-pagan
    df = merge_rmse(df)
    df = merge_vol(df, avg=True)

    if group_tf:
        for tf in config.timeframes:
            print(tf)
            tf_df = df[df["Time Frame"] == tf]
            print("Results for Breusch-Pagan:")
            print(tf_df[tf_df["Breusch-Pagan"] == "homoskedasticity"]["Breusch-Pagan"])
            mannwhiteny_test(
                tf_df,
                "heteroskedasticity",
                "homoskedasticity",
                result_column="Breusch-Pagan",
                use_RMSE=use_RMSE,
                alternative=alternative,
            )

            print("\nResults for Goldfeld-Quandt:")
            print(
                tf_df[tf_df["Goldfeld-Quandt"] == "homoskedasticity"]["Goldfeld-Quandt"]
            )
            mannwhiteny_test(
                tf_df,
                "heteroskedasticity",
                "homoskedasticity",
                result_column="Goldfeld-Quandt",
                use_RMSE=use_RMSE,
                alternative=alternative,
            )
    else:
        print("Results for Breusch-Pagan:")
        mannwhiteny_test(
            df,
            "heteroskedasticity",
            "homoskedasticity",
            result_column="Breusch-Pagan",
            use_RMSE=use_RMSE,
            alternative=alternative,
        )

        print("\nResults for Goldfeld-Quandt:")
        mannwhiteny_test(
            df,
            "heteroskedasticity",
            "homoskedasticity",
            result_column="Goldfeld-Quandt",
            use_RMSE=use_RMSE,
            alternative=alternative,
        )


def cond_het(group_tf: bool = True, use_RMSE: bool = True, alternative: str = "less"):
    df = pd.read_csv(f"{config.statistics_dir}/cond_heteroskedasticity_log_returns.csv")

    # Add RMSE data to the DataFrame
    df = merge_rmse(df)
    df = merge_vol(df, avg=True)

    if group_tf:
        for tf in config.timeframes:
            print(tf)
            tf_df = df[df["Time Frame"] == tf]
            mannwhiteny_test(
                tf_df,
                "heteroskedasticity",
                "homoskedasticity",
                result_column="result",
                use_RMSE=use_RMSE,
                alternative=alternative,
            )
    else:
        mannwhiteny_test(
            df,
            "heteroskedasticity",
            "homoskedasticity",
            result_column="result",
            use_RMSE=use_RMSE,
            alternative=alternative,
        )


def correlation(time_frame: str = "1d", method: str = "pearson"):
    df = merge_rmse(None, merge=False)

    # group by time frame
    df = df[df["Time Frame"] == time_frame]

    # drop market cap
    df = df.drop(columns=["Market Cap", "Time Frame", "Market Cap Category"])

    # Make Coin index
    df = df.set_index("Coin", drop=True)

    df = df.T

    if method == "both":
        corr_methods = ["pearson", "spearman"]
    else:
        corr_methods = [method]

    for corr_method in corr_methods:
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

        print(corr_method)
        print(agg_results)


def stochasticity_mann(
    group_tf: bool = False, use_RMSE: bool = True, alternative: str = "less"
):
    df = pd.read_csv(f"{config.statistics_dir}/hurst_log_returns.csv")

    # Add RMSE data to the DataFrame
    df = merge_rmse(df)
    df = merge_vol(df, avg=True)

    if group_tf:
        for tf in config.timeframes:
            tf_df = df[df["Time Frame"] == tf]
            print(tf)
            mannwhiteny_test(
                tf_df,
                "Brownian motion",
                "Positively correlated",
                use_RMSE,
                alternative=alternative,
            )
    else:
        mannwhiteny_test(
            df,
            "Brownian motion",
            "Positively correlated",
            use_RMSE,
            alternative=alternative,
        )


def stochasticity_OLS(group_tf: bool = False, use_RMSE: bool = True):
    # Uses the H-value instead of category
    df = pd.read_csv(f"{config.statistics_dir}/hurst_log_returns.csv")

    # Add RMSE data to the DataFrame
    df = merge_rmse(df)
    df = merge_vol(df, avg=True)

    if group_tf:
        tf_results = []
        for tf in config.timeframes:
            tf_df = df[df["Time Frame"] == tf]
            tf_results.append(ols_test(tf_df, "Hurst exponent", use_RMSE))
        print(pd.DataFrame(tf_results).T)
    else:
        ols_test(df, "Hurst exponent", use_RMSE)


# Maybe improve this function to not use the mean
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


def vol_categories_mann(specific_test: bool = False):
    df = merge_rmse(merge_vol(None, merge=False), avg=False)

    # group by time frame
    for time_frame in config.timeframes:
        tf_df = df[df["Time Frame"] == time_frame]

        # Initialize an empty DataFrame to store aggregated results
        results = {}

        for vol_cat in ["low", "normal", "high"]:
            pvals = []
            for forecasting_model in config.all_models:
                period_results = pd.DataFrame()
                # Loop over the 5 periods
                for period in range(config.n_periods):
                    train = [lst[period] for lst in tf_df["train_volatility_class"]]
                    test = [lst[period] for lst in tf_df["test_volatility_class"]]
                    combined = [(train[i], test[i]) for i in range(len(train))]

                    y = [lst[period] for lst in tf_df[forecasting_model]]

                    # Convert to dataframe
                    new_df = pd.DataFrame({"Volatility": combined, "RMSE": y})

                    # Drop results where test volatility != vol_cat
                    new_df = new_df[
                        new_df["Volatility"].apply(lambda x: x[1] == vol_cat)
                    ]

                    period_results = pd.concat([period_results, new_df])

                # Splitting the DataFrame
                same_volatility = period_results[
                    period_results["Volatility"].apply(lambda x: x[0] == x[1])
                ]
                different_volatility = period_results[
                    period_results["Volatility"].apply(lambda x: x[0] != x[1])
                ]

                if specific_test:
                    if time_frame == "15m":
                        if vol_cat == "normal":
                            different_volatility = period_results[
                                period_results["Volatility"].apply(
                                    lambda x: x[0] == "high"
                                )
                            ]

                        if vol_cat == "high":
                            different_volatility = period_results[
                                period_results["Volatility"].apply(
                                    lambda x: x[0] == "low"
                                )
                            ]

                # Perform the Mann-Whitney U test with 'less' as the alternative hypothesis
                if len(same_volatility) != 0 and len(different_volatility) != 0:
                    U, pval = mannwhitneyu(
                        same_volatility["RMSE"],
                        different_volatility["RMSE"],
                        alternative="less",
                    )

                    pvals.append(pval)
                else:
                    pvals.append(np.nan)

            results[vol_cat] = pvals

        print(time_frame)
        print(pd.DataFrame(results, index=config.all_models))


def vol_categories_kruskal():
    df = merge_rmse(merge_vol(None, merge=False), avg=False)

    # group by time frame
    for time_frame in config.timeframes:
        tf_df = df[df["Time Frame"] == time_frame]

        # Initialize an empty DataFrame to store aggregated results
        results = {}

        for vol_cat in ["low", "normal", "high"]:
            pvals = []
            for forecasting_model in config.all_models:
                period_results = pd.DataFrame()
                # Loop over the 5 periods
                for period in range(config.n_periods):
                    train = [lst[period] for lst in tf_df["train_volatility_class"]]
                    test = [lst[period] for lst in tf_df["test_volatility_class"]]
                    combined = [(train[i], test[i]) for i in range(len(train))]

                    y = [lst[period] for lst in tf_df[forecasting_model]]

                    # Convert to dataframe
                    new_df = pd.DataFrame({"Volatility": combined, "RMSE": y})

                    # Drop results where test volatility != vol_cat
                    new_df = new_df[
                        new_df["Volatility"].apply(lambda x: x[1] == vol_cat)
                    ]

                    period_results = pd.concat([period_results, new_df])

                # Splitting the DataFrame
                low_train = period_results[
                    period_results["Volatility"].apply(lambda x: x[0] == "low")
                ]
                normal_train = period_results[
                    period_results["Volatility"].apply(lambda x: x[0] == "normal")
                ]
                high_train = period_results[
                    period_results["Volatility"].apply(lambda x: x[0] == "high")
                ]

                # Perform the Mann-Whitney U test with 'less' as the alternative hypothesis
                if (
                    len(low_train) != 0
                    and len(normal_train) != 0
                    and len(high_train) != 0
                ):
                    U, pval = kruskal(
                        low_train["RMSE"], normal_train["RMSE"], high_train["RMSE"]
                    )

                    pvals.append(pval)
                else:
                    pvals.append(np.nan)

            results[vol_cat] = pvals

        print(time_frame)
        print(pd.DataFrame(results, index=config.all_models))


def mcap_cat(kruskal_test: bool, group_tf: bool):
    df = merge_rmse(None, merge=False)

    dfs = []
    if group_tf:
        for tf in config.timeframes:
            tf_df = df[df["Time Frame"] == tf]
            dfs.append(tf_df)
    else:
        dfs = [df]

    for df in dfs:
        if group_tf:
            print(df["Time Frame"].iloc[0])
        for model in config.all_models:
            small = df[df["Market Cap Category"] == "Small"][model].dropna()
            mid = df[df["Market Cap Category"] == "Mid"][model].dropna()
            large = df[df["Market Cap Category"] == "Large"][model].dropna()

            if kruskal_test:
                # Perform the Mann-Whitney U test with 'less' as the alternative hypothesis
                U, pval = kruskal(small, mid, large)

                print(
                    f"Kruskal-Wallis test for {model}: U-statistic={U}, p-value={pval}"
                )
            else:
                U, pval = mannwhitneyu(mid, small, alternative="less")
                U, pval2 = mannwhitneyu(large, small, alternative="less")
                U, pval3 = mannwhitneyu(large, mid, alternative="less")

                print(f"Mann-Whitney test for {model}:", pval, pval2, pval3)


def mcap(group_tf: bool):
    df = merge_rmse(None, merge=False)

    dfs = []
    if group_tf:
        for tf in config.timeframes:
            tf_df = df[df["Time Frame"] == tf]
            dfs.append(tf_df)
    else:
        dfs = [df]

    for df in dfs:
        agg_results = pd.DataFrame()
        if group_tf:
            print(df["Time Frame"].iloc[0])
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


def mcap_cat_vol(kruskal_test: bool, group_tf: bool):
    df = merge_vol(None, merge=False)

    dfs = []
    if group_tf:
        for tf in config.timeframes:
            tf_df = df[df["Time Frame"] == tf]
            dfs.append(tf_df)
    else:
        dfs = [df]

    for df in dfs:
        if group_tf:
            print(df["Time Frame"].iloc[0])

        # for period in range(config.n_periods):
        # Separate the data into the two groups you wish to compare
        small = df[df["Market Cap Category"] == "Small"]["period_volatility"].dropna()

        # This groupo should perform better (lower RMSE)
        mid = df[df["Market Cap Category"] == "Mid"]["period_volatility"].dropna()

        large = df[df["Market Cap Category"] == "Large"]["period_volatility"].dropna()

        # Convert lists to list of mean
        small = [sum(x) / len(x) for x in zip(*small.to_list())]
        mid = [sum(x) / len(x) for x in zip(*mid.to_list())]
        large = [sum(x) / len(x) for x in zip(*large.to_list())]

        # Perform the Mann-Whitney U test with 'less' as the alternative hypothesis
        if kruskal_test:
            U, pval = kruskal(small, mid, large)
            print(f"Kruskal-Wallis test: U-statistic={U}, p-value={pval}")
        else:
            U, pval = mannwhitneyu(mid, small, alternative="less")
            U, pval2 = mannwhitneyu(large, small, alternative="less")
            U, pval3 = mannwhitneyu(large, mid, alternative="less")

            print(f"Mann-Whitney test:", pval, pval2, pval3)


def volatility_mcap(group_tf: bool):
    # We only have 1 mcap value

    df = merge_vol(None, merge=False)

    dfs = []
    if group_tf:
        for tf in config.timeframes:
            tf_df = df[df["Time Frame"] == tf]
            dfs.append(tf_df)
    else:
        dfs = [df]

    for df in dfs:
        # Initialize an empty DataFrame to store aggregated results
        agg_results = pd.DataFrame()
        if group_tf:
            print(df["Time Frame"].iloc[0])

        # Get the index that corresponds to the period
        X = df["Market Cap"]

        # Reset X index, to make it work for group_tf
        X = X.reset_index(drop=True)

        # Add constant term for intercept
        X = sm.add_constant(X)

        # Get the last period
        y = df["period_volatility"].apply(lambda x: x[-1])
        print(y)
        y = y.reset_index(drop=True)

        # Fit regression model
        model = sm.OLS(y, X).fit()

        results = pd.DataFrame(
            [
                {
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

        # Perform the Kruskal-Wallis test because there are more than 2 groups
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


def data_timespan_kruskal(pred: str = config.extended_pred):
    df = merge_rmse(None, avg=False, merge=False, pred=pred)

    for time_frame in config.timeframes:
        tf_df = df[df["Time Frame"] == time_frame]
        print(time_frame)
        for model in config.ml_models:
            # Get list of RMSE for each period
            periods = [list(item) for item in zip(*tf_df[model])]

            # Perform kruskal test
            U, pval = kruskal(*periods)

            # Change color to red if p-value is less than 0.05
            if pval < 0.05:
                pval = f"\033[91m{pval}\033[0m"

            print(f"Kruskal-Wallis test for {model}: U-statistic={U}, p-value={pval}")


def data_timespan_mann(
    pred: str = config.extended_pred,
    all_periods: bool = False,
    comparison_period: int = 4,
    alternative: str = "less",
):
    df = merge_rmse(None, avg=False, merge=False, pred=pred)

    if pred == config.extended_pred:
        models = config.ml_models
    else:
        models = config.all_models

    for time_frame in config.timeframes:
        tf_df = df[df["Time Frame"] == time_frame]
        print(time_frame)
        for model in models:
            # Get list of RMSE for each period
            periods = [list(item) for item in zip(*tf_df[model])]

            if all_periods:
                for p in range(config.n_periods):
                    if p == comparison_period:
                        continue

                    # Perform kruskal test
                    U, pval = mannwhitneyu(
                        periods[p], periods[comparison_period], alternative=alternative
                    )

                    # Change color to red if p-value is less than 0.05
                    if pval < 0.05:
                        pval = f"\033[91m{pval}\033[0m"

                    print(
                        f"Mann-Whitney test for {model} and period{p+1}: U-statistic={U}, p-value={pval}"
                    )
            else:
                # Perform kruskal test
                U, pval = mannwhitneyu(
                    periods[0], periods[comparison_period], alternative=alternative
                )

                # Change color to red if p-value is less than 0.05
                if pval < 0.05:
                    pval = f"\033[91m{pval}\033[0m"

                print(f"Mann-Whitney test for {model}: U-statistic={U}, p-value={pval}")
