from collections import Counter

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import mannwhitneyu, kruskal

import config
from experiment.rmse import read_rmse_csv, assign_mcap_category, assign_mcap
from experiment.volatility import read_volatility_csv


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


def merge_rmse(df, avg: bool = True, merge: bool = True):
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

        # Perform the Mann-Whitney U test with 'less' as the alternative hypothesis
        U, pval = mannwhitneyu(group2, group1, alternative=alternative)

        print(f"Mann-Whitney U test for {model}: U-statistic={U}, p-value={pval}")


def auto_correlation():
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

    mannwhiteny_test(df, "Autocorrelated", "Not Autocorrelated")


def find_majority(row):
    # Count the frequency of each unique result in the row
    counter = Counter(row)
    # Find the most common result
    most_common_result, freq = counter.most_common(1)[0]
    return most_common_result


def trend():
    # trend_tests(as_csv=True)
    df = pd.read_csv(f"{config.statistics_dir}/trend_results_log_returns.csv")

    # Finding rows where all test columns have the same value
    # result_df = df[df.iloc[:, 2:].apply(lambda row: len(row.unique()) == 1, axis=1)]

    # Apply the function across the rows
    df["Result"] = df.apply(find_majority, axis=1)

    # Drop the columns that are not needed
    df = df[["Coin", "Time Frame", "Result"]]

    # Change Results to trend if its increasing or decreasing
    df["Result"] = df["Result"].str.replace("increasing", "trend")
    df["Result"] = df["Result"].str.replace("decreasing", "trend")

    # Add RMSE data to the DataFrame
    df = merge_rmse(df)

    mannwhiteny_test(df, "trend", "no trend")


def seasonality():
    # Get seasonality data
    # seasonal_strength_test(log_returns=True)

    # Read seasonality data
    df = pd.read_csv(f"{config.statistics_dir}/stl_seasonality_log_returns.csv")

    # Add RMSE data to the DataFrame
    df = merge_rmse(df)

    results = {}

    for forecasting_model in config.all_models:
        # Prepare the independent variable 'Seasonal Strength' and add a constant term for the intercept
        X = sm.add_constant(df["Seasonal Strength"])

        # Prepare the dependent variable. This is for RandomForest. Repeat for other models.
        y = df[forecasting_model]

        # Perform linear regression
        model = sm.OLS(y, X).fit()

        # Convert the summary results to a DataFrame
        results_df = pd.DataFrame(model.summary2().tables[1])

        # Access the p-value for "Seasonal Strength"
        p_value_seasonal_strength = results_df.loc["Seasonal Strength", "P>|t|"]

        results[forecasting_model] = p_value_seasonal_strength

    print(results)


def heteroskedasticity():
    cond_het()
    uncon_het()


def uncon_het():
    df = pd.read_csv(
        f"{config.statistics_dir}/unconditional_heteroskedasticity_log_returns.csv"
    )
    # Find rows where 'Breusch-Pagan' and 'Goldfeld-Quandt' have the same result
    # same_result_df = df[df['Breusch-Pagan'] == df['Goldfeld-Quandt']]

    # First test using breusch-pagan
    df = merge_rmse(df)

    # Rename column result to Result
    df = df.rename(columns={"result": "Result"})

    mannwhiteny_test(df)

    print("Results for Breusch-Pagan:")
    mannwhiteny_test(
        df, "heteroskedasticity", "homoskedasticity", result_column="Breusch-Pagan"
    )

    print("\nResults for Goldfeld-Quandt:")
    mannwhiteny_test(
        df, "heteroskedasticity", "homoskedasticity", result_column="Goldfeld-Quandt"
    )


def cond_het():
    df = pd.read_csv(f"{config.statistics_dir}/cond_heteroskedasticity_log_returns.csv")

    # Add RMSE data to the DataFrame
    df = merge_rmse(df)

    mannwhiteny_test(
        df, "heteroskedasticity", "homoskedasticity", result_column="result"
    )


def correlation():
    pass


def stochasticity():
    # calc_hurst()

    df = pd.read_csv(f"{config.statistics_dir}/hurst_log_returns.csv")

    # Add RMSE data to the DataFrame
    df = merge_rmse(df)

    mannwhiteny_test(df, "Brownian motion", "Positively correlated")


def volatility():
    vol_dfs = []

    for time_frame in config.timeframes:
        vol_df = read_volatility_csv(time_frame=time_frame)
        # Add timeframe to it
        vol_df["Time Frame"] = time_frame

        # Name index coin
        vol_df["Coin"] = vol_df.index

        vol_dfs.append(vol_df)

    # Concatenate the DataFrames
    vol_df = pd.concat(vol_dfs, axis=0, ignore_index=True)

    df = merge_rmse(vol_df, avg=False)

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
                        "Time Frame": time_frame,
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
    df = pd.concat(vol_dfs, axis=0, ignore_index=True)

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
