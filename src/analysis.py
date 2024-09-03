import os

import config
from data_analysis import (
    auto_correlation,
    heteroskedasticity,
    seasonality,
    stationarity,
    stochasticity,
    trend,
    volatility_analysis,
)
from experiment import (
    baseline,
    boxplots,
    data_properties,
    data_timespan,
    forecast,
    rmse,
    volatility,
)


def data_analysis_tests(data_type: str = "log returns", as_csv=True, as_excel=False):
    """
    Performs all the data analysis tests:
    - Stationarity
    - Autocorrelation
    - Trend
    - Seasonality
    - Heteroskedasticity
    - Stochasticity

    data_type : str
        Options are: "close", "returns" "log returns", and "scaled", by default "log returns"
    as_csv : bool, optional
        Whether to save the results as a CSV, by default True
    as_excel : bool, optional
        Whether to save the results as an Excel file, by default False
    """
    # Create the directory
    os.makedirs(config.statistics_dir, exist_ok=True)

    # Stationarity
    stationarity.stationarity_test(
        data_type=data_type, as_csv=as_csv, as_excel=as_excel
    )

    # Auto correlation
    auto_correlation.autocorrelation_tests(
        data_type=data_type, as_csv=as_csv, as_excel=as_excel
    )

    # Trend
    trend.trend_tests(data_type=data_type, as_csv=as_csv, as_excel=as_excel)

    # Seasonality
    seasonality.seasonal_strength_test(
        data_type=data_type, to_excel=as_excel, to_csv=as_csv
    )

    # Heteroskedasticity
    heteroskedasticity.uncon_het_tests(
        data_type=data_type, to_excel=as_excel, to_csv=as_csv
    )
    heteroskedasticity.con_het_test(
        data_type=data_type, to_excel=as_excel, to_csv=as_csv
    )

    # Stochasticity
    stochasticity.calc_hurst(data_type=data_type, to_excel=as_excel, to_csv=as_excel)


def forecast_models():
    forecast.forecast_all()


def stress_test_all():
    forecast.stress_test_all()


def forecast_models_extended():
    forecast.forecast_all(pred=config.extended_pred)


def forecast_analysis():
    for time_frame in config.timeframes:
        section_4_1(time_frame)


def section_4_1(time_frame):
    if time_frame == "1d":
        models = ["ARIMA", "LightGBM", "TCN", "TBATS", "LSTM"]
        coin = "ETH"
        # Only use this when time_frame is 1d
        boxplots.complete_models_boxplot(preds=config.raw_preds, time_frame=time_frame)
    elif time_frame == "4h":
        models = ["ARIMA", "LightGBM", "TCN", "TBATS", "RNN"]
        coin = "TRX"
    elif time_frame == "15m":
        models = ["ARIMA", "XGB", "TCN", "TBATS", "GRU"]
        coin = "IOTA"
    elif time_frame == "1m":
        models = ["ARIMA", "LightGBM", "TCN", "TBATS", "LSTM"]
        coin = "LTC"

    # Boxplots of all models and all datasets
    boxplots.complete_models_boxplot(time_frame=time_frame)

    boxplots.plt_forecasting_models_comparison(
        time_frame=time_frame,
        forecasting_models=models,
    )
    # Boxplots of predictions
    boxplots.prediction_boxplots(time_frame=time_frame, models=models, coin=coin)

    # Tables of RMSE
    rmse.complete_models_ranking()
    rmse.complete_models_ranking(pred=config.scaled_to_log_pred)


def forecast_statistical_tests():
    """
    Performs all the data analysis tests for section 4.2:
    - Autocorrelation
    - Trend
    - Seasonality
    - Heteroskedasticity
    - Stochasticity
    """

    data_properties.auto_correlation()
    data_properties.trend()
    data_properties.seasonality()
    data_properties.heteroskedasticity()
    data_properties.stochasticity_mann()
    data_properties.stochasticity_OLS()


def market_factors_impact():
    # Volatility
    volatility.volatility_rmse_heatmap(config.log_returns_pred)
    data_properties.volatility()

    # Market cap and volatility
    volatility.mcap_vol_boxplot()
    data_properties.mcap_cat_vol()
    data_properties.volatility_mcap()

    # Market cap and RMSE
    volatility.mcap_rmse_boxplot()
    data_properties.mcap()
    data_properties.mcap_cat()


def time_frame_impact():
    baseline.results_table()

    data_properties.time_frames()
    data_properties.time_frames(pred=config.scaled_to_log_pred)


def section_4_5():
    volatility_analysis.plot_all_periods(show_validation=False)

    data_timespan.plt_extended_model_rmse()
    data_properties.data_timespan_mann(config.extended_pred, True)

    data_timespan.plt_stress_test_rmse()
    data_properties.data_timespan_mann(
        config.log_returns_stress_pred, True, 4, "greater"
    )
