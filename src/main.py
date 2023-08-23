import torch
import os

# Local imports
from data_analysis import (
    auto_correlation,
    correlation,
    heteroskedasticity,
    seasonality,
    stationarity,
    stochasticity,
    trend,
    volatility,
)
from experiment import forecast, analysis, train_test, utils, boxplots, rmse
import config


def methods():
    """
    Generates all analysis data and plots.
    """

    # Stationarity
    stationarity.stationarity_tests()

    # Autocorrelation
    auto_correlation.auto_cor_tests()

    # Trend
    trend.trend_tests()

    # Seasonality
    seasonality.seasonal_strength_test()

    # Heteroskedasticity
    heteroskedasticity.uncon_het_tests()
    heteroskedasticity.con_het_test()

    # Correlation
    correlation.correlation_tests()

    # Volatility
    volatility.volatility_tests()

    # Stochasticity
    stochasticity.calc_hurst()


if __name__ == "__main__":
    # analysis.compare_predictions("extended_models", "BTC", "1d")
    # boxplots.plotly_coin_boxplot(model=config.log_returns_model, time_frame="1d")
    # boxplots.plotly_model_boxplot(model=config.log_returns_model, time_frame="1d")
    # analysis.compare_two_predictions()
    # boxplots.plotly_model_boxplot_comparison("1d")
    # rmse.rmse_comparison()
    # forecast.forecast_all(
    #    model_dir=config.scaled_model_dir, ignore_model=["TBATS", "Prophet"]
    # )
    # analysis.compare_predictions(
    #    model=config.extended_model, coin="BTC", time_frame="1d"
    # )
    # rmse.extended_models_comparison_per_model(model_name="RNN", time_frame="1d")
    # boxplots.plotly_extended_model_rmse(time_frame="1d")
    # rmse.rmse_heatmap(time_frame="1d", model=config.extended_model)
    # rmse.baseline_comparison()
    # analysis.compare_predictions(config.raw_model, coin="BTC", time_frame="1d")
    # volatility.plot_periods()
    # utils.log_model_to_price(config.scaled_to_log_model)
    # forecast.find_missing_forecasts(config.scaled_model)
    rmse.build_comlete_rmse_database()
    # utils.raw_model_to_log()
