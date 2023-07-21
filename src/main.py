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
from experiment import forecast, analysis, train_test


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
    pass
    # volatility.plotly_volatility("15m")

    # forecast.forecast_all(ignore_model=["ARIMA", "TBATS", "Prophet"])
    # forecast.forecast_model(
    #   "Prophet", start_from_coin="XTZ", start_from_time_frame="1m"
    # )
    # forecast.find_missing_forecasts(["Prophet"])
    # forecast.create_missing_forecasts(["Prophet"])
    # forecast.generate_extended_forecasts("NBEATS", "BTC", "1d")
