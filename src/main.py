# Local imports
from data_analysis import (
    auto_correlation,
    correlation,
    heteroskedasticity,
    seasonality,
    stationarity,
    stochasticity,
    trend,
)
from experiment import forecast, analysis, train_test, utils, boxplots, rmse, volatility
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
    # volatility.volatility_tests()

    # Stochasticity
    stochasticity.calc_hurst()


if __name__ == "__main__":
    # Run this on cluster
    forecast.stress_test_all(
        model=config.log_returns_model, ignore_model=["Prophet", "TBATS"]
    )
    forecast.stress_test_all(
        model=config.scaled_model, ignore_model=["Prophet", "TBATS"]
    )
    forecast.stress_test_all(model=config.raw_model, ignore_model=["Prophet", "TBATS"])
