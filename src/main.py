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
from experiment import forecast, analysis, train_test, utils, boxplots


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
    # utils.build_all_rmse_databases()
    # analysis.compare_predictions("extended_models", "BTC", "1d")
    # boxplots.plotly_coin_boxplot(model_dir="models", time_frame="1d")
    # boxplots.plotly_model_boxplot(model_dir="models", time_frame="1d")
    # utils.extended_model_predictions("BTC", "1d")
    # forecast.fix_extended_test_train()
    forecast.test()
