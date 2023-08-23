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
    # rmse.all_models_comparison2()
    # rmse.all_models_stacked_bar(log_data=False)
    # boxplots.plotly_model_boxplot(model=config.log_to_raw_model)
    # boxplots.plotly_model_boxplot(model=config.log_returns_model)
    boxplots.plotly_boxplot_comparison(
        model_1=config.log_to_raw_model, model_2=config.raw_model
    )
