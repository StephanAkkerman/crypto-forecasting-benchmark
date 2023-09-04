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
from experiment import (
    forecast,
    train_test,
    ts_analysis,
    utils,
    boxplots,
    rmse,
    volatility,
    baseline,
)
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


def baseline_plots(model: str = config.log_returns_model):
    baseline.baseline_comparison_heatmap(model)
    baseline.bar_plot(model)
    baseline.box_plot(model)


def volatility_plots(model: str = config.log_returns_model):
    volatility.boxplot(model)
    volatility.model_boxplot()
    volatility.coin_boxplot()
    volatility.volatility_rmse_heatmap()
    volatility.mcap_rmse_boxplot()
    volatility.mcap_rmse_heatmap()
    volatility.mcap_volatility_heatmap()


if __name__ == "__main__":
    pass
