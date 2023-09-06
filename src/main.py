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
from hyperopt.analysis import best_hyperparameters
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


def model_performance(model: str, time_frame: str):
    boxplots.plotly_model_boxplot()
    boxplots.plotly_coin_boxplot()
    ts_analysis.compare_predictions()
    # ts_analysis.compare_two_predictions()
    rmse.all_models_heatmap()
    rmse.forecasting_models_stacked()
    rmse.stacked_bar_plot()
    rmse.stacked_bar_plot_all_tf()
    # rmse.rmse_comparison()


def time_frame_analysis(model: str = config.log_returns_model):
    # Use baseline comparison code
    # Also compare between time frames and RMSE
    baseline.baseline_comparison_heatmap(model)
    baseline.bar_plot(model)
    baseline.box_plot(model)


def volatility_analysis(model: str = config.log_returns_model):
    volatility.boxplot(model)
    volatility.model_boxplot()
    volatility.coin_boxplot()
    volatility.volatility_rmse_heatmap()
    volatility.mcap_rmse_boxplot()
    volatility.mcap_rmse_heatmap()
    volatility.mcap_volatility_heatmap()


if __name__ == "__main__":
    boxplots.complete_models_boxplot()
