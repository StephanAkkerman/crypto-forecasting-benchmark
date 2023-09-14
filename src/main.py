# Local imports
from data_analysis import (
    auto_correlation,
    correlation,
    heteroskedasticity,
    seasonality,
    stationarity,
    stochasticity,
    trend,
    volatility_analysis,
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


def model_performance(pred: str, time_frame: str):
    boxplots.plotly_model_boxplot()
    boxplots.plotly_coin_boxplot()
    ts_analysis.compare_predictions()
    # ts_analysis.compare_two_predictions()
    rmse.all_models_heatmap()
    rmse.forecasting_models_stacked()
    rmse.stacked_bar_plot()
    rmse.stacked_bar_plot_all_tf()
    # rmse.rmse_comparison()


def time_frame_analysis(pred: str = config.log_returns_pred):
    # Use baseline comparison code
    # Also compare between time frames and RMSE
    baseline.baseline_comparison_heatmap(pred)
    baseline.bar_plot(pred)
    baseline.box_plot(pred)


def volatilities(pred: str = config.log_returns_pred):
    volatility.boxplot(pred)
    volatility.model_boxplot()
    volatility.coin_boxplot()
    volatility.volatility_rmse_heatmap()
    volatility.mcap_rmse_boxplot()
    volatility.mcap_rmse_heatmap()
    volatility.mcap_volatility_heatmap()


def section_4():
    section_4_1("1m", coin="LTC", models=["ARIMA", "TCN", "LSTM"])


def section_4_1(time_frame, coin, models):
    # boxplots.complete_models_boxplot(preds=config.raw_preds, time_frame=time_frame)
    # boxplots.complete_models_boxplot(time_frame=time_frame)
    # rmse.rmse_means(preds=config.log_preds, time_frame=time_frame)
    # boxplots.plt_forecasting_models_comparison(
    #    time_frame=time_frame,
    #    forecasting_models=models,
    # )
    rmse.rmse_table(coin=coin, time_frame=time_frame, models=models)
    # ts_analysis.compare_predictions(coin=coin, time_frame=time_frame)
    ts_analysis.plot_predictions(coin=coin, time_frame=time_frame, models=models)
    volatility_analysis.plot_periods(timeframe=time_frame, coin=coin)


def section_4_2():
    pass


if __name__ == "__main__":
    baseline.tf_correlation()
