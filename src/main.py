# Local imports
from experiment import (
    forecast,
    train_test,
    ts_analysis,
    utils,
    boxplots,
    rmse,
    volatility,
    baseline,
    data_properties,
    data_timespan,
)
import config
from data_analysis import volatility_analysis


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

    # Black and white boxplots of all models and all datasets
    boxplots.complete_models_boxplot(time_frame=time_frame)

    # rmse.rmse_means(preds=config.log_preds, time_frame=time_frame)
    boxplots.plt_forecasting_models_comparison(
        time_frame=time_frame,
        forecasting_models=models,
    )
    # rmse.rmse_table(coin=coin, time_frame=time_frame, models=models)
    # ts_analysis.compare_predictions(coin=coin, time_frame=time_frame)
    # ts_analysis.plot_predictions(coin=coin, time_frame=time_frame, models=models)
    # volatility_analysis.plot_periods(timeframe=time_frame, coin=coin)

    # Boxplots of predictions
    boxplots.prediction_boxplots(time_frame=time_frame, models=models, coin=coin)


def section_4_2():
    data_properties.auto_correlation()
    data_properties.trend()
    data_properties.seasonality()
    data_properties.heteroskedasticity()
    data_properties.stochasticity()


def section_4_3():
    # volatility.volatility_rmse_heatmap(config.scaled_to_log_pred)
    volatility.mcap_rmse_boxplot()
    data_properties.mcap()
    data_properties.volatility_mcap()


def section_4_4():
    # data_timespan.plt_extended_model_rmse()
    # data_properties.extended_performance()
    # volatility_analysis.plot_all_periods(show_validation=False)
    # volatility_analysis.plot_periods(timeframe="4h")
    pass


if __name__ == "__main__":
    # section_4_2()
    # baseline.scaled_heatmap()
    # baseline.tf_significance()
    # baseline.box_plot(config.log_returns_pred)

    # section_4_1("1m")
    # data_properties.coin_correlation()
    # volatility.create_volatility_data()

    # Improve function to show all timeframes at once

    # section_4_4()
    # rmse.build_rmse_database(config.raw_to_log_stress_pred, skip_existing=False)
    rmse.build_rmse_database(config.scaled_to_log_stress_pred, skip_existing=False)
    # data_properties.seasonality()
    # data_properties.correlation(time_frame="1d", method="both")
