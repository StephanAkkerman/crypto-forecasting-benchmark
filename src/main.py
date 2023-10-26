# Local imports
from data_analysis import (
    stationarity,
    auto_correlation,
    trend,
    seasonality,
    heteroskedasticity,
    stochasticity,
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
    data_properties,
    data_timespan,
)
import config


def data_analyses():
    # stationarity.stationarity_test(data_type="scaled", file_name="")

    # auto_correlation.durbin_watson(data_type="log returns")
    # auto_correlation.durbin_watson(data_type="scaled")
    # auto_correlation.autocorrelation_test(data_type="log returns", file_name="")
    # auto_correlation.autocorrelation_test(data_type="scaled", file_name="")

    # trend.trend_tests(data_type="log returns", as_csv=False, as_excel=False)
    # trend.trend_tests(data_type="scaled", as_csv=False, as_excel=False)

    # seasonality.seasonal_strength_test(data_type="log returns", to_excel=False, to_csv=False)
    # seasonality.seasonal_strength_test(data_type="scaled", to_excel=False, to_csv=False)

    # heteroskedasticity.uncon_het_tests(data_type="scaled", to_excel=False, to_csv=False)
    # heteroskedasticity.con_het_test(
    #    data_type="log returns", to_excel=False, to_csv=False
    # )

    stochasticity.calc_hurst(data_type="log returns", to_excel=False, to_csv=False)
    stochasticity.calc_hurst(data_type="scaled", to_excel=False, to_csv=False)


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

    # Tables of RMSE
    rmse.complete_models_ranking()
    rmse.complete_models_ranking(pred=config.scaled_to_log_pred)


def section_4_2():
    baseline.results_table()

    data_properties.time_frames()
    data_properties.time_frames(pred=config.scaled_to_log_pred)


def section_4_3():
    data_properties.auto_correlation()
    data_properties.trend()
    data_properties.seasonality()
    data_properties.heteroskedasticity()
    # data_properties.coin_correlation()
    # data_properties.correlation()
    data_properties.stochasticity_mann()
    data_properties.stochasticity_OLS()


def section_4_4():
    volatility.volatility_rmse_heatmap(config.log_returns_pred)
    data_properties.volatility()

    volatility.mcap_vol_boxplot()
    data_properties.mcap_cat_vol()
    data_properties.volatility_mcap()

    volatility.mcap_rmse_boxplot()
    data_properties.mcap()
    data_properties.mcap_cat()


def section_4_5():
    # volatility_analysis.plot_all_periods(show_validation=False)

    data_timespan.plt_extended_model_rmse()
    data_properties.data_timespan_mann(config.extended_pred, True)

    data_timespan.plt_stress_test_rmse()
    data_properties.data_timespan_mann(
        config.log_returns_stress_pred, True, 4, "greater"
    )


if __name__ == "__main__":
    # auto_correlation.durbin_watson(data_type="close")
    # auto_correlation.durbin_watson(data_type="returns")
    # auto_correlation.durbin_watson(data_type="log returns")

    # auto_correlation.autocorrelation_tests(data_type="close", as_csv=True)
    # auto_correlation.autocorrelation_tests(data_type="returns", as_csv=True)
    auto_correlation.autocorrelation_tests(data_type="scaled", as_csv=True)
