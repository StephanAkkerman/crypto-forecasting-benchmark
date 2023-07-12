import os
import logging

from tqdm import tqdm
import pytorch_lightning as pl
import torch


# Models
from darts.models import (
    StatsForecastAutoARIMA,
    RNNModel,
    TCNModel,
    NBEATSModel,
    TFTModel,
    RandomForest,
    XGBModel,
    LightGBMModel,
    NHiTSModel,
    TBATS,
    Prophet,
)

# Local imports
from experiment.train_test import get_train_test
from hyperopt.analysis import best_hyperparameters
from hyperopt.config import all_coins, timeframes, n_periods
from hyperopt.search_space import model_config

# Ignore fbprophet warnings
logger = logging.getLogger("cmdstanpy")
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)


def get_model(model_name, coin, time_frame):
    # TODO: Also add the model unspecific parameters

    if model_name == "ARIMA":
        return StatsForecastAutoARIMA(
            start_p=0,
            start_q=0,
            start_P=0,
            start_Q=0,
            max_p=5,
            max_d=5,
            max_q=5,
            max_P=5,
            max_Q=5,
        )
    elif model_name == "RandomForest":
        return RandomForest(**best_hyperparameters(model_name, coin, time_frame))
    elif model_name == "XGB":
        return XGBModel(**best_hyperparameters(model_name, coin, time_frame))
    elif model_name == "LightGBM":
        return LightGBMModel(**best_hyperparameters(model_name, coin, time_frame))
    elif model_name == "Prophet":
        return Prophet(**best_hyperparameters(model_name, coin, time_frame))
    elif model_name == "TBATS":
        # https://medium.com/analytics-vidhya/time-series-forecasting-using-tbats-model-ce8c429442a9
        return TBATS(
            use_arma_errors=None,
            n_jobs=1,  # Seems to be quicker
        )
    elif model_name == "NBEATS":
        return NBEATSModel(
            **best_hyperparameters(model_name, coin, time_frame), model_name=model_name
        )
    elif model_name == "RNN":
        return RNNModel(
            **best_hyperparameters(model_name, coin, time_frame), model_name=model_name
        )
    elif model_name == "LSTM":
        return RNNModel(
            **best_hyperparameters(model_name, coin, time_frame),
            model_name=model_name,
            model="LSTM",
        )
    elif model_name == "GRU":
        return RNNModel(
            **best_hyperparameters(model_name, coin, time_frame),
            model_name=model_name,
            model="GRU",
        )
    elif model_name == "TCN":
        return TCNModel(
            **best_hyperparameters(model_name, coin, time_frame), model_name=model_name
        )
    elif model_name == "TFT":
        return TFTModel(
            **best_hyperparameters(model_name, coin, time_frame), model_name=model_name
        )
    elif model_name == "NHiTS":
        return NHiTSModel(
            **best_hyperparameters(model_name, coin, time_frame), model_name=model_name
        )
    else:
        raise ValueError(f"Model {model_name} is not supported.")


def generate_forecasts(model_name: str, coin: str, time_frame: str):
    # Get the training and testing data for each period
    train_set, test_set, time_series = get_train_test(
        coin=coin,
        time_frame=time_frame,
        n_periods=n_periods,
    )

    model = get_model(model_name, coin, time_frame)

    # Certain models need to be retrained for each period
    retrain = False
    train_length = None
    if model_name in ["Prophet", "TBATS", "ARIMA"]:
        retrain = True
        train_length = len(train_set[0])

    for period in tqdm(
        range(n_periods),
        desc=f"Forecasting periods for {model_name}/{coin}/{time_frame}",
        leave=False,
    ):
        model.fit(series=time_series[period])

        pred = model.historical_forecasts(
            time_series[period],
            start=len(train_set[period]),
            forecast_horizon=1,  # 1 step ahead forecasting
            stride=1,  # 1 step ahead forecasting
            retrain=retrain,
            train_length=train_length,
            verbose=True,
        )

        # Save all important information
        pred.pd_dataframe().to_csv(
            f"data/models/{model_name}/{coin}/{time_frame}/pred_{period}.csv"
        )
        train_set[period].pd_dataframe().to_csv(
            f"data/models/{model_name}/{coin}/{time_frame}/train_{period}.csv"
        )
        test_set[period].pd_dataframe().to_csv(
            f"data/models/{model_name}/{coin}/{time_frame}/test_{period}.csv"
        )


def forecast_model(model_name, start_from_coin="BTC", start_from_time_frame="1m"):
    for coin in all_coins[all_coins.index(start_from_coin) :]:
        for time_frame in timeframes[timeframes.index(start_from_time_frame) :]:
            # Create directories
            os.makedirs(f"data/models/{model_name}/{coin}/{time_frame}", exist_ok=True)

            generate_forecasts(model_name, coin, time_frame)


def forecast_all(
    start_from_model=None, start_from_coin=None, start_from_time_frame=None
):
    models = list(model_config) + ["ARIMA", "TBATS"]

    if start_from_model:
        models = models[models.index(start_from_model) :]

    for model in tqdm(models, desc="Generating forecast for all models", leave=False):
        coin = "BTC"
        time_frame = "1m"

        if start_from_coin and start_from_model == model:
            coin = start_from_coin
            if start_from_time_frame:
                time_frame = start_from_time_frame

        forecast_model(model, coin, time_frame)


def test_models():
    for model in list(model_config) + ["ARIMA", "TBATS"]:
        for coin in all_coins:
            for time_frame in timeframes:
                print(f"Testing {model} for {coin} {time_frame}")
                get_model(model, coin, time_frame)
