import os
from tqdm import tqdm
import pandas as pd

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
from experiment.eval import eval_model
from hyperopt.data import models, all_coins, timeframes


def get_model(model_name: str):
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found in {models}")

    if model_name == "ARIMA":  # Basically hyperparameter tuning for ARIMA
        # https://nixtla.github.io/statsforecast/models.html#arima-methods
        model = StatsForecastAutoARIMA(
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

    elif model_name == "RNN":
        model = RNNModel(
            input_chunk_length=30,
            training_length=248,
            output_chunk_length=1,  # 1 for one-step-ahead forecasting
            force_reset=True,  # This should be done with every new period (or change model_name per period)
            pl_trainer_kwargs={"accelerator": "gpu", "devices": [0]},  # for GPU
            n_epochs=5,
            random_state=42,
            # save_checkpoints=True,
            # log_tensorboard=True,
            # work_dir=f"data/models/rnn/{coin}",
        )
    elif model_name == "lstm":
        model = RNNModel(
            model="LSTM",
            pl_trainer_kwargs={"accelerator": "gpu", "devices": [0]},  # for GPU
        )
    elif model_name == "gru":
        model = RNNModel(
            model="GRU",
            pl_trainer_kwargs={"accelerator": "gpu", "devices": [0]},  # for GPU
        )
    elif model_name == "tcn":
        model = TCNModel()
    elif model_name == "nbeats":
        model = NBEATSModel()
    elif model_name == "tft":
        model = TFTModel()
    elif model_name == "random forest":
        model = RandomForest()
    elif model_name == "xgboost":
        model = XGBModel()
    elif model_name == "lightgbm":
        model = LightGBMModel()
    elif model_name == "nhits":
        model = NHiTSModel()
    elif model_name == "tbats":
        model = TBATS()
    elif model_name == "prophet":
        model = Prophet()

    return model


def all_forecasts(model_name):
    model_name = model_name.upper()

    for coin in tqdm(all_coins):
        create_dirs(model_name, coin)

        print(f"Generating forecasts for {coin}...")
        for time_frame in tqdm(timeframes):
            generate_forecasts(
                model_name, coin, time_frame, n_periods=5, show_plot=False
            )


def create_dirs(model_name, coin):
    # Add the folders if they don't exist
    if not os.path.exists(f"data/models/{model_name}/{coin}"):
        if not os.path.exists(f"data/models/{model_name}"):
            if not os.path.exists("data/models"):
                os.makedirs("data/models")
            os.makedirs(f"data/models/{model_name}")
        os.makedirs(f"data/models/{model_name}/{coin}")
        os.makedirs(f"data/models/{model_name}/{coin}/plots")


def generate_forecasts(
    model_name: str, coin: str, time_frame: str, n_periods=5, show_plot=True
):
    model_name = model_name.upper()

    # Get the training and testing data for each period
    train_set, test_set, time_series = get_train_test(
        coin=coin,
        time_frame=time_frame,
        n_periods=n_periods,
    )

    model = get_model(model_name)

    predictions = []
    params = []

    for period in tqdm(range(n_periods)):
        model.fit(series=time_series[period])

        pred = model.historical_forecasts(
            time_series[period],
            start=len(time_series[period]) - len(test_set[period]),
            forecast_horizon=1,  # 1 step ahead forecasting
            stride=1,  # 1 step ahead forecasting
            retrain=True,
            train_length=len(time_series[period]) - len(test_set[period]),
            verbose=False,
        )

        predictions.append(pred)
        params.append(model.model.model_["arma"])

    # Save params
    params_df = pd.DataFrame(params, columns=["p", "d", "q", "P", "D", "Q", "C"])
    params_df.to_csv(
        f"data/models/{model_name}/{coin}/{time_frame}_params.csv", index=False
    )

    eval_model(
        model_name, coin, time_frame, train_set, test_set, predictions, show_plot
    )
