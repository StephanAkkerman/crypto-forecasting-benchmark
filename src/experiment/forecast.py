from tqdm import tqdm

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


def one_step_forecast(model_name, train, test) -> list:
    model = get_model(model_name)

    if model is None:
        return

    forecast = []

    # Loop over each period in the test set
    for t in tqdm(range(len(test))):
        model.fit(train)
        # Only use one, for one-step-ahead forecasting
        prediction = model.predict(n=1)
        forecast.append(prediction.first_value())
        # Add the current test value to the train set for the next loop
        train = train.append(test[t])

    return forecast


def get_model(model_name: str):
    if model_name == "arima":  # Basically hyperparameter tuning for ARIMA
        # https://nixtla.github.io/statsforecast/models.html#arima-methods
        model = StatsForecastAutoARIMA()

    elif model_name == "rnn":
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
    # TODO add DeepAR
    else:
        print("Invalid model name")
        return

    return model


def generate_forecasts(
    model_name: str, coin: str, time_frame: str, n_periods=9, show_plot=True
):
    model_name = model_name.lower()

    # Get the training and testing data for each period
    trains, tests = get_train_test(
        coin=coin, time_frame=time_frame, n_periods=n_periods
    )

    # Store the one-step-ahead forecasts for each period
    predictions = []

    for i, (train, test) in enumerate(zip(trains, tests)):
        print(f"Training on period {i + 1}...")
        predictions.append(one_step_forecast(model_name, train, test))

    eval_model(model_name, coin, time_frame, trains, tests, predictions, show_plot)
