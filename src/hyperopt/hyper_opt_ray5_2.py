import os
import matplotlib.pyplot as plt
from ray import tune
from darts.metrics import rmse, mae

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

# Local files
from config import config, search_alg, get_reporter, scheduler
from data import get_train_test, all_coins, timeframes, models


def plot_results(val, pred):
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(val.univariate_values(), label="Test Set")
    plt.plot(pred.univariate_values(), label="Forecast")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Test Set vs. Forecast")
    plt.show()
    plt.close()


def get_model(full_model_name: str, model_args: dict):
    model_name = full_model_name.split("_")[0]

    if model_name.split("_")[0] not in models:
        raise ValueError(f"Model {model_name} not found in {models}")

    default_args = {
        "output_chunk_length": 1,  # 1 step ahead forecasting
        "pl_trainer_kwargs": {
            "enable_progress_bar": False,
            "accelerator": "auto",
        },
    }

    # add default args to model_args
    model_args.update(default_args)

    if model_name == "ARIMA":
        return StatsForecastAutoARIMA(model_name=full_model_name, **model_args)
    elif model_name == "NBEATS":
        return NBEATSModel(model_name=full_model_name, **model_args)


# n_epochs = 10 , MAE 0.036, RMSE 0.045, 11 minutes
# ..., input_chunk_length = 48, MAE: 0.047, RMSE: 0.059 10 minutes
# n_epochs = 20 , MAE 0.036, RMSE 0.045, 21 minutes
def train_model(
    model_args: dict,
    model_name: str,
    checkpoint_loc: str,
    period: int,
    train_series,
    plot_trial=False,
    save_checkpoints=False,
):
    if save_checkpoints:
        model_args.update({"save_checkpoints": True, "force_reset": True})

    model = get_model(model_name, model_args)

    val_len = int(0.1 * len(train_series[0]))
    val = train_series[period][-val_len:]

    # Train the model
    model.fit(series=train_series[period][:-val_len], verbose=False)

    # Evaluate the model
    pred = model.historical_forecasts(
        series=train_series[period],
        start=len(train_series[period]) - val_len,
        forecast_horizon=1,  # 1 step ahead forecasting
        stride=1,  # 1 step ahead forecasting
        retrain=False,
        verbose=False,
    )

    # Calculate the metrics
    rmse_val = rmse(val, pred)
    tune.report(mae=mae(val, pred), rmse=rmse_val)

    # Add metric score to checkpoint_dir
    if save_checkpoints:
        new_model_name = f"{model_name}_{round(rmse_val, 3)}"
        model.save(path=os.path.join(checkpoint_loc, new_model_name))
        # os.rename(
        #    os.path.join(checkpoint_loc, model_name),
        #    os.path.join(checkpoint_loc, new_model_name),
        # )

    if plot_trial:
        plot_results(val, pred)


def hyperopt(
    train_series,
    model_name: str,
    period: int,
    coin: str,
    time_frame: str,
    num_samples: int,
):
    train_fn_with_parameters = tune.with_parameters(
        train_model,
        model_name=f"{model_name}_{coin}_{time_frame}_{period}",
        checkpoint_loc=os.path.join(os.getcwd(), "darts_logs"),
        period=period,
        train_series=train_series,
        save_checkpoints=False,
    )

    # optimize hyperparameters by minimizing the MAPE on the validation set
    # https://docs.ray.io/en/latest/tune/key-concepts.html#analysis
    analysis = tune.run(
        train_fn_with_parameters,
        resources_per_trial={"cpu": 12, "gpu": 1},  # CPU number is the number of cores
        config=config[model_name],
        num_samples=num_samples,  # the number of combinations to try
        scheduler=scheduler,
        metric="rmse",
        mode="min",  # "min" or "max
        progress_reporter=get_reporter(model_name),
        search_alg=search_alg,
        verbose=2,  # 0: silent, 1: only status updates, 2: status and trial results 3: most detailed
        local_dir="ray_results",
        name=f"{model_name}_{coin}_{time_frame}_{period}",  # folder in local_dir
        trial_name_creator=lambda trial: model_name,  # folder in name file
    )

    # Save the results
    analysis.results_df.to_csv(
        f"hyperopt_results/{model_name}_{coin}_{time_frame}_{period}.csv", index=False
    )


def hyperopt_dataset(model_name: str, coin: str, time_frame: str, num_samples: int):
    """
    Performs hyperparameter optimization for a given dataset.

    Parameters
    ----------
    model_name : str
        The name of the model to be used.
    coin : str
        The name of the coin to be used.
    time_frame : str
        The time frame to be used.
    num_samples : int
        The number of samples to be used.
    """

    # load data
    train_series, _ = get_train_test(coin=coin, time_frame=time_frame)

    # Do for all periods
    for period in range(0, 5):
        hyperopt(train_series, model_name, period, coin, time_frame, num_samples)


def hyperopt_full():
    # for model in models:
    for coin in all_coins:
        for tf in timeframes:
            hyperopt_dataset("NBEATS", coin, tf, 10)


if __name__ == "__main__":
    hyperopt_dataset("NBEATS", "BTC", "1d", 1)
