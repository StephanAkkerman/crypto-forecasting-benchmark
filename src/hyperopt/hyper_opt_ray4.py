from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from darts.models import NBEATSModel
from darts.metrics import rmse, mae
import matplotlib.pyplot as plt

# Local files
from config import config, search_alg, reporter, scheduler
from data import get_train_test

# load data
train_series, _ = get_train_test(coin="BTC", time_frame="1d", n_periods=9)


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


# n_epochs = 1, MAE 0.244, RMSE 0.278, 2 minutes
# n_epochs = 5, MAE 0.039, RMSE 0.049, 5 minutes
# n_epochs = 7, MAE 0.042, RMSE 0.04788, 8 minutes
# n_epochs = 10 , MAE 0.036, RMSE 0.045, 11 minutes
# ..., input_chunk_length = 48, MAE: 0.047, RMSE: 0.059 10 minutes
# n_epochs = 20 , MAE 0.036, RMSE 0.045, 21 minutes
def train_model(model_args, model_name: str, period: int, plot_trial=False):
    # Define a logger
    logger = TensorBoardLogger(save_dir="tb_logs", name="my_model")

    # Create the model using model_args from Ray Tune
    model = NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=1,  # 1 step ahead forecasting
        n_epochs=1,
        # torch_metrics=torch_metrics,
        pl_trainer_kwargs={
            # "callbacks": callbacks,
            "enable_progress_bar": False,
            "logger": logger,
            "accelerator": "auto",
        },
        **model_args,
    )

    val_len = int(0.1 * len(train_series[0]))
    val = train_series[period][-val_len:]

    # Train the model
    pred = model.historical_forecasts(
        series=train_series[period],
        start=len(train_series[period]) - val_len,
        forecast_horizon=1,
        stride=1,
        retrain=True,
        verbose=False,
    )

    # Calculate the metrics
    tune.report(mae=mae(val, pred), rmse=rmse(val, pred))

    if plot_trial:
        plot_results(val, pred)


def start_analysis(model_name):
    train_fn_with_parameters = tune.with_parameters(
        train_model,
        model_name=model_name,
    )

    # optimize hyperparameters by minimizing the MAPE on the validation set
    analysis = tune.run(
        train_fn_with_parameters,
        resources_per_trial={"cpu": 12, "gpu": 1},  # CPU number is the number of cores
        config=config,
        num_samples=1,  # the number of combinations to try
        scheduler=scheduler,
        metric="rmse",
        mode="min",  # "min" or "max
        progress_reporter=reporter,
        search_alg=search_alg,
        # local_dir="ray_results",
        # name="NBEATS",
        trial_name_creator=lambda trial: f"{model_name}_{trial.trial_id}",  # f"NBEATS_{trial.trainable_name}_{trial.trial_id}"
        verbose=1,  # 0: silent, 1: only status updates, 2: status and trial results 3: most detailed
        # trial_dirname_creator=custom_trial_name,
    )

    best = analysis.get_best_config(metric="rmse", mode="min")
    print(
        f"Best config: {best}\nHad a RMSE of {analysis.best_result['rmse']} and MAE of {analysis.best_result['mae']}"
    )

    # TODO: Save the best model (weights)


if __name__ == "__main__":
    start_analysis("NBEATS")
