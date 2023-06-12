import os
from ray import tune
from darts.models import NBEATSModel
from darts.metrics import rmse, mae
import matplotlib.pyplot as plt

# Local files
from config import config, search_alg, get_reporter, scheduler
from data import get_train_test


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

    # Create the model using model_args from Ray Tune
    model = NBEATSModel(
        input_chunk_length=5,  # Need some experimentation
        output_chunk_length=1,  # 1 step ahead forecasting
        dropout=0.1,
        pl_trainer_kwargs={
            "enable_progress_bar": False,
            "accelerator": "auto",
        },
        model_name=model_name,
        **model_args,
    )

    val_len = int(0.1 * len(train_series[0]))
    val = train_series[period][-val_len:]

    # Train the model
    pred = model.historical_forecasts(
        series=train_series[period],
        start=len(train_series[period]) - val_len,
        forecast_horizon=1,  # 1 step ahead forecasting
        stride=1,  # 1 step ahead forecasting
        retrain=True,
        verbose=False,
    )

    # Calculate the metrics
    rmse_val = rmse(val, pred)
    tune.report(mae=mae(val, pred), rmse=rmse_val)

    # Add metric score to checkpoint_dir
    if save_checkpoints:
        new_model_name = f"{model_name}_{round(rmse_val, 3)}"
        os.rename(
            os.path.join(checkpoint_loc, model_name),
            os.path.join(checkpoint_loc, new_model_name),
        )

    if plot_trial:
        plot_results(val, pred)


def start_analysis(model_name, period, coin, time_frame):
    # load data
    train_series, _ = get_train_test(coin=coin, time_frame=time_frame)

    train_fn_with_parameters = tune.with_parameters(
        train_model,
        model_name=f"{model_name}_{coin}_{time_frame}_{period}",
        checkpoint_loc=os.path.join(os.getcwd(), "darts_logs"),
        period=period,
        train_series=train_series,
        save_checkpoints=False,
    )

    # optimize hyperparameters by minimizing the MAPE on the validation set
    analysis = tune.run(
        train_fn_with_parameters,
        resources_per_trial={"cpu": 12, "gpu": 1},  # CPU number is the number of cores
        config=config[model_name],
        num_samples=1,  # the number of combinations to try
        scheduler=scheduler,
        metric="rmse",
        mode="min",  # "min" or "max
        progress_reporter=get_reporter(model_name),
        search_alg=search_alg,
        verbose=2,  # 0: silent, 1: only status updates, 2: status and trial results 3: most detailed
        local_dir=f"hyperopt_results",
        name=f"{model_name}_{coin}_{time_frame}_{period}",  # folder in local_dir
        trial_name_creator=lambda trial: model_name,  # folder in name file
    )

    best = analysis.get_best_config(metric="rmse", mode="min")
    print(
        f"Best config: {best}\nHad a RMSE of {analysis.best_result['rmse']} and MAE of {analysis.best_result['mae']}"
    )

    # Save best config + results to file


if __name__ == "__main__":
    start_analysis("NBEATS", 0, "BTC", "1d")
