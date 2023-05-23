from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from torchmetrics import (
    MeanSquaredError,
    MeanAbsoluteError,
    MetricCollection,
)
from darts.models import NBEATSModel
from darts.metrics import rmse, mae

# Local files
from config import config, stopper, tune_callback, search_alg, reporter, scheduler
from data import get_train_test

# load data
trains, tests = get_train_test(coin="BTC", time_frame="1d", n_periods=9)


def train_model(model_args, callbacks):
    # This is necessary for the TuneReportCallback
    torch_metrics = MetricCollection(
        [MeanAbsoluteError(), MeanSquaredError(squared=False)]
    )

    # Define a logger
    logger = TensorBoardLogger(save_dir="tb_logs", name="my_model")

    # Create the model using model_args from Ray Tune
    model = NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=1,  # 1 step ahead forecasting
        n_epochs=1,
        torch_metrics=torch_metrics,
        pl_trainer_kwargs={
            "callbacks": callbacks,
            "enable_progress_bar": False,
            "logger": logger,
            "accelerator": "auto",
        },
        **model_args,
    )

    NBEATSModel.backtest

    # Merge this with the get_data()
    val_len = int(0.1 * len(trains[0]))
    n_periods = 1

    total_mae = 0
    total_rmse = 0
    for period in range(n_periods):
        print("PERIOD:", period, "\n")

        pred = model.backtest(
            series=trains[period],
            start=len(trains[period]) - val_len,
            forecast_horizon=1,
            stride=1,
            retrain=True,
            verbose=False,
            metric=[mae, rmse],
            last_points_only=True,
        )

        total_mae += pred[0]
        total_rmse += pred[1]

    # Average test loss
    # Tune reports the metrics back to its optimization engine
    tune.report(mae=total_mae / n_periods, rmse=total_rmse / n_periods)


train_fn_with_parameters = tune.with_parameters(
    train_model, callbacks=[tune_callback]  # [stopper, tune_callback],
)

# optimize hyperparameters by minimizing the MAPE on the validation set
analysis = tune.run(
    train_fn_with_parameters,
    resources_per_trial={"cpu": 12, "gpu": 1},  # CPU number is the number of cores
    config=config,
    num_samples=2,  # the number of combinations to try
    scheduler=scheduler,
    metric="rmse",
    mode="min",  # "min" or "max
    progress_reporter=reporter,
    search_alg=search_alg,
    # local_dir="ray_results",
    # name="NBEATS",
    trial_name_creator=lambda trial: f"NBEATS_{trial.trial_id}",  # f"NBEATS_{trial.trainable_name}_{trial.trial_id}"
    verbose=1,  # 0: silent, 1: only status updates, 2: status and trial results 3: most detailed
    # trial_dirname_creator=custom_trial_name,
)

best = analysis.get_best_config(metric="rmse", mode="min")
print(
    f"Best config: {best}\nHad a RMSE of {analysis.best_result['rmse']} and MAE of {analysis.best_result['mae']}"
)
