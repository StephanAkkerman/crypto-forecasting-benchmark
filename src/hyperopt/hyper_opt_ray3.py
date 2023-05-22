from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, rmse
from darts.models import NBEATSModel
from darts.datasets import AirPassengersDataset

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.skopt import SkOptSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback

# Load the data and perform scaling
series = AirPassengersDataset().load()
scaler = Scaler()
series_scaled = scaler.fit_transform(series)

# Define the search space for hyperparameters
config = {
    "input_chunk_length": 12,
    "num_layers": tune.choice([1, 2, 3]),
    "num_blocks": tune.choice([3, 4, 5]),
    "layer_widths": tune.choice([10, 20, 30]),
}

tune_callback = TuneReportCallback(
    {
        "loss": "val_loss",
        "mape": "val_MeanAbsolutePercentageError",
        "rmse": "val_MeanSquaredError",
    },
    on="validation_end",
)


# Define the objective function to minimize
def objective(config):
    model = NBEATSModel(
        input_chunk_length=config["input_chunk_length"],
        output_chunk_length=1,  # Set to 1 for one-step-ahead forecasting
        num_blocks=int(config["num_blocks"] * config["num_layers"]),
        layer_widths=int(config["layer_widths"] * config["num_layers"]),
        n_epochs=1,
        random_state=0,
        pl_trainer_kwargs={
            "accelerator": "auto",
            "enable_progress_bar": False,
            "callbacks": [tune_callback],
        },
    )

    # Splitting into non-overlapping training and test sets
    test_size_percentage = 0.25
    n_periods = 1
    test_size = int(len(series_scaled) / (1 / test_size_percentage - 1 + n_periods))
    train_size = int(test_size * (1 / test_size_percentage - 1))

    total_mape = 0
    total_rmse = 0
    for i in range(n_periods):
        start = i * (train_size + test_size)
        end = start + train_size

        # Initial training data
        train_ts = series_scaled[start:end]

        # Train the model on the current training data
        # model.fit(series=train_ts, verbose=True)

        pred = model.backtest(
            series=series_scaled,
            start=end,
            forecast_horizon=1,
            stride=1,
            retrain=True,
            verbose=False,
            metric=[mape, rmse],
        )

        total_mape += pred[0]
        total_rmse += pred[1]

    # Average test loss
    avg_mape = total_mape / n_periods
    avg_rmse = total_rmse / n_periods

    # Tune reports the metrics back to its optimization engine
    tune.report(mape=avg_mape, rmse=avg_rmse)


# Define a reporter to track progress
reporter = CLIReporter(
    # parameter_columns=list(config.keys()),
    metric_columns=["mape", "rmse", "training_iteration"],
)

# Use Ray Tune to perform hyperparameter tuning
# https://docs.ray.io/en/latest/_modules/ray/tune/tune.html
tune.run(
    objective,
    resources_per_trial={"cpu": 12, "gpu": 1},
    config=config,
    num_samples=10,
    scheduler=ASHAScheduler(
        metric="rmse", mode="min", max_t=100, grace_period=1, reduction_factor=2
    ),
    search_alg=SkOptSearch(metric="rmse", mode="min"),
    progress_reporter=reporter,
    verbose=1,  # 0: silent, 1: only status updates, 2: status and trial results 3: most detailed
    # stop={"training_iteration": 3},
)
