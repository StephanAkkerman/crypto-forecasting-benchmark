import os
import pandas as pd
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.skopt import SkOptSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from torchmetrics import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MetricCollection,
)
from darts import TimeSeries
from darts.models import NBEATSModel


def read_csv(coin: str, timeframe: str, col_names: list = ["close"]):
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Go up two levels to the parent directory
    crypto_forecasting_folder = os.path.dirname(os.path.dirname(current_dir))

    # Go to the data folder
    data_folder = os.path.join(crypto_forecasting_folder, "data")

    # Go to the coins folder
    coins_folder = os.path.join(data_folder, "coins")

    df = pd.read_csv(f"{coins_folder}/{coin}/{coin}USDT_{timeframe}.csv")

    # Set date as index
    df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)

    return df[col_names]


# Load your data (replace this with your actual data)
def get_train_test(coin="BTC", time_frame="1d", n_periods=9, test_size_percentage=0.25):
    # Read data from a CSV file
    data = read_csv(coin, time_frame, ["log returns"]).dropna()
    data["date"] = data.index

    # Create a Darts TimeSeries from the DataFrame
    time_series = TimeSeries.from_dataframe(data, "date", "log returns")

    # Set parameters for sliding window and periods
    test_size = int(len(time_series) / (1 / test_size_percentage - 1 + n_periods))
    train_size = int(test_size * (1 / test_size_percentage - 1))

    print("Train size:", train_size)
    print("Test size:", test_size)

    # Save the training and test sets as lists of TimeSeries
    train_set = []
    test_set = []

    for i in range(n_periods):
        # The train start shifts by the test size each period
        train_start = i * test_size
        train_end = train_start + train_size

        train_set.append(time_series[train_start:train_end])
        test_set.append(time_series[train_end : train_end + test_size])

    return train_set, test_set


# load data
trains, tests = get_train_test(coin="BTC", time_frame="1d", n_periods=9)


def train_model(model_args, callbacks, train, val):
    # This is necessary for the TuneReportCallback
    torch_metrics = MetricCollection(
        [MeanAbsolutePercentageError(), MeanAbsoluteError()]
    )

    # Define a logger
    logger = TensorBoardLogger(save_dir="tb_logs", name="my_model")

    # Create the model using model_args from Ray Tune
    model = NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=1,  # 1 step ahead forecasting
        n_epochs=500,
        torch_metrics=torch_metrics,
        pl_trainer_kwargs={
            "callbacks": callbacks,
            "enable_progress_bar": False,
            "logger": logger,
            "accelerator": "gpu",
            "devices": [0],
        },
        **model_args,
    )

    model.fit(
        series=train,
        val_series=val,
    )


# Early stop callback
my_stopper = EarlyStopping(
    monitor="val_MeanAbsolutePercentageError",
    patience=5,
    min_delta=0.05,
    mode="min",
)

# set up ray tune callback
tune_callback = TuneReportCallback(
    {
        "loss": "val_loss",
        "MAPE": "val_MeanAbsolutePercentageError",
    },
    on="validation_end",
)

# define the hyperparameter space
config = {
    "batch_size": tune.choice([16, 32, 64, 128]),
    "num_blocks": tune.choice([1, 2, 3, 4, 5]),
    "num_stacks": tune.choice([32, 64, 128]),
    "dropout": tune.uniform(0, 0.2),
}

reporter = CLIReporter(
    parameter_columns=list(config.keys()),
    metric_columns=["loss", "MAPE", "training_iteration"],
)

# Merge this with the get_data()
val_len = int(0.1 * len(trains[0]))
print("Validation length: ", val_len)
for period in range(9):
    print("Period: ", period)
    for v in range(val_len):
        # train = trains[period][: -val_len + v]
        # val = trains[period][-val_len + v]

        train = trains[period][: -val_len + v]
        # Add the input_chunk_length to the validation set
        val = trains[period][-val_len + v - 24 : -val_len + v + 1]

        train_fn_with_parameters = tune.with_parameters(
            train_model,
            callbacks=[my_stopper, tune_callback],
            train=train,
            val=val,
        )

        # optimize hyperparameters by minimizing the MAPE on the validation set
        analysis = tune.run(
            train_fn_with_parameters,
            resources_per_trial={"cpu": 8, "gpu": 1},
            # Using a metric instead of loss allows for
            # comparison between different likelihood or loss functions.
            # metric="MAPE",  # any value in TuneReportCallback.
            # mode="min",
            config=config,
            num_samples=1,  # the number of combinations to try
            scheduler=ASHAScheduler(
                metric="MAPE",
                mode="min",
                max_t=1000,
                grace_period=3,
                reduction_factor=2,
            ),
            progress_reporter=reporter,
            search_alg=SkOptSearch(metric="MAPE", mode="min"),
            local_dir="ray_results",
            name="NBEATS",
            trial_name_creator=lambda trial: f"NBEATS_{trial.trial_id}",  # f"NBEATS_{trial.trainable_name}_{trial.trial_id}"
            # trial_dirname_creator=custom_trial_name,
        )
        print(
            "Best hyperparameters found were: ",
            analysis.get_best_config(metric="MAPE", mode="min"),
        )
