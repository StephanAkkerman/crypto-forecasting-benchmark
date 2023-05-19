import pandas as pd
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
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

from darts.dataprocessing.transformers import Scaler
from darts.datasets import AirPassengersDataset
from darts.models import NBEATSModel


def get_data():
    # Read data:
    series = AirPassengersDataset().load()

    # Create training and validation sets:
    train, val = series.split_after(pd.Timestamp(year=1957, month=12, day=1))

    # Normalize the time series (note: we avoid fitting the transformer on the validation set)
    transformer = Scaler()
    transformer.fit(train)
    train = transformer.transform(train)
    val = transformer.transform(val)

    return train, val


def train_model(model_args, callbacks, train, val):
    # This is necessary for the TuneReportCallback
    torch_metrics = MetricCollection(
        [MeanAbsolutePercentageError(), MeanAbsoluteError()]
    )

    # Define a logger
    logger = TensorBoardLogger("tb_logs", name="my_model")

    # Create the model using model_args from Ray Tune
    model = NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=12,
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
# https://docs.ray.io/en/latest/tune/api/doc/ray.tune.integration.pytorch_lightning.TuneReportCallback.html
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

# Customizable command-line reporter that provides information about a tuning run.
# https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CLIReporter.html
reporter = CLIReporter(
    parameter_columns=list(config.keys()),
    metric_columns=["loss", "MAPE", "training_iteration"],
)

train, val = get_data()
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
    metric="MAPE",  # any value in TuneReportCallback.
    mode="min",
    config=config,
    num_samples=1,  # the number of combinations to try
    scheduler=ASHAScheduler(max_t=1000, grace_period=3, reduction_factor=2),
    progress_reporter=reporter,
    search_alg=SkOptSearch(),
    local_dir="ray_results",
    name="NBEATS",
    trial_name_creator=lambda trial: f"NBEATS_{trial.trial_id}",  # f"NBEATS_{trial.trainable_name}_{trial.trial_id}"
    # trial_dirname_creator=custom_trial_name,
)

print("Best hyperparameters found were: ", analysis.best_config)
