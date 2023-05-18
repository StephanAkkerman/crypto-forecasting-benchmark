from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape
from darts.datasets import AirPassengersDataset
import pandas as pd
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.skopt import SkOptSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune import CLIReporter
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchmetrics import (
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MetricCollection,
)

# Suppose you have two TimeSeries instances: `train` and `val`
# Read data:
series = AirPassengersDataset().load()

# Create training and validation sets:
train, val = series.split_after(pd.Timestamp(year=1957, month=12, day=1))

# Normalize the time series (note: we avoid fitting the transformer on the validation set)
transformer = Scaler()
transformer.fit(train)
train = transformer.transform(train)
val = transformer.transform(val)

# https://lightning.ai/docs/pytorch/stable/common/early_stopping.html
my_stopper = EarlyStopping(
    monitor="val_loss",
    patience=5,
    min_delta=0.05,
    mode="min",
)

# https://docs.ray.io/en/latest/tune/api/doc/ray.tune.integration.pytorch_lightning.TuneReportCallback.html
tune_callback = TuneReportCallback(
    {
        "loss": "val_Loss",
        "MAPE": "val_MeanAbsolutePercentageError",
    },
    on="validation_end",
)

# This is necessary for the TuneReportCallback
torch_metrics = MetricCollection([MeanAbsolutePercentageError(), MeanAbsoluteError()])


def train_model(model_args, callbacks, train, val):
    model = NBEATSModel(
        # input_chunk_length=config["input_chunk_length"],
        output_chunk_length=1,
        # n_epochs=config["n_epochs"],
        random_state=0,
        pl_trainer_kwargs={
            "callbacks": callbacks,
            "accelerator": "gpu",
            "devices": [0],
        },
        **model_args,
        torch_metrics=torch_metrics,
    )
    model.fit(series=train, val_series=val)

    # Otherwise MAPE will not be returned
    # This probably overwrites the TuneReportCallback
    # val_loss = mape(model.predict(len(val)), val)
    # tune.report(loss=val_loss)


# Necesasry for the reporter and tune.run
config = {
    "input_chunk_length": tune.choice([10, 20, 30]),
    "n_epochs": tune.choice([10, 20, 30]),
}


# Customizable command-line reporter that provides information about a tuning run.
# https://docs.ray.io/en/latest/tune/api/doc/ray.tune.CLIReporter.html
reporter = CLIReporter(
    parameter_columns=list(config.keys()),
    metric_columns=["loss", "MAPE", "training_iteration"],
)

train_fn_with_parameters = tune.with_parameters(
    train_model,
    callbacks=[my_stopper, tune_callback],
    train=train,
    val=val,
)

analysis = tune.run(
    train_fn_with_parameters,
    resources_per_trial={"cpu": 8, "gpu": 1},
    config=config,
    metric="MAPE",  # any value in TuneReportCallback.
    mode="min",
    search_alg=SkOptSearch(),
    scheduler=ASHAScheduler(),
    progress_reporter=reporter,
)

print("Best config: ", analysis.get_best_config(metric="mape", mode="min"))
