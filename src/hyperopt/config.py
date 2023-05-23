from pytorch_lightning.callbacks import EarlyStopping
from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.skopt import SkOptSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler

# define the hyperparameter space
config = {
    "batch_size": tune.choice([16, 32, 64, 128]),
    "num_blocks": tune.choice([1, 2, 3, 4, 5]),
    "num_stacks": tune.choice([32, 64, 128]),
    "dropout": tune.uniform(0, 0.2),
}

# Early stop callback
stopper = EarlyStopping(
    monitor="val_MeanSquaredError",
    patience=5,
    min_delta=0.05,
    mode="min",
)

# set up ray tune callback
tune_callback = TuneReportCallback(
    {
        "loss": "val_loss",
        "mae": "val_MeanAbsoluteError",
        "rmse": "val_MeanSquaredError",
    },
    on="validation_end",
)

reporter = CLIReporter(
    parameter_columns=list(config.keys()),
    metric_columns=["loss", "mae", "rmse", "training_iteration"],
)

scheduler = ASHAScheduler(
    # metric="rmse",
    # mode="min",
    max_t=100,
    grace_period=3,
    reduction_factor=2,
)

search_alg = SkOptSearch(metric="mape", mode="min")
