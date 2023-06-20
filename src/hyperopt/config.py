from ray import tune
from ray.tune import CLIReporter

val_percentage = 0.1

# These are the default args for all models
default_args = {
    "output_chunk_length": 1,  # 1 step ahead forecasting
    "pl_trainer_kwargs": {
        "enable_progress_bar": False,
        "accelerator": "auto",
    },
}

# Except for autoARIMA
model_unspecific = {
    "input_chunk_length": tune.choice([1, 5, 10, 15, 20, 30, 40, 50]),
    "n_epochs": tune.choice([25, 50, 75, 100]),
    "batch_size": tune.choice([16, 32, 64, 128]),
    "dropout": tune.uniform(0, 0.5),
}

# define the hyperparameter space
model_config = {
    "ARIMA": {},  # auto arima is used
    "NBEATS": {
        "num_layers": tune.choice([2, 3, 4, 5]),
        "num_blocks": tune.choice([1, 2, 3, 4, 5]),
    },
}

test_config = {
    "NBEATS": {
        "n_epochs": 1,
        "batch_size": 16,
        "num_blocks": tune.choice([2, 3]),
        "num_stacks": tune.choice([32]),
        # "dropout": tune.uniform(0.1, 0.2),
    }
}


def get_reporter(model_name):
    return CLIReporter(
        parameter_columns=list(model_config[model_name].keys()),
        metric_columns=["loss", "mae", "rmse"],
    )
