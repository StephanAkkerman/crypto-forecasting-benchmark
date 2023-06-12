from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

# define the hyperparameter space
config2 = {
    "NBEATS": {
        "n_epochs": tune.choice([10, 25, 50, 100]),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "optimizer_kwargs": {"lr": tune.loguniform(1e-4, 1e-1)},
        "dropout": tune.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "num_layers": tune.choice([2, 3, 4, 5]),
        "num_blocks": tune.choice([1, 2, 3, 4, 5]),
    }
}

config = {
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
        parameter_columns=list(config[model_name].keys()),
        metric_columns=["loss", "mae", "rmse"],
    )


# https://docs.ray.io/en/latest/tune/api/schedulers.html
# https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.AsyncHyperBandScheduler.html
scheduler = ASHAScheduler()

# https://docs.ray.io/en/latest/tune/api/suggestion.html
search_alg = OptunaSearch()  # SkOptSearch()
