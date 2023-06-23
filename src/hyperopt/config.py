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
    # Lookback period
    "input_chunk_length": tune.choice([1, 3, 6, 9, 12, 24]),
    "n_epochs": tune.choice([25, 50, 75, 100]),
    "batch_size": tune.choice([16, 32, 64, 128, 256]),
    "dropout": tune.uniform(0, 0.5),
}

# define the hyperparameter space
model_config = {
    ## Regression Models
    # https://unit8co.github.io/darts/generated_api/darts.models.forecasting.random_forest.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor
    "RandomForest": {
        "n_estimators": tune.choice([10, 100, 250, 500, 1000]),
        "max_depth": tune.choice([None, 2, 4, 8, 10, 12]),
    },
    # https://unit8co.github.io/darts/generated_api/darts.models.forecasting.xgboost.html
    # https://xgboost.readthedocs.io/en/stable/parameter.html
    # https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor
    "XGB": {
        "subsample": tune.uniform(0.8, 1),
        "max_leaves": tune.choice([0, 10, 25, 50, 100, 200]),
        "max_depth": tune.choice([None, 5, 10, 20, 30]),
        "gamma": tune.uniform(0, 0.02),
        "colsample_bytree": tune.uniform(0.8, 1),
        "min_child_weight": tune.choice([0, 2, 5, 7, 10]),
    },
    # https://unit8co.github.io/darts/generated_api/darts.models.forecasting.lgbm.html
    # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMRegressor.html
    "LightGBM": {
        "num_leaves": tune.choice([31, 100, 200, 400, 600]),
        "n_estimators": tune.choice([50, 80, 100, 150]),
        "max_depth": tune.choice([-1, 0, 40, 80]),
        "min_child_samples": tune.choice([20, 35, 50, 65]),
        "reg_alpha": tune.uniform(0, 10),  # L1 regularization
        "reg_lambda": tune.uniform(0, 0.1),  # L2 regularization
    },
    ## Machine Learning Models
    # https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nbeats.html
    "NBEATS": {
        "num_layers": tune.choice([2, 3, 4]),
        "num_blocks": tune.choice([1, 2, 3, 5, 10]),
        "layer_widths": tune.choice([256, 512, 1024]),
    },
    # https://unit8co.github.io/darts/generated_api/darts.models.forecasting.rnn_model.html
    "RNN": {
        "hidden_dim": tune.choice([25]),
        "n_rnn_layers": tune.choice([1, 2, 3, 4]),
        "training_length": tune.choice([25, 50, 75])
        # should have a higher value than input_chunk_length
    },
    "LSTM": {
        "hidden_dim": tune.choice([25]),
        "n_rnn_layers": tune.choice([1, 2, 3, 4]),
    },
    "GRU": {
        "hidden_dim": tune.choice([25]),
        "n_rnn_layers": tune.choice([1, 2, 3, 4]),
    },
    # https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tcn_model.html
    "TCN": {
        "kernel_size ": tune.choice([3, 5, 7, 9]),
        "num_filters": tune.choice([3, 8, 16, 24, 32]),
        "dilation_base": tune.choice([2, 4, 8, 16, 32]),
        "num_layers": tune.choice([None, 8, 12, 16, 20]),
    },
    # https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tft_model.html
    "TFT": {
        "hidden_size": tune.choice([2, 4, 8, 12, 16]),
        "lstm_layers": tune.choice([1, 2, 3, 4]),
        "num_attention_heads": tune.choice([1, 2, 3, 4]),
        "hidden_continuous_size": tune.choice([2, 4, 8, 10, 12]),
    },
    # https://unit8co.github.io/darts/generated_api/darts.models.forecasting.nhits.html
    "NHiTS": {
        "num_stacks": tune.choice([2, 3, 4]),
        "num_blocks": tune.choice([1, 2, 3, 50, 10]),
        "num_layers": tune.choice([1, 2, 3, 4]),
        "layer_widths": tune.choice([256, 512, 1024]),
    },
    # https://unit8co.github.io/darts/generated_api/darts.models.forecasting.tbats_model.html
    # https://github.com/intive-DataScience/tbats
    "TBATS": {
        "use_box_cox": tune.choice([True, False]),
        "use_trend": tune.choice([True, False]),
        "seasonal_periods": tune.choice([None, 7, 30]),
        "use_arma_errors": tune.choice([True, False]),
    },
    # https://unit8co.github.io/darts/generated_api/darts.models.forecasting.prophet_model.html
    # https://github.com/facebook/prophet/blob/main/python/prophet/forecaster.py
    # https://facebook.github.io/prophet/docs/diagnostics.html
    "Prophet": {
        "growth": tune.choice(["linear", "logistic"]),
        "changepoint_prior_scale": tune.choice([0.001, 0.01, 0.1, 0.5, 1]),
        "seasonality_prior_scale": tune.choice([0.01, 0.1, 1.0, 10.0]),
    },
}


def get_reporter(model_name):
    return CLIReporter(
        parameter_columns=list(model_config[model_name].keys()),
        metric_columns=["loss", "mae", "rmse"],
    )
