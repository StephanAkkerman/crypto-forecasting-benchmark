import os
from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from torchmetrics import MeanSquaredError, MetricCollection, MeanAbsoluteError
from darts.models import NBEATSModel
from darts.metrics import mae, rmse
from darts.timeseries import concatenate
import matplotlib.pyplot as plt

# Local files
from config import config, stopper, tune_callback, search_alg, reporter, scheduler
from data import get_train_val

# load data
train_val = get_train_val()


# NBEATS: {'batch_size': 16, 'num_blocks': 2, 'num_stacks': 32, 'dropout': 0.025}
# n_epochs = 10, 30 minutes. mae: 0.086 rmse: 0.1
def train_model(model_args, callbacks):
    # This is necessary for the TuneReportCallback
    torch_metrics = MetricCollection(
        [MeanAbsoluteError(), MeanSquaredError(squared=False)]
    )

    # Define a logger
    # Can also try WandbLogger
    logger = TensorBoardLogger(save_dir="tb_logs", name="my_model")

    # Create the model using model_args from Ray Tune
    model = NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=1,  # 1 step ahead forecasting
        n_epochs=10,
        torch_metrics=torch_metrics,
        pl_trainer_kwargs={
            "callbacks": callbacks,
            "enable_progress_bar": False,
            "logger": logger,
            "accelerator": "auto",
        },
        # save_checkpoints=True,
        force_reset=True,
        model_name="nbeats_model",
        **model_args,
    )

    n_periods = 1

    all_val = []
    all_pred = []
    for period in range(len(train_val)):
        if period == n_periods:
            break

        for train, val in train_val[period]:
            model.fit(
                series=train,
                val_series=val,  # Necessary for EarlyStopping
                verbose=False,
                num_loader_workers=4,
            )

            # One-step-ahead forecasting
            pred = model.predict(n=1)
            all_val.append(val[-1])
            all_pred.append(pred)

    # Convert list to TimeSeries
    all_val = concatenate(all_val, ignore_time_axis=True)
    all_pred = concatenate(all_pred, ignore_time_axis=True)

    # Tune reports the metrics back to its optimization engine
    tune.report(mae=mae(all_val, all_pred), rmse=rmse(all_val, all_pred))

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(all_val.univariate_values(), label="Test Set")
    plt.plot(all_pred.univariate_values(), label="Forecast")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Test Set vs. Forecast")
    plt.show()
    plt.close()


train_fn_with_parameters = tune.with_parameters(
    train_model,
    callbacks=[stopper, tune_callback],
)

# optimize hyperparameters by minimizing the MAPE on the validation set
analysis = tune.run(
    train_fn_with_parameters,
    resources_per_trial={"cpu": 12, "gpu": 1},  # CPU number is the number of cores
    config=config,
    num_samples=1,  # the number of combinations to try
    scheduler=scheduler,
    metric="rmse",
    mode="min",  # "min" or "max
    progress_reporter=reporter,
    search_alg=search_alg,
    local_dir="ray_results",
    # name="NBEATS",
    trial_name_creator=lambda trial: f"NBEATS_{trial.trial_id}",  # f"NBEATS_{trial.trainable_name}_{trial.trial_id}"
    # log_to_file=False,
    verbose=1,
    # trial_dirname_creator=custom_trial_name,
)
best = analysis.get_best_config(metric="rmse", mode="min")
print(
    f"Best config: {best}\nHad a RMSE of {analysis.best_result['rmse']} and MAE of {analysis.best_result['mae']}"
)
