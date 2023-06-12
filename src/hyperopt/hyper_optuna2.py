import numpy as np
import optuna
import time
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping

from darts.metrics import smape, rmse
from darts.models import NBEATSModel
from darts.utils.likelihood_models import GaussianLikelihood
import matplotlib.pyplot as plt

from pytorch_pruning import PyTorchLightningPruningCallback

from data import get_train_test

# load data
train_series, _ = get_train_test(coin="BTC", time_frame="1d", n_periods=9)

# Best value: 0.04120539551300215, Best params: {'dropout': 0.11387285206833504}
# Took 627.7269632816315 seconds to run.


def plot_results(val, pred):
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(val.univariate_values(), label="Test Set")
    plt.plot(pred.univariate_values(), label="Forecast")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Test Set vs. Forecast")
    plt.show()
    plt.close()


# define objective function
def objective(trial):
    # throughout training we'll monitor the validation loss for both pruning and early stopping
    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=3, verbose=True)
    callbacks = [pruner]  # [early_stopper, pruner]

    # reproducibility
    torch.manual_seed(42)

    # build the TCN model
    model = NBEATSModel(
        input_chunk_length=24,
        output_chunk_length=1,
        batch_size=16,  # trial.suggest_int("batch_size", 16, 16),
        n_epochs=10,
        num_blocks=2,
        num_stacks=32,
        dropout=trial.suggest_float("dropout", 0.1, 0.2),
        pl_trainer_kwargs={
            "accelerator": "auto",
            "callbacks": callbacks,
            "enable_progress_bar": False,
        },
        # model_name="tcn_model",
        # force_reset=True,
        # save_checkpoints=True,
    )
    # Merge this with the get_data()
    val_len = int(0.1 * len(train_series[0]))
    val = train_series[0][-val_len:]
    n_periods = 1

    pred = model.historical_forecasts(
        series=train_series[0],
        start=len(train_series[0]) - val_len,
        forecast_horizon=1,
        stride=1,
        retrain=True,
        verbose=False,
    )

    return rmse(val, pred)


# for convenience, print some optimization trials information
def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


# optimize hyperparameters by minimizing the sMAPE on the validation set
if __name__ == "__main__":
    # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html
    study = optuna.create_study(direction="minimize")
    # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
    start = time.time()
    study.optimize(
        objective,
        n_trials=1,
        callbacks=[print_callback],
        n_jobs=1,
        show_progress_bar=False,
    )
