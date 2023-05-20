import optuna
from optuna.integration import PyTorchLightningPruningCallback
import torch
import numpy as np
from tqdm import tqdm
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler

from darts.datasets import ElectricityDataset
from darts.models import TCNModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import smape

all_series = ElectricityDataset(multivariate=False).load()

NR_DAYS = 80
DAY_DURATION = 24 * 4  # 15 minutes frequency

all_series_fp32 = [
    s[-(NR_DAYS * DAY_DURATION) :].astype(np.float32) for s in tqdm(all_series)
]

# Split in train/val/test
val_len = 14 * DAY_DURATION  # 14 days

train = [s[: -(2 * val_len)] for s in all_series_fp32]
val = [s[-(2 * val_len) : -val_len] for s in all_series_fp32]
test = [s[-val_len:] for s in all_series_fp32]

# Scale so that the largest value is 1.
# This way of scaling perserves the sMAPE
scaler = Scaler(scaler=MaxAbsScaler())
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)


def build_fit_tcn_model(
    in_len,
    out_len,
    kernel_size,
    num_filters,
    weight_norm,
    dilation_base,
    dropout,
    lr,
    include_dayofweek,
    likelihood=None,
    callbacks=None,
):
    # reproducibility
    torch.manual_seed(42)

    # some fixed parameters that will be the same for all models
    BATCH_SIZE = 1024
    MAX_N_EPOCHS = 1
    NR_EPOCHS_VAL_PERIOD = 1
    MAX_SAMPLES_PER_TS = 1000

    # throughout training we'll monitor the validation loss for early stopping
    early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=3, verbose=True)
    if callbacks is None:
        callbacks = [early_stopper]
    else:
        callbacks = [early_stopper] + callbacks

    # detect if a GPU is available
    if torch.cuda.is_available():
        pl_trainer_kwargs = {
            "accelerator": "auto",
            "callbacks": callbacks,
        }
        num_workers = 4
    else:
        pl_trainer_kwargs = {"callbacks": callbacks}
        num_workers = 0

    # optionally also add the day of the week (cyclically encoded) as a past covariate
    encoders = {"cyclic": {"past": ["dayofweek"]}} if include_dayofweek else None

    # build the TCN model
    model = TCNModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        batch_size=BATCH_SIZE,
        n_epochs=MAX_N_EPOCHS,
        nr_epochs_val_period=NR_EPOCHS_VAL_PERIOD,
        kernel_size=kernel_size,
        num_filters=num_filters,
        weight_norm=weight_norm,
        dilation_base=dilation_base,
        dropout=dropout,
        optimizer_kwargs={"lr": lr},
        add_encoders=encoders,
        likelihood=likelihood,
        pl_trainer_kwargs=pl_trainer_kwargs,
        model_name="tcn_model",
        force_reset=True,
        save_checkpoints=True,
    )

    # when validating during training, we can use a slightly longer validation
    # set which also contains the first input_chunk_length time steps
    model_val_set = scaler.transform(
        [s[-((2 * val_len) + in_len) : -val_len] for s in all_series_fp32]
    )

    # train the model
    model.fit(
        series=train,
        val_series=model_val_set,
        max_samples_per_ts=MAX_SAMPLES_PER_TS,
        num_loader_workers=num_workers,
    )

    # reload best model over course of training
    model = TCNModel.load_from_checkpoint("tcn_model")

    return model


def objective(trial):
    callback = [PyTorchLightningPruningCallback(trial, monitor="val_loss")]

    # set input_chunk_length, between 5 and 14 days
    days_in = trial.suggest_int("days_in", 5, 14)
    in_len = days_in * DAY_DURATION

    # set out_len, between 1 and 13 days (it has to be strictly shorter than in_len).
    days_out = trial.suggest_int("days_out", 1, days_in - 1)
    out_len = days_out * DAY_DURATION

    # Other hyperparameters
    kernel_size = trial.suggest_int("kernel_size", 5, 25)
    num_filters = trial.suggest_int("num_filters", 5, 25)
    weight_norm = trial.suggest_categorical("weight_norm", [False, True])
    dilation_base = trial.suggest_int("dilation_base", 2, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    include_dayofweek = trial.suggest_categorical("dayofweek", [False, True])

    # build and train the TCN model with these hyper-parameters:
    model = build_fit_tcn_model(
        in_len=in_len,
        out_len=out_len,
        kernel_size=kernel_size,
        num_filters=num_filters,
        weight_norm=weight_norm,
        dilation_base=dilation_base,
        dropout=dropout,
        lr=lr,
        include_dayofweek=include_dayofweek,
        callbacks=callback,
    )

    # Evaluate how good it is on the validation set
    preds = model.predict(series=train, n=val_len)
    smapes = smape(val, preds, n_jobs=-1, verbose=True)
    smape_val = np.mean(smapes)

    return smape_val if smape_val != np.nan else float("inf")


def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


# This is the main function that will run the optimization
if __name__ == "__main__":
    # https://github.com/unit8co/darts/blob/master/examples/17-hyperparameter-optimization.ipynb
    # We use optuna to find the best hyperparameters
    study = optuna.create_study(direction="minimize")

    study.optimize(objective, timeout=100, callbacks=[print_callback])

    # We could also have used a command as follows to limit the number of trials instead:
    # study.optimize(objective, n_trials=100, callbacks=[print_callback])

    # Finally, print the best value and best hyperparameters:
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")
