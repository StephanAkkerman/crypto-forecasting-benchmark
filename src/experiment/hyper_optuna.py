import numpy as np
import optuna
import torch
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.preprocessing import MaxAbsScaler

from darts import concatenate
from darts.dataprocessing.transformers import Scaler
from darts.metrics import smape
from darts.models import TCNModel
from darts.utils.likelihood_models import GaussianLikelihood

from experiment.train_test import get_train_test

# load data
trains, tests = get_train_test(coin="BTC", time_frame="1d", n_periods=9)


# define objective function
def objective(trial):
    # select input and output chunk lengths
    in_len = trial.suggest_int("in_len", 12, 36)
    out_len = 1

    # Other hyperparameters
    kernel_size = trial.suggest_int("kernel_size", 2, 5)
    num_filters = trial.suggest_int("num_filters", 1, 5)
    weight_norm = trial.suggest_categorical("weight_norm", [False, True])
    dilation_base = trial.suggest_int("dilation_base", 2, 4)
    dropout = trial.suggest_float("dropout", 0.0, 0.4)
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    include_year = trial.suggest_categorical("year", [False, True])

    # throughout training we'll monitor the validation loss for both pruning and early stopping
    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping("val_loss", min_delta=0.001, patience=3, verbose=True)
    callbacks = [pruner, early_stopper]

    # detect if a GPU is available
    if torch.cuda.is_available():
        pl_trainer_kwargs = {
            "accelerator": "gpu",
            "devices": [0],
            "callbacks": callbacks,
        }
    else:
        pl_trainer_kwargs = {"callbacks": callbacks}

    # optionally also add the (scaled) year value as a past covariate
    if include_year:
        encoders = {"datetime_attribute": {"past": ["year"]}, "transformer": Scaler()}
    else:
        encoders = None

    # reproducibility
    torch.manual_seed(42)

    # build the TCN model
    model = TCNModel(
        input_chunk_length=in_len,
        output_chunk_length=out_len,
        batch_size=32,
        n_epochs=100,
        nr_epochs_val_period=1,
        kernel_size=kernel_size,
        num_filters=num_filters,
        weight_norm=weight_norm,
        dilation_base=dilation_base,
        dropout=dropout,
        optimizer_kwargs={"lr": lr},
        add_encoders=encoders,
        likelihood=GaussianLikelihood(),
        pl_trainer_kwargs=pl_trainer_kwargs,
        model_name="tcn_model",
        force_reset=True,
        save_checkpoints=True,
    )

    # when validating during training, we can use a slightly longer validation
    # set which also contains the first input_chunk_length time steps
    period_smapes = []
    val_len = int(0.1 * len(trains[0]))
    for period in range(9):
        print("Period: ", period)
        one_step_ahead_smapes = []
        for val in range(val_len):
            train = trains[period][: -val_len + val]
            model_val_set = trains[period][-val_len + val - in_len : -val_len + val + 1]

            # train the model
            model.fit(
                series=train,
                val_series=model_val_set,
            )

            # reload best model over course of training
            model = TCNModel.load_from_checkpoint("tcn_model")

            # Evaluate how good it is on the validation set, using sMAPE
            preds = model.predict(series=train, n=1)
            smapes = smape(val, preds, n_jobs=-1, verbose=True)
            smape_val = np.mean(smapes)

            one_step_ahead_smapes.append(smape_val)
        period_smapes.append(np.mean(one_step_ahead_smapes))

    return np.mean(period_smapes)


# for convenience, print some optimization trials information
def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")


# optimize hyperparameters by minimizing the sMAPE on the validation set
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100, callbacks=[print_callback])
