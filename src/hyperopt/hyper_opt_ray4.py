import glob
import re
import os

from pytorch_lightning.loggers import TensorBoardLogger
from ray import tune
from ray.air.checkpoint import Checkpoint
from darts.models import NBEATSModel
from darts.metrics import rmse, mae
import matplotlib.pyplot as plt

# Local files
from config import config, search_alg, reporter, scheduler
from data import get_train_test

# load data
train_series, _ = get_train_test(coin="BTC", time_frame="1d", n_periods=9)


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


# n_epochs = 1, MAE 0.244, RMSE 0.278, 2 minutes
# n_epochs = 5, MAE 0.039, RMSE 0.049, 5 minutes
# n_epochs = 7, MAE 0.042, RMSE 0.04788, 8 minutes
# n_epochs = 10 , MAE 0.036, RMSE 0.045, 11 minutes
# ..., input_chunk_length = 48, MAE: 0.047, RMSE: 0.059 10 minutes
# n_epochs = 20 , MAE 0.036, RMSE 0.045, 21 minutes
class TrainModel(tune.Trainable):
    def setup(self, config, model_name: str, period: int, plot_trial=False):
        self.period = period
        self.best_rmse = float("inf")

        self.model = NBEATSModel(
            input_chunk_length=24,
            output_chunk_length=1,  # 1 step ahead forecasting
            n_epochs=1,
            pl_trainer_kwargs={
                "enable_progress_bar": False,
                "accelerator": "auto",
            },
            **config,
        )

    def step(self):
        val_len = int(0.1 * len(train_series[0]))
        val = train_series[self.period][-val_len:]

        # Train the model
        pred = self.model.historical_forecasts(
            series=train_series[self.period],
            start=len(train_series[self.period]) - val_len,
            forecast_horizon=1,
            stride=1,
            retrain=True,
            verbose=False,
        )

        result = {"mae": mae(val, pred), "rmse": rmse(val, pred)}

        if result["rmse"] < self.best_rmse:
            result.update(should_checkpoint=True)
            self.best_rmse = result["rmse"]

        return result

    # https://github.com/ray-project/ray/issues/10290
    def save_checkpoint(self, checkpoint_dir):
        path = os.path.join(checkpoint_dir, "model.pt")
        f = open(path, "w")
        f.write("%s" % self.best_rmse)
        f.close()
        return path

    def load_checkpoint(self, path):
        self.model = NBEATSModel.load(path)


def start_analysis(model_name):
    train_fn_with_parameters = tune.with_parameters(
        TrainModel, model_name=model_name, period=1
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
        name=model_name,
        trial_name_creator=lambda trial: f"{model_name}_{trial.trial_id}",
        verbose=1,  # 0: silent, 1: only status updates, 2: status and trial results 3: most detailed
        # trial_dirname_creator=custom_trial_name,
        keep_checkpoints_num=1,
        # checkpoint_at_end=True,
        checkpoint_score_attr="rmse",
        checkpoint_freq=0,
        stop={"training_iteration": 1},
    )

    best = analysis.get_best_config(metric="rmse", mode="min")
    print(
        f"Best config: {best}\nHad a RMSE of {analysis.best_result['rmse']} and MAE of {analysis.best_result['mae']}"
    )

    # Gets best trial based on max accuracy across all training iterations.
    best_trial = analysis.get_best_trial(metric="rmse", mode="min", scope="all")
    print(best_trial)

    # Gets best checkpoint for trial based on accuracy.
    best_checkpoint = analysis.get_best_checkpoint(
        best_trial, metric="rmse", mode="min"
    )

    model_path = os.path.join(best_checkpoint, "model.pt")
    loaded = NBEATSModel.load(model_path)
    print(loaded)


if __name__ == "__main__":
    start_analysis("NBEATS")
    # path = "C:\\Users\\Stephan\\OneDrive\\GitHub\\Crypto_Forecasting\\ray_results\\NBEATS\\NBEATS_66fde9a5_1_batch_size=16,dropout=0.1365,num_blocks=2,num_stacks=32_2023-06-12_10-29-24\\NBEATSModel_2023-06-12_10_31_19.pt"
    # NBEATSModel.load(path)
