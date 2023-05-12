from tqdm import tqdm

# Models
from darts.models import ARIMA
from darts.models.forecasting.auto_arima import AutoARIMA

# Local imports
from models.train_test import get_train_test
from models.eval import plot_results


def one_step_forecast(model, train, test) -> list:
    forecast = []

    # Loop over each period in the test set
    for t in tqdm(range(len(test))):
        model.fit(train)
        # Only use one, for one-step-ahead forecasting
        prediction = model.predict(n=1)
        forecast.append(prediction.first_value())
        # Add the current test value to the train set for the next loop
        train = train.append(test[t])

    return forecast


def generate_forecasts(
    model_name: str, coin: str, time_frame: str, n_periods=9, show_plot=True
):
    if model_name.lower() == "arima":
        model = ARIMA()
    elif model_name.lower() == "autoarima":
        # https://unit8co.github.io/darts/generated_api/darts.models.forecasting.auto_arima.html
        model = AutoARIMA()

    # Get the training and testing data for each period
    trains, tests = get_train_test(
        coin=coin, time_frame=time_frame, n_periods=n_periods
    )
    predictions = []

    for i, (train, test) in enumerate(zip(trains, tests)):
        print(f"Training on period {i + 1}...")
        predictions.append(one_step_forecast(model, train, test))

    plot_results(coin, time_frame, trains, tests, predictions, show_plot)
