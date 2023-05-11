from darts.models import ARIMA
from tqdm import tqdm

# Local imports
from models.train_test import get_train_test
from models.eval import plot_results


def forecast(train, test) -> list:
    model = ARIMA()
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


def arima():
    # Get the training and testing data for each period
    trains, tests = get_train_test(n_periods=5)
    predictions = []

    for i, (train, test) in enumerate(zip(trains, tests)):
        print(f"Training on period {i + 1}...")
        predictions.append(forecast(train, test))

    plot_results(trains, tests, predictions)
