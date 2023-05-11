import numpy as np
from pmdarima import auto_arima
from darts.models import ARIMA
from darts.metrics import mape, mase, rmse

from models.train_test import get_train_test
from models.eval import plot_results


# Function to perform one-step-ahead forecasting using ARIMA
def arima_forecast(train, test):
    # model = auto_arima(train, suppress_warnings=True, stepwise=True)
    model = ARIMA()
    forecast = []

    for t in range(len(test)):
        model.fit(train)
        prediction = model.predict(n=1)
        forecast.append(prediction.first_value())
        train = train.append(test[t])

    return forecast


def arima():
    # Perform one-step-ahead forecasting for each period
    predictions = []
    data, trains, tests = get_train_test()
    counter = 1
    for train_data, test_data in zip(trains, tests):
        print("Training on period", counter)
        period_forecast = arima_forecast(train_data, test_data)
        predictions.append(period_forecast)
        counter += 1

    plot_results(data, trains, tests, predictions)
