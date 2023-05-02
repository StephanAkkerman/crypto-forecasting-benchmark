import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import ARIMA
from darts.metrics import mape, mase, rmse
import matplotlib.pyplot as plt

# Local imports
from vars import all_coins, timeframes
from csv_data import read_csv

def split_time_series_periods(ts, n_periods = 10, test_split = 0.25):
    """
    Splits a TimeSeries into train and test sets, with the test set being the last n_periods of the TimeSeries.
    """
    train_test_splits = []

    train_length = len(ts) * (1 - test_split) / n_periods
    test_length = len(ts) * test_split / n_periods

    for i in range(n_periods):
        # Train seperately or together?
        #train = ts[int(train_length * i):int(train_length * (i + 1))]
        #test = ts[int(train_length * (i + 1)):int(train_length * (i + 1)) + int(test_length)]

        # Train together
        train = ts[:int(train_length * (i + 1))]
        test = ts[int(train_length * (i + 1)):int(train_length * (i + 1)) + int(test_length)]

        train_test_splits.append((train, test))

    return train_test_splits

# Read data from a CSV file
data = read_csv("BTC", "1d", ["log returns"]).dropna()
data["date"] = data.index

# Create a Darts TimeSeries from the DataFrame
time_series = TimeSeries.from_dataframe(data, "date", "log returns")

# Split the data into train and test
n_periods = 10
train_test_splits = split_time_series_periods(time_series, n_periods)

mape_values = []
mase_values = []
rmse_values = []

model = ARIMA()
for train, test in train_test_splits:
    model.fit(train)

    predictions = model.predict(len(test))

    mape_value = mape(predictions, test)
    mase_value = mase(predictions, test, train)
    rmse_value = rmse(predictions, test)

    mape_values.append(mape_value)
    mase_values.append(mase_value)
    rmse_values.append(rmse_value)

results = pd.DataFrame({"MAPE":mape_values, "MASE":mase_values, "RMSE":rmse_values})
print(results)

# Plot the results
plt.figure(figsize=(15, 5))
plt.plot(predictions.univariate_values(), label="Predictions")
plt.plot(test.univariate_values(), label="Actual")
plt.legend()
plt.show()