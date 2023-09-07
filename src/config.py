all_models = [
    "RandomForest",
    "XGB",
    "LightGBM",
    "Prophet",
    "NBEATS",
    "RNN",
    "LSTM",
    "GRU",
    "TCN",
    "TFT",
    "NHiTS",
    "ARIMA",
    "TBATS",
]
ml_models = ["NBEATS", "RNN", "LSTM", "GRU", "TCN", "TFT", "NHiTS"]

large_cap = ["BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "MATIC"]
mid_cap = ["LINK", "ETC", "XLM", "LTC", "TRX", "ATOM", "XMR"]
small_cap = ["VET", "ALGO", "EOS", "CHZ", "IOTA", "NEO", "XTZ"]
all_coins = large_cap + mid_cap + small_cap

timeframes = ["1m", "15m", "4h", "1d"]

# If GPU is available use it
use_GPU = True

# If -1 use all CPU cores, otherwise specify the number of cores to use
cpu_cores = -1

# Use 25% of the data for testing, the rest for training
test_percentage = 0.25

# Use 10% of the training data for validation
val_percentage = 0.1

# Split the data in periods of 5
n_periods = 5

# Data directories
coin_dir = "data/coins"

# Output directories
statistics_dir = "output/statistics"
plots_dir = "output/plots"
rmse_dir = "output/rmse"
volatility_dir = "output/volatility"
comparison_dir = "output/comparison"

# Model directories
model_output_dir = "output/model_output"

# Prediction output names
log_returns_pred = "log_returns"
raw_pred = "raw"
extended_pred = "extended"
scaled_pred = "scaled"

# Transformed predictions
log_to_raw_pred = "log_to_raw"
raw_to_log_pred = "raw_to_log"
scaled_to_log_pred = "scaled_to_log"
scaled_to_raw_pred = "scaled_to_raw"
extended_to_raw_pred = "extended_to_raw"

# Stress test output
stress_test_dir = "output/stress_test"

# Fancy model names dict
pred_names = {
    log_returns_pred: "Log Returns",
    log_to_raw_pred: "Log Returns",
    raw_pred: "Raw Price",
    raw_to_log_pred: "Raw Price",
    extended_pred: "Extended",
    extended_to_raw_pred: "Extended",
    scaled_pred: "Scaled",
    scaled_to_log_pred: "Scaled",
    scaled_to_raw_pred: "Scaled",
}

log_preds = [
    log_returns_pred,
    raw_to_log_pred,
    scaled_to_log_pred,
]

raw_preds = [log_to_raw_pred, raw_pred, scaled_to_raw_pred]

all_preds = pred_names.keys()
