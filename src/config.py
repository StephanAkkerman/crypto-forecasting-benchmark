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

# Directories
coin_dir = "data/coins"

# Output directories
statistics_dir = "output/statistics"
plots_dir = "output/plots"

# Model directories
model_output_dir = "output/model_output"
log_returns_model_dir = f"{model_output_dir}/log_returns"
raw_model_dir = f"{model_output_dir}/raw"
extended_model_dir = f"{model_output_dir}/extended"
transformed_model_dir = f"{model_output_dir}/transformed"

rmse_dir = "output/rmse"
