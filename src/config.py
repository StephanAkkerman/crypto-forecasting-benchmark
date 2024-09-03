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

market_cap_dict = {
    "BTC": 586_582_359_795,
    "ETH": 251_325_536_962,
    "BNB": 51_966_811_410,
    "XRP": 26_933_378_690,
    "ADA": 15_763_932_024,
    "DOGE": 12_363_343_259,
    "MATIC": 10_813_456_408,
    # Mid
    "LTC": 7_043_573_473,
    "TRX": 5_993_703_041,
    "LINK": 4_143_670_480,
    "ETC": 3_107_711_056,
    "XLM": 2_852_025_305,
    "ATOM": 3_549_441_677,
    "XMR": 2_968_462_104,
    # Small
    "VET": 1_866_409_658,
    "ALGO": 1_673_942_369,
    "EOS": 1_353_268_876,
    "XTZ": 1_083_367_662,
    "CHZ": 925_733_735,
    "NEO": 915_645_133,
    "IOTA": 627_241_565,
}

time_frames = timeframes = ["1m", "15m", "4h", "1d"]
tf_names = [
    "One-Minute Time Frame",
    "Fifteen-Minute Time Frame",
    "Four-Hour Time Frame",
    "Daily Time Frame",
]
tf_names2 = [
    "One-Minute",
    "Fifteen-Minute",
    "Four-Hour",
    "Daily",
]

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
model_output_dir = "output/forecasts"

# Prediction output names
log_returns_pred = "log_returns"
raw_pred = "raw"
scaled_pred = "scaled"
extended_pred = "extended"

# Transformed predictions
log_to_raw_pred = "log_to_raw"
raw_to_log_pred = "raw_to_log"
scaled_to_log_pred = "scaled_to_log"
scaled_to_raw_pred = "scaled_to_raw"
extended_to_raw_pred = "extended_to_raw"

# Stress test prediction
log_returns_stress_pred = "log_returns_stress"
raw_stress_pred = "raw_stress"
scaled_stress_pred = "scaled_stress"
scaled_to_log_stress_pred = "scaled_to_log_stress"
raw_to_log_stress_pred = "raw_to_log_stress"

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
