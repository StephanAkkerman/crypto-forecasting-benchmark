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

# Perform hyperparameter optimization on the 0th (first) period
hyperopt_period = 0

# The number of parallel trials to run using Ray Tune
parallel_trials = 10

# The number of samples to draw from the hyperparameter space
num_samples = 20

# Results folder name
results_folder = "hyperopt_results"
