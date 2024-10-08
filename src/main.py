import os

import analysis

# Local Import
import config
from data.binance_data import fetchData
from experiment import baseline, rmse

if __name__ == "__main__":
    # Start by testing if the data is available
    for coin in config.all_coins:
        if os.path.exists(f"{config.coin_dir}/{coin}"):
            # Test if the .csv files exist in the folder
            for time_frame in config.time_frames:
                if not os.path.exists(
                    f"{config.coin_dir}/{coin}/{coin}USDT_{time_frame}.csv"
                ):
                    fetchData(symbol=coin, timeframe=time_frame, as_csv=True)
        else:
            fetchData(symbol=coin, timeframe=time_frame, as_csv=True)

    # Perfrom the data analyses
    analysis.data_analysis_tests()

    # Perform hyperparameter optimization
    # Run: python src/hyperopt/hyperopt_ray.py

    # Start the forecasting
    analysis.forecast_models()
    rmse.build_complete_rmse_database()

    # Optional: extended forecasts
    # analysis.forecast_models_extended()

    # Optional: stress test
    # analysis.stress_test_all()

    # Analysis of the results
    analysis.forecast_analysis()
    analysis.forecast_statistical_tests()
    analysis.market_factors_impact(group_tf=False)

    # # Compare to baseline
    baseline.create_all_baseline_comparison()
