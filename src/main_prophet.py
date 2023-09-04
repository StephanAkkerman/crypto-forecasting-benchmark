import config
from experiment import forecast

if __name__ == "__main__":
    # Run this on laptop
    # forecast.stress_test_model(model=config.log_returns_model,forecasting_model="Prophet")
    # forecast.stress_test_model(model=config.scaled_model, forecasting_model="Prophet")
    forecast.stress_test_all(
        model=config.raw_model,
        ignore_model=["Prophet", "TBATS"],
        start_from_model="NBEATS",
        start_from_coin="XRP",
        start_from_time_frame="1d",
    )
