import config
from experiment import forecast

if __name__ == "__main__":
    # Run this on laptop
    forecast.stress_test_model(
        model=config.log_returns_model, forecasting_model="Prophet"
    )
    forecast.stress_test_model(model=config.scaled_model, forecasting_model="Prophet")
    forecast.stress_test_model(model=config.raw_model, forecasting_model="Prophet")
