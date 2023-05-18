import timeit
import sys

# Local imports
from analysis import (
    auto_correlation,
    correlation,
    heteroskedasticity,
    seasonality,
    stationarity,
    stochasticity,
    trend,
    volatility,
)
from experiment import forecast

from experiment import hyper_opt2


def analysis():
    """
    Generates all analysis data and plots.
    """

    # Stationarity
    stationarity.stationarity_tests()

    # Autocorrelation
    auto_correlation.auto_cor_tests()

    # Trend
    trend.trend_tests()

    # Seasonality
    seasonality.seasonal_strength_test()

    # Heteroskedasticity
    heteroskedasticity.uncon_het_tests()
    heteroskedasticity.con_het_test()

    # Correlation
    correlation.correlation_tests()

    # Volatility
    volatility.volatility_tests()

    # Stochasticity
    stochasticity.calc_hurst()


def models():
    forecast.generate_forecasts("autoarima", "BTC", "1d", n_periods=9, show_plot=True)
    # 9 periods: 490 sec, 0.037 RMSE
    # 5 periods: 432 sec, 0.035 RMSE


if __name__ == "__main__":
    # elapsed_time = timeit.timeit(models, number=1)
    # print(f"Time elapsed: {elapsed_time:.2f} seconds")
    sys.path.append(
        "c:\\Users\\Stephan\\OneDrive\\GitHub\\Crypto_Forecasting\\src\\experiment"
    )
