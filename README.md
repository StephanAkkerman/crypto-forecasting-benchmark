# Benchmarking Cryptocurrency Forecasting Models in the Context of Data Properties and Market Factors
[![Python 3.9.16](https://img.shields.io/badge/python-3.9.16-blue.svg)](https://www.python.org/downloads/release/python-3916/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![MIT License](https://img.shields.io/github/license/StephanAkkerman/Crypto_Forecasting.svg?color=brightgreen)](https://opensource.org/licenses/MIT)

This repository contains the code used in our research on the influence of volatility on the predictive performance of time series forecasting models. We have implemented a variety of models including ARIMA, XGBoost, N-BEATS, and Prophet, among others, using libraries such as Darts, StatsForecast, and fbprophet.

The code includes functionalities for data preparation, model training, hyperparameter optimization, and performance evaluation. We have used Ray Tune for hyperparameter optimization and have implemented custom strategies to handle GPU memory management.

The research focuses on financial time series data, specifically cryptocurrency price data, and investigates how different periods of market factors affect the performance of forecasting models. The results of this research can provide valuable insights for financial forecasting, risk management, and trading strategy development.

Please refer to the individual scripts for more detailed information about the specific procedures and methodologies used. Contributions and suggestions for improvements are welcome.

## Features & Usage Guide

### Configuration & Data Retrieval
- **Setup Configurations:** Begin by specifying your preferred cryptocurrency symbols and the time frames for analysis in `src/config.py`. This sets up your environment for the data you're interested in.
- **Download Market Data:** Leverage `create_all_data()` in `src/data/create_data.py` to fetch data directly from the Binance API. Our script enriches your dataset with essential financial metrics such as logarithmic returns and volatility measures, outputting them in neatly organized .csv files for easy use.

### Data Analysis & Testing
- **Time Series Analysis:** Employ `data_analysis_tests()` from `src/analysis.py` to conduct thorough testing on your time series datasets. We've incorporated checks for key properties including stationarity, autocorrelation, trends, seasonality, heteroskedasticity, and random walks to ensure comprehensive understanding of data behavior.
- **Hyperparameter Optimization:** Tailor the forecasting models to your needs. Fill out the parameters in `src/hyperopt/config.py` and execute `src/hyperopt/hyperopt_ray.py` to determine the optimal settings for achieving the best prediction performance.

### Forecasting & Results Evaluation
- **Run Forecasts:** With the fine-tuned hyperparameters, you can run `forecast_models()` in `src/analysis.py` to start forecasting the future of your chosen cryptocurrencies.
- **Analyze Forecasts:** To evaluate how well your forecasts are performing, `forecast_analysis()` is your go-to function in the same `src/analysis.py` script. It will help you visualize and understand the predictive capabilities of your models.
- **Examine Data Impact:** To assess how various data properties may be influencing your forecasts, `forecast_statistical_tests()` in `src/analysis.py` will run diagnostic checks.
- **Market Factors Analysis:** Understand the influence of external market factors with `market_factors_impact()` also found in `src/analysis.py`, giving you insights into how different variables affect cryptocurrency prices.

### Supported Forecasting Models
Our toolkit supports an extensive range of univariate forecasting models to cater to a variety of data patterns and prediction needs:

- **Traditional Models:**
  - ARIMA: Autoregressive Integrated Moving Average, for capturing linear trends and seasonality.
- **Machine Learning Models:**
  - Random Forest: A robust ensemble of decision trees for non-linear trend capture.
  - XGBoost: eXtreme Gradient Boosting for efficient and powerful predictive performance.
  - LightGBM: Light Gradient Boosting Machine, renowned for its speed and accuracy.
  - Prophet: Designed for forecasting with daily observations that display patterns on different time scales.
- **Advanced Time Series Models:**
  - TBATS: Incorporating multiple seasonalities, Box-Cox transformation, ARMA errors, Trend and Seasonal components.
  - N-BEATS: Neural Basis Expansion Analysis for interpretable time series forecasting.
- **Deep Learning Models:**
  - RNN: Recurrent Neural Network for capturing temporal dynamic behavior.
  - LSTM: Long Short-Term Memory networks, ideal for making predictions based on long-term sequential patterns.
  - GRU: Gated Recurrent Units, for modeling sequences with fewer parameters than LSTM.
  - TCN: Temporal Convolutional Network, a convolution-based architecture designed to handle sequence modeling tasks.
  - TFT: Temporal Fusion Transformers, for high-performance interpretable forecasting.
  - NHiTS: A recent deep learning approach that exploits hierarchical time series forecasting without the need for pre-defined hierarchies.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

This project requires Python 3.9.16. You can download it [here](https://www.python.org/downloads/release/python-3916/). 

Additionally, you will need to install several Python packages which are listed in `requirements.txt`. You can install these packages using pip by running the following command in your terminal:

```
pip install -r requirements.txt
```
Or
```
pip install git+https://github.com/StephanAkkerman/Crypto_Forecasting.git
```

> [!NOTE]
> While using a GPU is optional, it is highly recommended for performing hyperparameter optimization and fitting and evaluating the models. You can do that by simply running the following line of code in your terminal, this will install the latest version of PyTorch with CUDA 12.1 support. 
> ``` 
>pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
> ```

> [!WARNING]
> You might need to change the `batch_size` and `parallel_trials` parameters to prevent GPU OOM errors. You can find these parameters in the `search_space.py` file located in the `src/hyperopt` directory.

## Contributing

Contributions are welcome! If you have a feature request, bug report, or proposal for code refactoring, please feel free to open an issue on GitHub. I appreciate your help in improving this project.

## Citation

If you use this project in your research, please cite this repository and the associated master's thesis. The BibTeX entry for the thesis is:

```bibtex
@mastersthesis{Akkerman2023,
  author  = {Stephan Akkerman},
  title   = {Benchmarking Cryptocurrency Forecasting Models in the Context of Data Properties and Market Factors},
  school  = {Utrecht University},
  year    = {2023},
  address = {Utrecht, The Netherlands},
  month   = {October},
  note    = {Available at: \url{https://github.com/StephanAkkerman/crypto-forecasting-benchmark}}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
