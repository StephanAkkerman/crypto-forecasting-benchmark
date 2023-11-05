# Benchmarking Cryptocurrency Forecasting Models in the Context of Data Properties and Market Factors
[![Python 3.9.16](https://img.shields.io/badge/python-3.9.16-blue.svg)](https://www.python.org/downloads/release/python-3916/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![MIT License](https://img.shields.io/github/license/StephanAkkerman/Crypto_Forecasting.svg?color=brightgreen)](https://opensource.org/licenses/MIT)

This repository contains the code used in our research on the influence of volatility on the predictive performance of time series forecasting models. We have implemented a variety of models including ARIMA, XGBoost, N-BEATS, and Prophet, among others, using libraries such as Darts, StatsForecast, and fbprophet.

The code includes functionalities for data preparation, model training, hyperparameter optimization, and performance evaluation. We have used Ray Tune for hyperparameter optimization and have implemented custom strategies to handle GPU memory management.

The research focuses on financial time series data, specifically cryptocurrency price data, and investigates how different periods of market factors affect the performance of forecasting models. The results of this research can provide valuable insights for financial forecasting, risk management, and trading strategy development.

Please refer to the individual scripts for more detailed information about the specific procedures and methodologies used. Contributions and suggestions for improvements are welcome.

## Supported models
The following univariate models are supported.

- ARIMA
- Random Forest
- XGBoost
- LightGBM
- Prophet
- TBATS
- N-BEATS
- RNN
- LSTM
- GRU
- TCN
- TFT
- NHiTS

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

This project requires Python 3.9.16. You can download it [here](https://www.python.org/downloads/release/python-3916/). 

Additionally, you will need to install several Python packages which are listed in `requirements.txt`. You can install these packages using pip by running the following command in your terminal:

```
pip install -r requirements.txt
```
If you want to use the development version:
```
pip install git+https://github.com/StephanAkkerman/Crypto_Forecasting.git
```

> **Note**
> While using a GPU is optional, it is highly recommended for performing hyperparameter optimization and fitting and evaluating the models. You can do that by simply running the following line of code in your terminal, this will install the latest version of PyTorch with CUDA 12.1 support. 
> ``` 
>pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
> ```

> **Warning**
> You might need to change the `batch_size` and `parallel_trials` parameters to prevent GPU OOM errors. You can find these parameters in the `search_space.py` file located in the `src/hyperopt` directory.

## Usage

The main script of this project is `main.py` located in the `src` directory. You can adjust this script to perform data analysis, evaluate models, etc. To perform hyperparameter optimization using Ray Tune use the `hyperopt_ray.py` file located in the `src/hyperopt` directory.

## Examples 

TODO

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
