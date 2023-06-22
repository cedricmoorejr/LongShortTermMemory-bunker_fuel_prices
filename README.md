# Long Short Term Memory Bunker Fuel Price Prediction

 <!-- ![Project Logo](logo.png) Replace logo.png with your project's logo file name -->

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Description

The Long Short Term Memory (LSTM) Bunker Fuel Price Prediction project aims to forecast bunker fuel prices, specifically marine gas oil, using LSTM neural networks. This R script utilizes time series data, preprocessing techniques, and the Keras library for building and training the LSTM model. Please note that I will not be including the forecast plot. In addition, you will see in the code that I use the prefix of a package along with the function so that viewers are able to follow along. 

## Features

- Import and reformat dataset: Retrieve bunker fuel price data, preprocess, and transform it for modeling purposes.
- Rescale input data: Apply Z-score scaling to the dataset to enhance convergence during the LSTM training process.
- Create lagged variables: Generate lagged variables from the time series data to form input-output pairs for LSTM training.
- Build LSTM model: Define the architecture of the LSTM model using the Keras package, incorporating dropout layers for regularization.
- Train the model: Compile and train the LSTM model on the provided dataset, using early stopping for improved efficiency.
- Forecast model: Use the trained LSTM model to make predictions and extract forecasted bunker fuel prices.

## Installation

1. Clone the repository: `git clone https://github.com/cedricmoorejr/LongShortTermMemory-bunker_fuel_prices.git`
2. Install the required R packages: `tidyverse`, `keras`, `magrittr`, `caret`, `reticulate`, `RSocrata`, `tensorflow`.
3. Set up the Python environment and ensure the availability of `keras` and `tensorflow` libraries.
4. Provide API keys for accessing the bunker fuel price dataset.
5. Customize the script and dataset as per your requirements.

## Usage

- Modify the script parameters, such as the prediction horizon (`p`), number of lagged variables (`n`), and Python version (`python_version`).
- Run the script to import, preprocess, and format the dataset.
- Train the LSTM model using the prepared data and monitor the training progress.
- Utilize the trained model to make predictions and extract forecasted bunker fuel prices.

For detailed instructions and code explanations, please refer to the comments within the script itself.

## License

This project is licensed under the [MIT License](LICENSE.txt).

## Contact

For any questions or inquiries, please contact [Cedric Moore Jr.](mailto:cedricmoorejunior@outlook.com).
