# Advanced Algorithmic Trading App

Welcome to the **Advanced Algorithmic Trading App**, a comprehensive tool for training machine learning models, performing backtests, and predicting next-day stock movements. Built with Flask and styled using Bootstrap 5, this app offers a modern and intuitive interface for both novice and experienced traders.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Home Page](#home-page)
  - [Train Models](#train-models)
  - [Advanced Backtest (Single Ticker)](#advanced-backtest-single-ticker)
  - [Portfolio Backtest (Multiple Tickers)](#portfolio-backtest-multiple-tickers)
  - [Predict Next Day O/H/L/C](#predict-next-day-ohlc)
- [Data Preparation](#data-preparation)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Model Training**: Train classification models (Logistic Regression, Random Forest, MLP) and regression models for predicting next-day Open, High, Low, and Close prices.
- **Advanced Backtesting**: Perform single-ticker backtests with detailed metrics and visualizations.
- **Portfolio Backtesting**: Backtest multiple tickers simultaneously with an intuitive ticker selection interface.
- **Next-Day Predictions**: Predict next-day stock movements and receive actionable trading suggestions.
- **Real-Time Progress Tracking**: Monitor training progress using Server-Sent Events (SSE).
- **Modern UI**: Enjoy a sleek and responsive design powered by Bootstrap 5.

## Installation

### Prerequisites

- **Python 3.9** or higher
- **pip** (Python package installer)

### Clone the Repository

```bash
git clone https://github.com/hamzapatwa/StockMarketML.git
cd StockMarketML
```

### Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

*If you don't have a `requirements.txt`, you can install the necessary packages manually:*

```bash
pip install flask pandas numpy scikit-learn matplotlib
```

### Prepare Data

Ensure you have a `hist/` directory in the project root containing CSV files for each ticker. Each CSV should include columns like `Date`, `Open`, `High`, `Low`, `Close`, and `Volume`.

This repo will already have a folder full of information, but it may not be up to date.

The 'download-nasdaq-historical-data.ipynb' has instructions on how to download the history of your chosen symbols. By deafult, this repo comes with information up to 12/27/2024 for the SP500 companies as of that date.

Example structure:

```
hist/
├── AAPL.csv
├── MSFT.csv
├── GOOG.csv
└── ...
```

## Usage

### Run the Application

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000/` to access the app.

### Home Page

The home page provides links to all major functionalities:

- **Train Models**: Initiate model training and monitor progress.
- **Run Advanced Backtest (Single Ticker)**: Perform backtests on individual tickers.
- **Run Portfolio Backtest (Multiple Tickers)**: Backtest a portfolio of selected tickers.
- **Predict Next Day O/H/L/C**: Get predictions for next-day stock prices.

### Train Models

1. Click on **Train Models (SSE Progress)**.
2. Click the **Start Training** button to begin training models for all available tickers.
3. Monitor the training progress through the progress bar and status messages.
4. Once training is complete, models are saved for future use.

### Advanced Backtest (Single Ticker)

1. Click on **Run Advanced Backtest (Single Ticker)**.
2. Select a ticker from the dropdown menu.
3. Configure backtesting parameters such as model choice, initial capital, stop-loss percentage, etc.
4. Run the backtest to view results, including final capital, percent return, Sharpe ratio, and max drawdown.
5. Optionally, refine your backtest parameters and re-run without returning to the setup page.

### Portfolio Backtest (Multiple Tickers)

1. Click on **Run Portfolio Backtest (Multiple Tickers)**.
2. Use the search bar to filter tickers and add desired tickers to the "Selected Tickers" list.
3. Configure portfolio backtesting parameters similar to the single ticker backtest.
4. Run the backtest to view combined portfolio results.
5. Refine and re-run as needed directly from the results page.

### Predict Next Day O/H/L/C

1. Click on **Predict Next Day O/H/L/C**.
2. Select a ticker to predict.
3. Submit the form to view predicted Open, High, Low, and Close prices for the next day.
4. Receive a trading suggestion based on the predicted price movement.

## Data Preparation

Ensure each CSV file in the `hist/` directory follows the required format:

- **Columns**: `Date`, `Open`, `High`, `Low`, `Close`, `Volume`
- **Date Format**: ISO 8601 (e.g., `2023-01-01T09:30:00Z`)

The app processes this data to compute indicators and create labels for model training.

## Troubleshooting

### Common Issues

- **BuildError: Could not build url for endpoint 'predict_next_day'**:
  - **Cause**: Flask couldn't find the `predict_next_day` route when rendering templates.
  - **Solution**: Ensure the `predict_next_day` route is defined **before** the `index` route in `app.py`.

- **No Models Found for Ticker**:
  - **Cause**: Models haven't been trained for the selected ticker.
  - **Solution**: Navigate to the **Train Models** page and initiate training.

- **Data Loading Errors**:
  - **Cause**: Missing or improperly formatted CSV files in the `hist/` directory.
  - **Solution**: Verify that all required CSV files exist and follow the correct format.

### Tips

- **Check Logs**: Review the terminal output for detailed error messages.
- **Browser Console**: Open the browser's developer console to inspect any frontend errors.
- **Dependencies**: Ensure all Python packages are correctly installed in your virtual environment.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes. For major changes, please open an issue first to discuss your ideas.

## License

This project is licensed under the [MIT License](LICENSE).

---

*Happy Trading!*
