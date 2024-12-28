import os
import glob
import pickle
import numpy as np
import pandas as pd
import time
import threading

from flask import Flask, request, render_template_string, redirect, url_for, Response, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use('Agg')  # For servers without GUI
import matplotlib.pyplot as plt
import io
import base64

###############################################################################
# Flask Initialization
###############################################################################
app = Flask(__name__)

###############################################################################
# GLOBAL STATE
###############################################################################
MODELS = {}
SCALERS = {}

# For SSE-based training progress
progress_data = {
    'current': 0,
    'total': 0,
    'status': 'idle'
}
progress_lock = threading.Lock()


###############################################################################
# 1. HELPER FUNCTIONS (DATA LOADING, INDICATORS, LABELING)
###############################################################################
def list_tickers(hist_folder='hist'):
    """Return all CSV filenames (minus .csv extension) in the hist folder."""
    csv_files = glob.glob(os.path.join(hist_folder, '*.csv'))
    return sorted([os.path.splitext(os.path.basename(f))[0] for f in csv_files])


def load_data_for_ticker(ticker, hist_folder='hist'):
    """Load CSV for a given ticker, parse dates, and set as index."""
    file_path = os.path.join(hist_folder, f"{ticker}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No CSV file found for ticker {ticker} at {file_path}.")

    # Adjust skiprows if needed based on actual CSV structure
    df = pd.read_csv(file_path, skiprows=[1, 2])
    df.rename(columns={
        "Price": "Date",  # Some CSVs might have "Price" as date
        "Datetime": "Date",  # If "Datetime" is present, rename to "Date"
        "Close": "Close",
        "High": "High",
        "Low": "Low",
        "Open": "Open",
        "Volume": "Volume"
    }, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df.set_index('Date', inplace=True, drop=True)
    return df


def compute_indicators(df):
    """Compute common and extra technical indicators on the DataFrame."""
    df = df.copy()
    df.sort_index(inplace=True)

    # Moving Averages
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()

    # Bollinger Bands (using 20-day window)
    sma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    df['Boll_Upper'] = sma_20 + 2 * std_20
    df['Boll_Lower'] = sma_20 - 2 * std_20

    # Daily Returns
    df['Daily_Return'] = df['Close'].pct_change()

    # RSI (14-day)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = (-1) * delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # On-Balance Volume (OBV)
    df['OBV'] = 0
    df['OBV'] = np.where(df['Close'] > df['Close'].shift(1),
                         df['Volume'],
                         np.where(df['Close'] < df['Close'].shift(1),
                                  -df['Volume'], 0)).cumsum()

    # Stochastic Oscillator (14-day)
    window = 14
    low14 = df['Low'].rolling(window=window).min()
    high14 = df['High'].rolling(window=window).max()
    df['Stoch_%K'] = 100 * (df['Close'] - low14) / (high14 - low14 + 1e-9)
    df['Stoch_%D'] = df['Stoch_%K'].rolling(3).mean()

    df.dropna(inplace=True)
    return df


def ensemble_predict(rf_model, mlp_model, X_in):
    p1 = rf_model.predict(X_in)
    p2 = mlp_model.predict(X_in)
    return (p1 + p2) / 2.0


def create_labels(df, threshold=0.0025):
    """
    Create multi-class labels for classification:
      2 => BUY, 0 => SELL, 1 => HOLD
      threshold=0.0025 => +/-0.25% for buy/sell triggers
    """
    df = df.copy()
    df['Next_Close'] = df['Close'].shift(-1)
    df.dropna(inplace=True)

    df['Pct_Change'] = (df['Next_Close'] - df['Close']) / df['Close']
    conditions = [
        (df['Pct_Change'] >= threshold),
        (df['Pct_Change'] <= -threshold)
    ]
    choices = [2, 0]  # 2=BUY, 0=SELL
    df['Action'] = np.select(conditions, choices, default=1)  # 1=HOLD

    class_counts = df['Action'].value_counts()
    print(f"Class distribution for threshold {threshold}:\n{class_counts}")
    return df


###############################################################################
# 2. MODEL TRAINING / SAVING / LOADING
###############################################################################
def train_models_for_ticker(ticker, df):
    """
    Train classification models (buy/sell/hold) and an ensemble regression
    (next-day O/H/L/C) with:
      - RandomForestRegressor
      - MLPRegressor
    Then we average their predictions for final 'NextOpenReg', 'NextHighReg', etc.

    Also clips extreme target values (±20% from today's Close).
    """
    # Feature columns: we've added Bollinger, OBV, Stoch, etc.
    feature_cols = [
        'Close', 'High', 'Low', 'Open', 'Volume',
        'MA_10', 'MA_50', 'MA_200',
        'Boll_Upper', 'Boll_Lower',
        'Daily_Return', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'OBV', 'Stoch_%K', 'Stoch_%D'
    ]

    if len(df) < 50:  # Arbitrary small check
        print(f"Skipping {ticker}: not enough data after extra indicators.")
        return None, None

    X_class = df[feature_cols].values
    y_class = df['Action'].values

    unique_classes = np.unique(y_class)
    if len(unique_classes) < 2:
        print(f"Skipping ticker {ticker}: only one class present ({unique_classes[0]})")
        return None, None

    # Train/test split
    X_train_c, X_test_c, y_train_c, _ = train_test_split(
        X_class, y_class, test_size=0.2, shuffle=False
    )

    # Normalize/scale
    scaler = StandardScaler()
    X_train_c_scaled = scaler.fit_transform(X_train_c)

    # --------------------------
    # 1) CLASSIFICATION MODELS
    # --------------------------
    lr = LogisticRegression(multi_class='multinomial', max_iter=1000)
    lr.fit(X_train_c_scaled, y_train_c)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train_c_scaled, y_train_c)

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000)
    mlp.fit(X_train_c_scaled, y_train_c)

    classification_models = {
        'LogisticRegression': lr,
        'RandomForest': rf,
        'MLP': mlp
    }

    # --------------------------
    # 2) REGRESSION MODELS
    # --------------------------
    df_reg = df.copy()
    df_reg['Next_Open'] = df_reg['Open'].shift(-1)
    df_reg['Next_High'] = df_reg['High'].shift(-1)
    df_reg['Next_Low'] = df_reg['Low'].shift(-1)
    df_reg['Next_Close'] = df_reg['Close'].shift(-1)
    df_reg.dropna(inplace=True)

    # Clip extreme next-day targets to ±20% from today's close
    MAX_SHIFT_PCT = 0.2  # 20%
    for col in ['Next_Open', 'Next_High', 'Next_Low', 'Next_Close']:
        diff = df_reg[col] - df_reg['Close']
        pct = diff / (df_reg['Close'] + 1e-9)
        pct_clipped = pct.clip(lower=-MAX_SHIFT_PCT, upper=MAX_SHIFT_PCT)
        df_reg[col] = df_reg['Close'] + pct_clipped * df_reg['Close']

    X_reg = df_reg[feature_cols].values
    y_open = df_reg['Next_Open'].values
    y_high = df_reg['Next_High'].values
    y_low = df_reg['Next_Low'].values
    y_close = df_reg['Next_Close'].values

    # Same # of rows for classification/regression (minus shift).
    X_train_r = X_reg[:len(X_train_c)]
    X_train_r_scaled = scaler.transform(X_train_r)

    # We'll do a simple 2-model ensemble for each target:
    # RandomForestRegressor + MLPRegressor -> average
    rf_open = RandomForestRegressor(n_estimators=50)
    mlp_open = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000)

    rf_high = RandomForestRegressor(n_estimators=50)
    mlp_high = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000)

    rf_low = RandomForestRegressor(n_estimators=50)
    mlp_low = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000)

    rf_close = RandomForestRegressor(n_estimators=50)
    mlp_close = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000)

    # Fit each regressor
    rf_open.fit(X_train_r_scaled, y_open[:len(X_train_r)])
    rf_high.fit(X_train_r_scaled, y_high[:len(X_train_r)])
    rf_low.fit(X_train_r_scaled, y_low[:len(X_train_r)])
    rf_close.fit(X_train_r_scaled, y_close[:len(X_train_r)])

    mlp_open.fit(X_train_r_scaled, y_open[:len(X_train_r)])
    mlp_high.fit(X_train_r_scaled, y_high[:len(X_train_r)])
    mlp_low.fit(X_train_r_scaled, y_low[:len(X_train_r)])
    mlp_close.fit(X_train_r_scaled, y_close[:len(X_train_r)])

    # Store them individually for optional debugging
    # but we also provide ensemble predictions as final
    # We'll just store closures that do an ensemble predict
    regression_models = {
        'NextOpenReg': (rf_open, mlp_open, ensemble_predict),
        'NextHighReg': (rf_high, mlp_high, ensemble_predict),
        'NextLowReg': (rf_low, mlp_low, ensemble_predict),
        'NextCloseReg': (rf_close, mlp_close, ensemble_predict)
    }

    all_models = {**classification_models, **regression_models}
    return all_models, scaler


def train_all_tickers_with_progress():
    """
    Train models for each CSV found in `hist/`, show progress via SSE.
    """
    global MODELS, SCALERS, progress_data

    tickers = list_tickers()
    with progress_lock:
        progress_data['current'] = 0
        progress_data['total'] = len(tickers)
        progress_data['status'] = 'training'

    for ticker in tickers:
        try:
            df = load_data_for_ticker(ticker)
            df = compute_indicators(df)
            df = create_labels(df)

            models_dict, scaler = train_models_for_ticker(ticker, df)
            if models_dict and scaler:
                MODELS[ticker] = models_dict
                SCALERS[ticker] = scaler
                print(f"Trained models for ticker: {ticker}")
            else:
                print(f"Skipped training for ticker: {ticker} (insufficient data or classes).")

        except Exception as e:
            print(f"Error training ticker {ticker}: {e}")

        with progress_lock:
            progress_data['current'] += 1
        time.sleep(0.5)  # purely to visualize progress

    save_models()
    with progress_lock:
        progress_data['status'] = 'done'


def save_models(filename='models.pkl'):
    """Pickle the MODELS and SCALERS dictionaries."""
    with open(filename, 'wb') as f:
        pickle.dump((MODELS, SCALERS), f)


def load_models(filename='models.pkl'):
    """Load the pickled MODELS and SCALERS dictionaries if they exist."""
    global MODELS, SCALERS
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            MODELS, SCALERS = pickle.load(f)


###############################################################################
# 3. BACKTESTING FUNCTIONS (NOW TRACKS TRADE STATS)
###############################################################################
def advanced_backtest(ticker, model_name,
                      initial_capital=10000,
                      stop_loss_percent=0.05,
                      partial_sell_ratio=0.5,
                      prob_threshold=0.6,
                      trailing_stop=True,
                      take_profit_percent=0.2):
    """
    Naive minute-based backtest for a single ticker.
    Returns:
      final_val,
      final_ret_str,
      daily_dates,
      daily_values,
      metrics (incl. #trades, wins, losses, maintains, win rate).
    """

    if ticker not in MODELS or ticker not in SCALERS:
        return None, "No models found for this ticker.", None, None, {}

    if model_name not in MODELS[ticker]:
        return None, f"Model {model_name} not found for {ticker}.", None, None, {}

    df = load_data_for_ticker(ticker)
    df = compute_indicators(df)
    df = create_labels(df)

    classifier = MODELS[ticker][model_name]
    # This must match the updated feature_cols in train_models_for_ticker
    feature_cols = [
        'Close', 'High', 'Low', 'Open', 'Volume',
        'MA_10', 'MA_50', 'MA_200',
        'Boll_Upper', 'Boll_Lower',
        'Daily_Return', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'OBV', 'Stoch_%K', 'Stoch_%D'
    ]
    if any(col not in df.columns for col in feature_cols):
        return None, "Data missing some indicators for this ticker.", None, None, {}

    X = df[feature_cols].values
    X_scaled = SCALERS[ticker].transform(X)

    # Classification predictions
    has_proba = hasattr(classifier, "predict_proba")
    if has_proba:
        probas = classifier.predict_proba(X_scaled)
        predicted_actions = classifier.predict(X_scaled)
    else:
        probas = None
        predicted_actions = classifier.predict(X_scaled)

    df['Prediction'] = np.roll(predicted_actions, 1, axis=0)
    shifted_probas = np.roll(probas, 1, axis=0) if probas is not None else None

    # Tracking
    positions = []
    capital = float(initial_capital)
    daily_values = []
    daily_dates = []

    # For trade stats
    total_trades = 0
    wins = 0
    losses = 0
    maintains = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        current_price = row['Close']
        daily_portfolio_value = capital + sum(pos['shares'] * current_price for pos in positions)
        daily_values.append(daily_portfolio_value)
        daily_dates.append(idx)

        # Update trailing stops
        for pos in positions:
            if trailing_stop:
                if current_price > pos['highest_price']:
                    pos['highest_price'] = current_price
                new_sl = pos['highest_price'] * (1 - stop_loss_percent)
                if new_sl > pos['stop_loss_price']:
                    pos['stop_loss_price'] = new_sl

        action = row['Prediction']
        buy_conf = shifted_probas[i, 2] if (shifted_probas is not None and i < len(shifted_probas)) else 0.0
        sell_conf = shifted_probas[i, 0] if (shifted_probas is not None and i < len(shifted_probas)) else 0.0

        # Manage Positions
        updated_positions = []
        for pos in positions:
            was_sold = False
            sell_price_for_this_pos = current_price

            # If triggered stop-loss
            if current_price <= pos['stop_loss_price']:
                shares_sold = pos['shares']
                cost_basis = pos['entry_price']
                capital += shares_sold * current_price
                total_trades += 1

                # Evaluate trade result
                pct_change = (current_price - cost_basis) / cost_basis
                if pct_change > 0.005:
                    wins += 1
                elif pct_change < -0.005:
                    losses += 1
                else:
                    maintains += 1

                was_sold = True

            # If triggered take-profit
            elif (take_profit_percent > 0.0 and
                  current_price >= pos['entry_price'] * (1 + take_profit_percent)):
                shares_to_sell = int(pos['shares'] * partial_sell_ratio)
                if shares_to_sell > 0:
                    # Partial or full?
                    ratio = shares_to_sell / pos['shares']
                    partial_cost_basis = pos['entry_price']  # same basis
                    capital += shares_to_sell * current_price
                    pos['shares'] -= shares_to_sell
                    total_trades += 1

                    pct_change = (current_price - partial_cost_basis) / partial_cost_basis
                    if pct_change > 0.005:
                        wins += 1
                    elif pct_change < -0.005:
                        losses += 1
                    else:
                        maintains += 1

                if pos['shares'] > 0:
                    updated_positions.append(pos)
                was_sold = True  # at least partial

            # else keep the position
            else:
                updated_positions.append(pos)

            # If sold everything, we skip re-adding
            # If partial, we re-add the position above

        positions = updated_positions

        # Action = BUY
        if action == 2 and buy_conf >= prob_threshold:
            funds_to_spend = capital * buy_conf
            shares_to_buy = int(funds_to_spend // current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                capital -= cost
                positions.append({
                    'shares': shares_to_buy,
                    'entry_price': current_price,
                    'highest_price': current_price,
                    'stop_loss_price': current_price * (1 - stop_loss_percent)
                })

        # Action = SELL
        if action == 0 and sell_conf >= prob_threshold:
            for pos in positions:
                shares_to_sell = int(pos['shares'] * partial_sell_ratio)
                if shares_to_sell > 0:
                    ratio = shares_to_sell / pos['shares']
                    cost_basis = pos['entry_price']
                    capital += shares_to_sell * current_price
                    pos['shares'] -= shares_to_sell
                    total_trades += 1

                    pct_change = (current_price - cost_basis) / cost_basis
                    if pct_change > 0.005:
                        wins += 1
                    elif pct_change < -0.005:
                        losses += 1
                    else:
                        maintains += 1

            positions = [p for p in positions if p['shares'] > 0]

    # Final liquidation
    if positions:
        last_price = df.iloc[-1]['Close']
        for pos in positions:
            shares_sold = pos['shares']
            cost_basis = pos['entry_price']
            capital += shares_sold * last_price
            total_trades += 1

            pct_change = (last_price - cost_basis) / cost_basis
            if pct_change > 0.005:
                wins += 1
            elif pct_change < -0.005:
                losses += 1
            else:
                maintains += 1

    final_val = capital
    final_ret = (final_val - initial_capital) / initial_capital * 100.0
    final_ret_str = f"{final_ret:.2f}%"

    if not daily_values:
        return None, "No daily values computed.", None, None, {}

    # Calculate Sharpe, MDD
    daily_returns = []
    for i in range(1, len(daily_values)):
        ret = (daily_values[i] - daily_values[i - 1]) / (daily_values[i - 1] + 1e-9)
        daily_returns.append(ret)

    if len(daily_returns) > 1:
        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns, ddof=1)
        sharpe = (mean_ret / (std_ret + 1e-9)) * np.sqrt(252 * 390)
    else:
        sharpe = 0.0

    running_max = -np.inf
    drawdowns = []
    for val in daily_values:
        if val > running_max:
            running_max = val
        drawdowns.append((val - running_max) / (running_max + 1e-9))
    max_drawdown = min(drawdowns)
    max_drawdown_str = f"{max_drawdown * 100:.2f}%"

    # Calculate Win Rate
    if total_trades > 0:
        win_rate = (wins / total_trades) * 100.0
        win_rate_str = f"{win_rate:.2f}%"
    else:
        win_rate_str = "N/A"

    metrics = {
        'FinalValue': f"{final_val:.2f}",
        'PercentReturn': final_ret_str,
        'SharpeRatio': f"{sharpe:.3f}",
        'MaxDrawdown': max_drawdown_str,
        'NumTrades': total_trades,
        'Wins': wins,
        'Losses': losses,
        'Maintains': maintains,
        'WinRate': win_rate_str  # Added Win Rate
    }
    return final_val, final_ret_str, daily_dates, daily_values, metrics


def advanced_backtest_portfolio(tickers, model_name,
                                initial_capital=10000,
                                stop_loss_percent=0.05,
                                partial_sell_ratio=0.5,
                                prob_threshold=0.6,
                                trailing_stop=True,
                                take_profit_percent=0.2):
    """
    Perform a naive minute-based backtest on multiple tickers by
    splitting capital equally among them and summing results.
    We'll sum up all trades/wins/losses/maintains across tickers.
    """
    if not tickers:
        return None, "No tickers selected!", [], [], {}

    n = len(tickers)
    capital_each = initial_capital / n

    ticker_values = {}
    total_trades = 0
    wins = 0
    losses = 0
    maintains = 0

    for t in tickers:
        final_val, ret_str, dates, vals, m = advanced_backtest(
            t, model_name,
            initial_capital=capital_each,
            stop_loss_percent=stop_loss_percent,
            partial_sell_ratio=partial_sell_ratio,
            prob_threshold=prob_threshold,
            trailing_stop=trailing_stop,
            take_profit_percent=take_profit_percent
        )
        if final_val is None:
            print(f"Skipping {t} due to error: {ret_str}")
            continue

        # Merge stats
        total_trades += m.get('NumTrades', 0)
        wins += m.get('Wins', 0)
        losses += m.get('Losses', 0)
        maintains += m.get('Maintains', 0)

        df_vals = pd.DataFrame({'Date': dates, 'Value': vals}).set_index('Date')
        ticker_values[t] = df_vals

    if not ticker_values:
        return None, "No valid tickers after backtest", [], [], {}

    # Combine all daily values
    combined = None
    for t, df_vals in ticker_values.items():
        if combined is None:
            combined = df_vals.rename(columns={'Value': t})
        else:
            combined = combined.join(df_vals.rename(columns={'Value': t}), how='outer')

    combined.sort_index(inplace=True)
    combined.fillna(method='ffill', inplace=True)
    combined.fillna(method='bfill', inplace=True)

    combined['PortfolioValue'] = combined.sum(axis=1)
    daily_vals = combined['PortfolioValue'].tolist()
    daily_dates = combined.index.tolist()

    final_val = daily_vals[-1] if daily_vals else initial_capital
    final_ret = (final_val - initial_capital) / initial_capital * 100.0
    final_ret_str = f"{final_ret:.2f}%"

    if len(daily_vals) <= 1:
        metrics = {
            'FinalValue': f"{final_val:.2f}",
            'PercentReturn': final_ret_str,
            'SharpeRatio': "N/A",
            'MaxDrawdown': "N/A",
            'NumTrades': total_trades,
            'Wins': wins,
            'Losses': losses,
            'Maintains': maintains,
            'WinRate': "N/A" if total_trades == 0 else f"{(wins / total_trades) * 100:.2f}%"
        }
        return final_val, final_ret_str, daily_dates, daily_vals, metrics

    # Sharpe & MDD
    daily_returns = []
    for i in range(1, len(daily_vals)):
        ret = (daily_vals[i] - daily_vals[i - 1]) / (daily_vals[i - 1] + 1e-9)
        daily_returns.append(ret)

    if len(daily_returns) > 1:
        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns, ddof=1)
        sharpe = (mean_ret / (std_ret + 1e-9)) * np.sqrt(252 * 390)
    else:
        sharpe = 0.0

    running_max = -np.inf
    drawdowns = []
    for val in daily_vals:
        if val > running_max:
            running_max = val
        drawdowns.append((val - running_max) / (running_max + 1e-9))
    max_drawdown = min(drawdowns)
    max_drawdown_str = f"{max_drawdown * 100:.2f}%"

    # Calculate Win Rate
    if total_trades > 0:
        win_rate = (wins / total_trades) * 100.0
        win_rate_str = f"{win_rate:.2f}%"
    else:
        win_rate_str = "N/A"

    metrics = {
        'FinalValue': f"{final_val:.2f}",
        'PercentReturn': final_ret_str,
        'SharpeRatio': f"{sharpe:.3f}",
        'MaxDrawdown': max_drawdown_str,
        'NumTrades': total_trades,
        'Wins': wins,
        'Losses': losses,
        'Maintains': maintains,
        'WinRate': win_rate_str  # Added Win Rate
    }
    return final_val, final_ret_str, daily_dates, daily_vals, metrics


###############################################################################
# 4. HELPER FUNCTIONS FOR PLOTTING & RENDERING
###############################################################################
def plot_portfolio(daily_dates, daily_values, title):
    """
    Plot daily_values against daily_dates, return a base64-encoded PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(daily_dates, daily_values, label=title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    fig.tight_layout()

    png_img = io.BytesIO()
    fig.savefig(png_img, format='png')
    png_img.seek(0)
    encoded = base64.b64encode(png_img.getvalue()).decode('ascii')
    plt.close(fig)
    return encoded


def render_bootstrap_page(title, body_html):
    """
    Generate a Tailwind-based HTML page with a given title and body content.
    (We keep the function name for minimal code changes, but inside we switch to Tailwind.)
    """
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <title>{title}</title>
      <!-- Tailwind CSS -->
      <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body class="bg-gray-100 text-gray-800">
      <div class="max-w-3xl mx-auto px-4 py-6">
        {body_html}
      </div>
    </body>
    </html>
    """
    return html_template


###############################################################################
# 5. FLASK ROUTES
###############################################################################

@app.route('/')
def index():
    """
    Main index with Tailwind styling.
    """
    body_html = """
    <h1 class="text-2xl font-bold mb-4">Welcome to the Advanced Algorithmic Trading App</h1>
    <ul class="divide-y border border-gray-200 rounded">
      <li class="p-3">
        <a href="{{ url_for('train') }}" class="text-blue-600 hover:underline">Train Models (SSE Progress)</a>
      </li>
      <li class="p-3">
        <a href="{{ url_for('select_backtest_advanced') }}" class="text-blue-600 hover:underline">Run Advanced Backtest (Single Ticker)</a>
      </li>
      <li class="p-3">
        <a href="{{ url_for('select_backtest_portfolio') }}" class="text-blue-600 hover:underline">Run Portfolio Backtest (Multiple Tickers)</a>
      </li>
      <li class="p-3">
        <a href="{{ url_for('predict_next_day') }}" class="text-blue-600 hover:underline">Predict Next Day O/H/L/C</a>
      </li>
    </ul>
    """
    return render_template_string(render_bootstrap_page("Algorithmic Trading App", body_html))


@app.route('/train')
def train():
    """
    SSE Training page with Tailwind-based layout.
    """
    body_html = """
    <h1 class="text-xl font-bold mt-2 mb-4">Train Models for All Tickers</h1>
<p class="mb-2">Click the button below to start training models.</p>
<button class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700" onclick="startTraining()">Start Training</button>

<div id="status" class="mt-3 text-sm font-medium text-gray-700"></div>

<!-- Updated Progress Bar with Percentage Outside -->
<div class="flex items-center mt-4">
  <!-- Progress Bar Container -->
  <div class="w-72 bg-gray-300 h-6 rounded overflow-hidden">
    <div id="progressbar" class="bg-green-500 h-6 transition-width duration-300" style="width:0%;"></div>
  </div>

  <!-- Percentage Indicator -->
  <span id="progresspercent" class="ml-4 text-sm font-medium">0.0%</span>
</div>

<p class="mt-6">
  <a href="{{ url_for('index') }}" class="bg-gray-300 text-black px-4 py-2 rounded hover:bg-gray-400">Back to Home</a>
</p>

<script>
  const statusDiv = document.getElementById('status');
  const progressBar = document.getElementById('progressbar');

  function startTraining(){
    fetch('{{ url_for("start_training") }}')
      .then(response => response.json())
      .then(data => {
        if(data.status === 'ok'){
          statusDiv.innerHTML = "Training started...";
          listenForProgress();
        } else {
          statusDiv.innerHTML = "Error or already training!";
        }
      });
  }

  function listenForProgress(){
    const evtSource = new EventSource('{{ url_for("train_progress") }}');
    evtSource.onmessage = function(e) {
      let [current, total, status] = e.data.split(",");
      if(status === 'training'){
        let pct = 0;
        if(total > 0){
          pct = (current / total) * 100;
        }
        progressBar.style.width = pct + "%";
        document.getElementById('progresspercent').innerText = pct.toFixed(1) + "%";
        statusDiv.innerHTML = "Training in progress... " + current + "/" + total;
      } else if(status === 'done'){
        progressBar.style.width = "100%";
        document.getElementById('progresspercent').innerText = "100.0%";
        statusDiv.innerHTML = "Training complete!";
        evtSource.close();
      }
    };
  }
</script>
    """
    return render_template_string(render_bootstrap_page("Train Models", body_html))


@app.route('/start_training')
def start_training():
    """
    Start the background training thread if not already training.
    """
    with progress_lock:
        if progress_data['status'] == 'training':
            return jsonify({"status": "already_training"})

        thread = threading.Thread(target=train_all_tickers_with_progress)
        thread.start()
    return jsonify({"status": "ok"})


@app.route('/train_progress')
def train_progress():
    """
    SSE endpoint: yields lines in format "data: current,total,status\n\n"
    """

    def generate():
        while True:
            time.sleep(0.3)
            with progress_lock:
                current = progress_data['current']
                total = progress_data['total']
                status = progress_data['status']
            yield f"data: {current},{total},{status}\n\n"
            if status == 'done':
                break

    return Response(generate(), mimetype='text/event-stream')


@app.route('/select_backtest_advanced', methods=['GET', 'POST'])
def select_backtest_advanced():
    """
    Page allowing user to pick a single ticker for advanced backtest.
    """
    tickers = list_tickers()
    model_names = ['LogisticRegression', 'RandomForest', 'MLP']

    if request.method == 'POST':
        data = request.form
        ticker = data.get('ticker')
        model_name = data.get('model_name')
        initial_cap = float(data.get('initial_capital', '10000'))
        stop_loss_percent = float(data.get('stop_loss_percent', '0.05'))
        partial_sell_ratio = float(data.get('partial_sell_ratio', '0.5'))
        prob_threshold = float(data.get('prob_threshold', '0.6'))
        trailing_stop = (data.get('trailing_stop', 'off') == 'on')
        take_profit_percent = float(data.get('take_profit_percent', '0.2'))

        return redirect(url_for('backtest_advanced',
                                ticker=ticker,
                                model_name=model_name,
                                initial_capital=initial_cap,
                                stop_loss_percent=stop_loss_percent,
                                partial_sell_ratio=partial_sell_ratio,
                                prob_threshold=prob_threshold,
                                trailing_stop='1' if trailing_stop else '0',
                                take_profit_percent=take_profit_percent))

    # GET => Render the form
    form_html = """
    <h1 class="text-xl font-bold mb-4">Advanced Backtesting Setup</h1>
    <form method="POST" class="grid gap-4 sm:grid-cols-2">
      <div>
        <label class="block mb-1 font-medium">Ticker:</label>
        <select name="ticker" class="w-full border border-gray-300 rounded px-2 py-1">
          {% for t in tickers %}
          <option value="{{t}}">{{t}}</option>
          {% endfor %}
        </select>
      </div>
      <div>
        <label class="block mb-1 font-medium">Model:</label>
        <select name="model_name" class="w-full border border-gray-300 rounded px-2 py-1">
          {% for m in model_names %}
          <option value="{{m}}">{{m}}</option>
          {% endfor %}
        </select>
      </div>

      <div>
        <label class="block mb-1 font-medium">Initial Capital:</label>
        <input type="number" name="initial_capital" value="10000" step="100"
               class="w-full border border-gray-300 rounded px-2 py-1"/>
      </div>
      <div>
        <label class="block mb-1 font-medium">Stop-Loss %:</label>
        <input type="number" name="stop_loss_percent" value="0.05" step="0.01"
               class="w-full border border-gray-300 rounded px-2 py-1"/>
      </div>
      <div>
        <label class="block mb-1 font-medium">Partial Sell Ratio:</label>
        <input type="number" name="partial_sell_ratio" value="0.5" step="0.1"
               class="w-full border border-gray-300 rounded px-2 py-1"/>
      </div>
      <div>
        <label class="block mb-1 font-medium">Probability Threshold (0~1):</label>
        <input type="number" name="prob_threshold" value="0.6" step="0.05"
               class="w-full border border-gray-300 rounded px-2 py-1"/>
      </div>
      <div>
        <label class="block mb-1 font-medium">Take Profit % (e.g. 0.2=20%):</label>
        <input type="number" name="take_profit_percent" value="0.2" step="0.05"
               class="w-full border border-gray-300 rounded px-2 py-1"/>
      </div>
      <div class="flex items-center space-x-2">
        <input type="checkbox" name="trailing_stop" class="h-4 w-4" id="trailingStopCheckSingle"/>
        <label for="trailingStopCheckSingle" class="text-sm">Trailing Stop?</label>
      </div>

      <div class="sm:col-span-2 mt-2">
        <button type="submit"
                class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 mr-2">
          Run Advanced Backtest
        </button>
        <a href="{{ url_for('index') }}"
           class="bg-gray-300 text-black px-4 py-2 rounded hover:bg-gray-400">Back to Home</a>
      </div>
    </form>
    """
    return render_template_string(render_bootstrap_page("Advanced Backtesting Setup", form_html),
                                  tickers=tickers, model_names=model_names)


@app.route('/backtest_advanced')
def backtest_advanced():
    """
    Perform advanced backtest for a single ticker.
    """
    ticker = request.args.get('ticker')
    model_name = request.args.get('model_name')
    initial_cap = float(request.args.get('initial_capital', '10000'))
    stop_loss_percent = float(request.args.get('stop_loss_percent', '0.05'))
    partial_sell_ratio = float(request.args.get('partial_sell_ratio', '0.5'))
    prob_threshold = float(request.args.get('prob_threshold', '0.6'))
    trailing_stop = (request.args.get('trailing_stop', '0') == '1')
    take_profit_percent = float(request.args.get('take_profit_percent', '0.2'))

    final_val, final_ret_str, daily_dates, daily_values, metrics = advanced_backtest(
        ticker, model_name, initial_cap, stop_loss_percent, partial_sell_ratio,
        prob_threshold, trailing_stop, take_profit_percent
    )

    if final_val is None:
        body_html = f"<p class='text-red-600'>Error: {final_ret_str}</p>"
        return render_template_string(render_bootstrap_page("Advanced Backtest Results", body_html))

    # Plot
    encoded_img = plot_portfolio(daily_dates, daily_values, f"{ticker} - {model_name}")

    # Include trade stats in results
    trade_stats_html = f"""
    <p><strong>Number of Trades:</strong> {metrics.get('NumTrades', 0)}</p>
    <p><strong>Wins:</strong> {metrics.get('Wins', 0)}</p>
    <p><strong>Losses:</strong> {metrics.get('Losses', 0)}</p>
    <p><strong>Maintains:</strong> {metrics.get('Maintains', 0)}</p>
    <p><strong>Win Rate:</strong> {metrics.get('WinRate', 'N/A')}</p>
    """

    result_html = f"""
    <p><strong>Final Capital:</strong> {metrics['FinalValue']}</p>
    <p><strong>Percent Return:</strong> {metrics['PercentReturn']}</p>
    <p><strong>Sharpe Ratio:</strong> {metrics['SharpeRatio']}</p>
    <p><strong>Max Drawdown:</strong> {metrics['MaxDrawdown']}</p>
    {trade_stats_html}
    """

    # Re-run form
    tickers = list_tickers()
    model_names_list = ['LogisticRegression', 'RandomForest', 'MLP']
    trailing_check = 'checked' if trailing_stop else ''

    re_run_form = f"""
    <hr class="my-4 border-gray-300">
    <h3 class="text-lg font-semibold mb-2">Refine Your Backtest</h3>
    <form method="GET" action="{url_for('backtest_advanced')}" class="grid gap-4 sm:grid-cols-2 mt-2">
      <div>
        <label class="block mb-1 font-medium">Ticker:</label>
        <select name="ticker" class="w-full border border-gray-300 rounded px-2 py-1">
          {"".join(f'<option value="{t}" {"selected" if t == ticker else ""}>{t}</option>' for t in tickers)}
        </select>
      </div>
      <div>
        <label class="block mb-1 font-medium">Model:</label>
        <select name="model_name" class="w-full border border-gray-300 rounded px-2 py-1">
          {"".join(f'<option value="{m}" {"selected" if m == model_name else ""}>{m}</option>'
                   for m in model_names_list)}
        </select>
      </div>
      <div>
        <label class="block mb-1 font-medium">Initial Capital:</label>
        <input type="number" name="initial_capital" value="{initial_cap}" step="100"
               class="w-full border border-gray-300 rounded px-2 py-1"/>
      </div>
      <div>
        <label class="block mb-1 font-medium">Stop-Loss %:</label>
        <input type="number" name="stop_loss_percent" value="{stop_loss_percent}" step="0.01"
               class="w-full border border-gray-300 rounded px-2 py-1"/>
      </div>
      <div>
        <label class="block mb-1 font-medium">Partial Sell Ratio:</label>
        <input type="number" name="partial_sell_ratio" value="{partial_sell_ratio}" step="0.1"
               class="w-full border border-gray-300 rounded px-2 py-1"/>
      </div>
      <div>
        <label class="block mb-1 font-medium">Probability Threshold:</label>
        <input type="number" name="prob_threshold" value="{prob_threshold}" step="0.05"
               class="w-full border border-gray-300 rounded px-2 py-1"/>
      </div>
      <div>
        <label class="block mb-1 font-medium">Take Profit %:</label>
        <input type="number" name="take_profit_percent" value="{take_profit_percent}" step="0.05"
               class="w-full border border-gray-300 rounded px-2 py-1"/>
      </div>
      <div class="flex items-center space-x-2">
        <input class="h-4 w-4" type="checkbox" name="trailing_stop" {trailing_check} id="checkTS"/>
        <label class="text-sm" for="checkTS">Trailing Stop?</label>
      </div>
      <div class="sm:col-span-2 mt-2">
        <button type="submit"
                class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
          Re-Run Backtest
        </button>
      </div>
    </form>
    """

    body_html = f"""
    <h1 class="text-xl font-bold mb-4">Advanced Backtest Results</h1>
    <h5 class="text-md font-semibold mb-3">{ticker} - {model_name}</h5>
    <div class="mt-3 text-sm">{result_html}</div>
    <div class="mt-4">
      <img src="data:image/png;base64,{encoded_img}" class="mx-auto" alt="Chart"/>
    </div>
    <div class="mt-5">
      {re_run_form}
    </div>
    <hr class="my-6">
    <p>
      <a href="{{{{ url_for('select_backtest_advanced') }}}}"
         class="bg-gray-300 text-black px-4 py-2 rounded hover:bg-gray-400">
         Go to Full Advanced Setup Page
      </a>
    </p>
    """
    return render_template_string(render_bootstrap_page("Advanced Backtest Results", body_html))


@app.route('/select_backtest_portfolio', methods=['GET', 'POST'])
def select_backtest_portfolio():
    """
    Displays a user-friendly form for selecting multiple tickers, with
    options to add/remove them, plus model/capital parameters.
    """
    tickers = list_tickers()
    model_names = ['LogisticRegression', 'RandomForest', 'MLP']

    if request.method == 'POST':
        selected_str = request.form.get('selected_tickers', '')
        selected_list = [t.strip() for t in selected_str.split(',') if t.strip()]

        model_name = request.form.get('model_name')
        initial_cap = float(request.form.get('initial_capital', '10000'))
        stop_loss_percent = float(request.form.get('stop_loss_percent', '0.05'))
        partial_sell_ratio = float(request.form.get('partial_sell_ratio', '0.5'))
        prob_threshold = float(request.form.get('prob_threshold', '0.6'))
        trailing_stop = (request.form.get('trailing_stop', 'off') == 'on')
        take_profit_percent = float(request.form.get('take_profit_percent', '0.2'))

        joined_tickers = ",".join(selected_list)
        return redirect(url_for('backtest_portfolio',
                                tickers=joined_tickers,
                                model_name=model_name,
                                initial_capital=initial_cap,
                                stop_loss_percent=stop_loss_percent,
                                partial_sell_ratio=partial_sell_ratio,
                                prob_threshold=prob_threshold,
                                trailing_stop='1' if trailing_stop else '0',
                                take_profit_percent=take_profit_percent))

    # GET => show the multiple ticker selection form
    form_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <title>Portfolio Backtesting Setup</title>
      <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <style>
        .ticker-box { min-width: 150px; max-height: 240px; }
      </style>
    </head>
    <body class="bg-gray-100 text-gray-800">
      <div class="max-w-4xl mx-auto px-4 py-6">
        <h1 class="text-xl font-bold mb-4">Portfolio Backtesting Setup</h1>
        <p class="text-sm mb-4">
          Type or search a ticker below and click <strong>Add</strong> to move it to your selected list.
          Click <strong>Select All</strong> to add all tickers at once. Then <strong>Remove</strong> to remove from your list if needed.
        </p>

        <div class="flex flex-col sm:flex-row mb-4 space-y-4 sm:space-y-0 sm:space-x-6">
          <div class="sm:w-1/2">
            <label for="ticker_search" class="block mb-1 font-medium">Search Ticker:</label>
            <input type="text" id="ticker_search"
                   onkeyup="filterTickers()"
                   class="w-full border border-gray-300 rounded px-2 py-1"
                   placeholder="Type to filter..." />
          </div>
        </div>

        <div class="flex flex-col sm:flex-row gap-6">
          <div>
            <h5 class="font-semibold mb-1">All Tickers</h5>
            <select id="all_tickers" size="10"
                    class="ticker-box w-full border border-gray-300 rounded px-2 py-1">
              {% for t in tickers %}
              <option value="{{t}}">{{t}}</option>
              {% endfor %}
            </select>
            <div class="mt-2 space-x-2">
              <button type="button"
                      class="bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700 text-sm"
                      onclick="addTicker()">
                Add &raquo;
              </button>
              <button type="button"
                      class="bg-green-600 text-white px-3 py-1 rounded hover:bg-green-700 text-sm"
                      onclick="selectAllTickers()">
                Select All
              </button>
            </div>
          </div>

          <div>
            <h5 class="font-semibold mb-1">Selected Tickers</h5>
            <select id="selected_tickers_list" size="10"
                    class="ticker-box w-full border border-gray-300 rounded px-2 py-1">
            </select>
            <button type="button"
                    class="bg-red-600 text-white px-3 py-1 rounded hover:bg-red-700 text-sm mt-2"
                    onclick="removeTicker()">
              &laquo; Remove
            </button>
          </div>
        </div>

        <hr class="my-6 border-gray-300">

        <form method="POST" id="portfolioForm" class="grid gap-4 sm:grid-cols-3">
          <input type="hidden" name="selected_tickers" id="hidden_selected_tickers">

          <div>
            <label class="block mb-1 font-medium">Model:</label>
            <select name="model_name" class="w-full border border-gray-300 rounded px-2 py-1">
              {% for m in model_names %}
              <option value="{{m}}">{{m}}</option>
              {% endfor %}
            </select>
          </div>

          <div>
            <label class="block mb-1 font-medium">Initial Capital:</label>
            <input type="number" name="initial_capital" value="10000" step="100"
                   class="w-full border border-gray-300 rounded px-2 py-1"/>
          </div>

          <div>
            <label class="block mb-1 font-medium">Stop-Loss %:</label>
            <input type="number" name="stop_loss_percent" value="0.05" step="0.01"
                   class="w-full border border-gray-300 rounded px-2 py-1"/>
          </div>

          <div>
            <label class="block mb-1 font-medium">Partial Sell Ratio:</label>
            <input type="number" name="partial_sell_ratio" value="0.5" step="0.1"
                   class="w-full border border-gray-300 rounded px-2 py-1"/>
          </div>

          <div>
            <label class="block mb-1 font-medium">Probability Threshold (0~1):</label>
            <input type="number" name="prob_threshold" value="0.6" step="0.05"
                   class="w-full border border-gray-300 rounded px-2 py-1"/>
          </div>

          <div class="flex items-center space-x-2 mt-6">
            <input type="checkbox" name="trailing_stop" class="h-4 w-4" id="trailingStopCheck" />
            <label for="trailingStopCheck" class="text-sm">Trailing Stop?</label>
          </div>

          <div>
            <label class="block mb-1 font-medium">Take Profit %:</label>
            <input type="number" name="take_profit_percent" value="0.2" step="0.05"
                   class="w-full border border-gray-300 rounded px-2 py-1"/>
          </div>

          <div class="sm:col-span-3">
            <button type="submit"
                    onclick="prepareSelectedTickers()"
                    class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 mr-2">
              Run Portfolio Backtest
            </button>
            <a href="{{ url_for('index') }}"
               class="bg-gray-300 text-black px-4 py-2 rounded hover:bg-gray-400">Back to Home</a>
          </div>
        </form>
      </div>

      <script>
        function addTicker(){
          const allList = document.getElementById('all_tickers');
          const selList = document.getElementById('selected_tickers_list');
          if(allList.selectedIndex >= 0){
            let opt = allList.options[allList.selectedIndex];
            let exists = false;
            for(let i=0; i<selList.options.length; i++){
              if(selList.options[i].value === opt.value){
                exists = true;
                break;
              }
            }
            if(!exists){
              let newOpt = document.createElement('option');
              newOpt.value = opt.value;
              newOpt.text = opt.text;
              selList.add(newOpt);
            }
          }
        }

        function selectAllTickers(){
          const allList = document.getElementById('all_tickers');
          const selList = document.getElementById('selected_tickers_list');
          for(let i=0; i<allList.options.length; i++){
            let opt = allList.options[i];
            let exists = false;
            for(let j=0; j<selList.options.length; j++){
              if(selList.options[j].value === opt.value){
                exists = true;
                break;
              }
            }
            if(!exists){
              let newOpt = document.createElement('option');
              newOpt.value = opt.value;
              newOpt.text = opt.text;
              selList.add(newOpt);
            }
          }
        }

        function removeTicker(){
          const selList = document.getElementById('selected_tickers_list');
          if(selList.selectedIndex >= 0){
            selList.remove(selList.selectedIndex);
          }
        }

        function prepareSelectedTickers(){
          const selList = document.getElementById('selected_tickers_list');
          const hiddenField = document.getElementById('hidden_selected_tickers');
          let values = [];
          for(let i=0; i<selList.options.length; i++){
            values.push(selList.options[i].value);
          }
          hiddenField.value = values.join(',');
        }

        function filterTickers(){
          let input = document.getElementById('ticker_search');
          let filter = input.value.toUpperCase();
          let allList = document.getElementById('all_tickers');
          for(let i=0; i<allList.options.length; i++){
            let txt = allList.options[i].text.toUpperCase();
            allList.options[i].style.display = (txt.indexOf(filter) > -1) ? "" : "none";
          }
        }
      </script>
    </body>
    </html>
    """
    return render_template_string(form_html, tickers=tickers, model_names=model_names)


@app.route('/backtest_portfolio')
def backtest_portfolio():
    """
    Executes a multi-ticker portfolio backtest, displays result.
    """
    tickers_str = request.args.get('tickers', '')
    model_name = request.args.get('model_name')
    initial_cap = float(request.args.get('initial_capital', '10000'))
    stop_loss_percent = float(request.args.get('stop_loss_percent', '0.05'))
    partial_sell_ratio = float(request.args.get('partial_sell_ratio', '0.5'))
    prob_threshold = float(request.args.get('prob_threshold', '0.6'))
    trailing_stop = (request.args.get('trailing_stop', '0') == '1')
    take_profit_percent = float(request.args.get('take_profit_percent', '0.2'))

    if not tickers_str.strip():
        return "<p class='text-red-600'>Error: No tickers provided.</p>"

    selected_tickers = [t.strip() for t in tickers_str.split(',') if t.strip()]
    final_val, final_ret_str, daily_dates, daily_values, metrics = advanced_backtest_portfolio(
        selected_tickers, model_name,
        initial_capital=initial_cap,
        stop_loss_percent=stop_loss_percent,
        partial_sell_ratio=partial_sell_ratio,
        prob_threshold=prob_threshold,
        trailing_stop=trailing_stop,
        take_profit_percent=take_profit_percent
    )
    if final_val is None:
        return f"<p class='text-red-600'>Error: {final_ret_str}</p>"

    # Plot the combined daily_values
    encoded_img = plot_portfolio(daily_dates, daily_values,
                                 f"{', '.join(selected_tickers)} - {model_name}")

    # Include trade stats in results
    trade_stats_html = f"""
    <p><strong>Number of Trades:</strong> {metrics.get('NumTrades', 0)}</p>
    <p><strong>Wins:</strong> {metrics.get('Wins', 0)}</p>
    <p><strong>Losses:</strong> {metrics.get('Losses', 0)}</p>
    <p><strong>Maintains:</strong> {metrics.get('Maintains', 0)}</p>
    <p><strong>Win Rate:</strong> {metrics.get('WinRate', 'N/A')}</p>
    """

    result_html = f"""
    <p><strong>Final Capital:</strong> {metrics['FinalValue']}</p>
    <p><strong>Percent Return:</strong> {metrics['PercentReturn']}</p>
    <p><strong>Sharpe Ratio:</strong> {metrics['SharpeRatio']}</p>
    <p><strong>Max Drawdown:</strong> {metrics['MaxDrawdown']}</p>
    {trade_stats_html}
    """

    trailing_check = 'checked' if trailing_stop else ''
    re_run_form = f"""
    <hr class="my-4 border-gray-300">
    <h3 class="text-lg font-semibold mb-2">Refine Your Portfolio Backtest</h3>
    <form method="GET" action="{url_for('backtest_portfolio')}" class="grid gap-4 sm:grid-cols-3 mt-2">
      <input type="hidden" name="tickers" value="{tickers_str}" />

      <div>
        <label class="block mb-1 font-medium">Model:</label>
        <select name="model_name" class="w-full border border-gray-300 rounded px-2 py-1">
          <option value="LogisticRegression" {"selected" if model_name == "LogisticRegression" else ""}>LogisticRegression</option>
          <option value="RandomForest" {"selected" if model_name == "RandomForest" else ""}>RandomForest</option>
          <option value="MLP" {"selected" if model_name == "MLP" else ""}>MLP</option>
        </select>
      </div>

      <div>
        <label class="block mb-1 font-medium">Initial Capital:</label>
        <input type="number" name="initial_capital" value="{initial_cap}" step="100"
               class="w-full border border-gray-300 rounded px-2 py-1"/>
      </div>

      <div>
        <label class="block mb-1 font-medium">Stop-Loss %:</label>
        <input type="number" name="stop_loss_percent" value="{stop_loss_percent}" step="0.01"
               class="w-full border border-gray-300 rounded px-2 py-1"/>
      </div>

      <div>
        <label class="block mb-1 font-medium">Partial Sell Ratio:</label>
        <input type="number" name="partial_sell_ratio" value="{partial_sell_ratio}" step="0.1"
               class="w-full border border-gray-300 rounded px-2 py-1"/>
      </div>

      <div>
        <label class="block mb-1 font-medium">Probability Threshold:</label>
        <input type="number" name="prob_threshold" value="{prob_threshold}" step="0.05"
               class="w-full border border-gray-300 rounded px-2 py-1"/>
      </div>

      <div class="flex items-center space-x-2">
        <input class="h-4 w-4" type="checkbox" name="trailing_stop" {trailing_check} id="trailingStopCheck2">
        <label class="text-sm" for="trailingStopCheck2">Trailing Stop?</label>
      </div>

      <div>
        <label class="block mb-1 font-medium">Take Profit %:</label>
        <input type="number" name="take_profit_percent" value="{take_profit_percent}" step="0.05"
               class="w-full border border-gray-300 rounded px-2 py-1"/>
      </div>

      <div class="sm:col-span-3">
        <button type="submit" class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
          Re-Run Portfolio Backtest
        </button>
      </div>
    </form>
    """

    body_html = f"""
    <h1 class="text-xl font-bold mb-4">Portfolio Backtest Results</h1>
    <h5 class="text-md font-semibold mb-2">Tickers: {', '.join(selected_tickers)} - {model_name}</h5>
    <div class="mt-3 text-sm">{result_html}</div>
    <div class="mt-4">
      <img src="data:image/png;base64,{encoded_img}" class="mx-auto" alt="Portfolio Chart"/>
    </div>

    <div class="mt-5">{re_run_form}</div>
    <hr class="my-6 border-gray-300">
    <p>
      <a href="{{{{ url_for('select_backtest_portfolio') }}}}"
         class="bg-gray-300 text-black px-4 py-2 rounded hover:bg-gray-400">
         Back to Portfolio Setup
      </a>
    </p>
    """
    return render_template_string(render_bootstrap_page("Portfolio Backtest Results", body_html))


@app.route('/predict_next_day', methods=['GET', 'POST'])
def predict_next_day():
    """
    Lets user pick a ticker, then predict tomorrow's O/H/L/C using the ensemble regression models.
    """
    tickers = list_tickers()

    if request.method == 'POST':
        ticker = request.form.get('ticker')
        if ticker not in MODELS:
            body_html = "<p class='text-red-600'>Error: No model for this ticker. Please train first.</p>"
            return render_template_string(render_bootstrap_page("Prediction Error", body_html))

        try:
            df = load_data_for_ticker(ticker)
            df = compute_indicators(df)
            if df.empty:
                body_html = "<p class='text-red-600'>Error: No data available.</p>"
                return render_template_string(render_bootstrap_page("Prediction Error", body_html))

            last_row = df.iloc[[-1]].copy()
            last_row.dropna(inplace=True)
            if last_row.empty:
                body_html = "<p class='text-red-600'>Error: Not enough data to predict.</p>"
                return render_template_string(render_bootstrap_page("Prediction Error", body_html))

            # Must match the updated feature_cols
            feature_cols = [
                'Close', 'High', 'Low', 'Open', 'Volume',
                'MA_10', 'MA_50', 'MA_200',
                'Boll_Upper', 'Boll_Lower',
                'Daily_Return', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
                'OBV', 'Stoch_%K', 'Stoch_%D'
            ]
            for col in feature_cols:
                if col not in last_row.columns:
                    body_html = "<p class='text-red-600'>Error: Missing indicator columns. Please re-train.</p>"
                    return render_template_string(render_bootstrap_page("Prediction Error", body_html))

            X_last = last_row[feature_cols].values
            scaler = SCALERS[ticker]
            X_last_scaled = scaler.transform(X_last)

            all_models = MODELS[ticker]

            # Each reg key = (rf_model, mlp_model)
            open_reg = all_models['NextOpenReg']
            high_reg = all_models['NextHighReg']
            low_reg = all_models['NextLowReg']
            close_reg = all_models['NextCloseReg']

            # Correctly unpack and pass arguments to ensemble_predict
            pred_open = ensemble_predict(open_reg[0], open_reg[1], X_last_scaled)[0]
            pred_high = ensemble_predict(high_reg[0], high_reg[1], X_last_scaled)[0]
            pred_low = ensemble_predict(low_reg[0], low_reg[1], X_last_scaled)[0]
            pred_close = ensemble_predict(close_reg[0], close_reg[1], X_last_scaled)[0]

            current_close = last_row['Close'].values[0]

            # Additional clamp for sanity-check:
            MAX_GAP_PERCENT = 0.2  # 20%

            # for each predicted price, don't deviate more than ±20% from current close
            def clamp_price(pred):
                diff = pred - current_close
                pct = diff / (current_close + 1e-9)
                if pct > MAX_GAP_PERCENT:
                    pred = current_close * (1 + MAX_GAP_PERCENT)
                elif pct < -MAX_GAP_PERCENT:
                    pred = current_close * (1 - MAX_GAP_PERCENT)
                return pred

            pred_open = clamp_price(pred_open)
            pred_high = clamp_price(pred_high)
            pred_low = clamp_price(pred_low)
            pred_close = clamp_price(pred_close)

            pct_diff = (pred_close - current_close) / current_close * 100.0

            if pct_diff > 2.0:
                suggestion = "BUY (Predicted close is 2%+ above current)"
            elif pct_diff < -2.0:
                suggestion = "SELL (Predicted close is 2%+ below current)"
            else:
                suggestion = "HOLD (Predicted move is within ±2%)"

            # Styled Result HTML
            result_html = f"""
            <div class="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
                <h3 class="text-2xl font-bold mb-4">Predictions for Ticker: {ticker}</h3>
                <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                        <p><strong>Tomorrow's Predicted OPEN:</strong> {pred_open:.2f}</p>
                        <p><strong>Tomorrow's Predicted HIGH:</strong> {pred_high:.2f}</p>
                        <p><strong>Tomorrow's Predicted LOW:</strong> {pred_low:.2f}</p>
                        <p><strong>Tomorrow's Predicted CLOSE:</strong> {pred_close:.2f}</p>
                    </div>
                    <div>
                        <p><strong>Current Close:</strong> {current_close:.2f}</p>
                        <p><strong>Predicted % Diff vs Current Close:</strong> {pct_diff:.2f}%</p>
                        <p><strong>Suggested Action:</strong> {suggestion}</p>
                    </div>
                </div>
                <div class="mt-6">
                    <a href='{url_for('predict_next_day')}' class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
                        Back to Predict Page
                    </a>
                </div>
            </div>
            """

            # Wrap the result_html within the Tailwind-styled page
            full_body_html = f"""
            <h1 class="text-2xl font-bold mb-6">Prediction Results</h1>
            {result_html}
            """

            return render_template_string(render_bootstrap_page("Prediction Results", full_body_html))

        except Exception as e:
            body_html = f"<p class='text-red-600'>Error during prediction: {str(e)}</p>"
            return render_template_string(render_bootstrap_page("Prediction Error", body_html))

    # GET method remains unchanged
    form_html = """
    <h1 class="text-xl font-bold mb-4">Predict Next Day O/H/L/C</h1>
    <form method="POST" class="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-4">
      <div class="mb-4">
        <label class="block text-gray-700 text-sm font-bold mb-2" for="ticker">
          Ticker:
        </label>
        <select name="ticker" class="shadow appearance-1 border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline">
          {% for t in tickers %}
          <option value="{{t}}">{{t}}</option>
          {% endfor %}
        </select>
      </div>
      <div class="flex items-center justify-between">
        <button type="submit" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline">
          Predict Next Day
        </button>
        <a href="{{ url_for('index') }}" class="inline-block align-baseline font-bold text-sm text-blue-500 hover:text-blue-800">
          Back to Home
        </a>
      </div>
    </form>
    """
    return render_template_string(render_bootstrap_page("Predict Next Day O/H/L/C", form_html),
                                  tickers=tickers)


###############################################################################
# MAIN ENTRY
###############################################################################
if __name__ == '__main__':
    load_models()
    app.run(debug=True)