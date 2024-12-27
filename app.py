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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use('Agg')  # For servers without GUI
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

###############################################################################
# GLOBALS
###############################################################################
MODELS = {}
SCALERS = {}

# Progress data for SSE-based training
progress_data = {
    'current': 0,
    'total': 0,
    'status': 'idle'
}
progress_lock = threading.Lock()


###############################################################################
# 1. HELPER FUNCTIONS TO LIST TICKERS / LOAD & PROCESS CSV
###############################################################################
def list_tickers(hist_folder='hist'):
    csv_files = glob.glob(os.path.join(hist_folder, '*.csv'))
    tickers = [os.path.splitext(os.path.basename(f))[0] for f in csv_files]
    return tickers


def load_data_for_ticker(ticker, hist_folder='hist'):
    file_path = os.path.join(hist_folder, f"{ticker}.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No CSV file found for ticker {ticker} at {file_path}.")

    # Rename "Datetime" to "Date" to keep consistency
    df = pd.read_csv(file_path, skiprows=[1, 2])
    df.rename(columns={
        "Price": "Date",  # Rename "Price" to "Date" if applicable
        "Datetime": "Date",  # Ensure "Datetime" is renamed to "Date"
        "Close": "Close",
        "High": "High",
        "Low": "Low",
        "Open": "Open",
        "Volume": "Volume"
    }, inplace=True)

    # Parse 1-min timestamps (may contain time zone)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df.set_index('Date', inplace=True, drop=True)
    return df


def compute_indicators(df):
    df = df.copy()
    df.sort_index(inplace=True)

    # Intraday rolling windows (e.g., 10-min, 50-min, 200-min)
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()

    df['Daily_Return'] = df['Close'].pct_change()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI_14'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    df.dropna(inplace=True)
    return df


def create_labels(df, threshold=0.0025):  # Lowered threshold to 0.25%
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

    # Log class distribution
    class_counts = df['Action'].value_counts()
    print(f"Class distribution for threshold {threshold}:")
    print(class_counts)

    return df


###############################################################################
# 2. TRAINING MODELS (CLASSIFICATION + REGRESSION FOR NEXT-DAY O/H/L/C)
###############################################################################
def train_models_for_ticker(ticker, df):
    """
    Trains classification models for BUY/SELL/HOLD,
    plus regression models for next-day Open, High, Low, Close.
    """
    # ==================== CLASSIFICATION (BUY/SELL/HOLD) ====================
    feature_cols = [
        'Close', 'High', 'Low', 'Open', 'Volume',
        'MA_10', 'MA_50', 'MA_200',
        'Daily_Return', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist'
    ]
    X_class = df[feature_cols].values
    y_class = df['Action'].values  # 0=SELL, 1=HOLD, 2=BUY

    # Check if there are at least two classes
    unique_classes = np.unique(y_class)
    if len(unique_classes) < 2:
        print(f"Skipping ticker {ticker}: only one class present ({unique_classes[0]})")
        return None, None

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_class, y_class, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_c_scaled = scaler.fit_transform(X_train_c)
    X_test_c_scaled = scaler.transform(X_test_c)

    # Classification models
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

    # ==================== REGRESSION FOR NEXT-DAY O/H/L/C ====================
    df_reg = df.copy()
    # SHIFT target columns by -1
    df_reg['Next_Open'] = df_reg['Open'].shift(-1)
    df_reg['Next_High'] = df_reg['High'].shift(-1)
    df_reg['Next_Low'] = df_reg['Low'].shift(-1)
    df_reg['Next_Close'] = df_reg['Close'].shift(-1)
    df_reg.dropna(inplace=True)

    X_reg = df_reg[feature_cols].values
    # Create separate Y vectors
    y_open = df_reg['Next_Open'].values
    y_high = df_reg['Next_High'].values
    y_low = df_reg['Next_Low'].values
    y_close = df_reg['Next_Close'].values

    # For consistency, let's match the classification split sizes
    # (Though you could do separate splits if you prefer)
    X_train_r, X_test_r = X_reg[:len(X_train_c)], X_reg[len(X_train_c):]

    X_train_r_scaled = scaler.transform(X_train_r)

    # Example: using RandomForestRegressor
    open_reg = RandomForestRegressor(n_estimators=50)
    high_reg = RandomForestRegressor(n_estimators=50)
    low_reg = RandomForestRegressor(n_estimators=50)
    close_reg = RandomForestRegressor(n_estimators=50)

    open_reg.fit(X_train_r_scaled, y_open[:len(X_train_r)])
    high_reg.fit(X_train_r_scaled, y_high[:len(X_train_r)])
    low_reg.fit(X_train_r_scaled, y_low[:len(X_train_r)])
    close_reg.fit(X_train_r_scaled, y_close[:len(X_train_r)])

    regression_models = {
        'NextOpenReg': open_reg,
        'NextHighReg': high_reg,
        'NextLowReg': low_reg,
        'NextCloseReg': close_reg
    }

    # Combine everything
    all_models = {**classification_models, **regression_models}
    return all_models, scaler


def train_all_tickers_with_progress():
    """
    Train models for each CSV found in `hist/`.
    Update progress_data for SSE-based progress bar.
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
            if models_dict is not None and scaler is not None:
                MODELS[ticker] = models_dict
                SCALERS[ticker] = scaler
                print(f"Trained models for ticker: {ticker}")
            else:
                print(f"Skipped training for ticker: {ticker} due to insufficient classes.")

        except Exception as e:
            print(f"Error training ticker {ticker}: {e}")

        with progress_lock:
            progress_data['current'] += 1

        time.sleep(0.5)  # purely to visualize progress in a slower manner

    save_models()
    with progress_lock:
        progress_data['status'] = 'done'


###############################################################################
# 3. SAVING / LOADING MODELS
###############################################################################
def save_models(filename='models.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((MODELS, SCALERS), f)


def load_models(filename='models.pkl'):
    global MODELS, SCALERS
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            MODELS, SCALERS = pickle.load(f)


###############################################################################
# 4. ADVANCED BACKTESTING (MINUTE-BASED)
###############################################################################
def advanced_backtest_portfolio(tickers, model_name,
                                initial_capital=10000,
                                stop_loss_percent=0.05,
                                partial_sell_ratio=0.5,
                                prob_threshold=0.6,
                                trailing_stop=True,
                                take_profit_percent=0.2):
    """
    Distributes 'initial_capital' equally among 'tickers'.
    Runs advanced_backtest on each ticker with that portion of capital,
    then sums up daily portfolio values.
    """
    if not tickers:
        return None, "No tickers selected!", [], [], {}

    # How many tickers? We'll split capital equally.
    n = len(tickers)
    capital_per_ticker = initial_capital / n

    # A dict to store each ticker's daily values by date
    ticker_values_by_date = {}

    for t in tickers:
        # Reuse your existing advanced_backtest but pass 'capital_per_ticker'
        final_val, final_ret_str, daily_dates, daily_vals, metrics = advanced_backtest(
            t, model_name,
            initial_capital=capital_per_ticker,
            stop_loss_percent=stop_loss_percent,
            partial_sell_ratio=partial_sell_ratio,
            prob_threshold=prob_threshold,
            trailing_stop=trailing_stop,
            take_profit_percent=take_profit_percent
        )
        if final_val is None:
            # if advanced_backtest returned an error, skip or handle
            print(f"Skipping {t} due to error: {final_ret_str}")
            continue

        # Store the time series in a DataFrame for easy merging
        df_vals = pd.DataFrame({
            'Date': daily_dates,
            'Value': daily_vals
        }).set_index('Date')
        ticker_values_by_date[t] = df_vals

    if not ticker_values_by_date:
        return None, "No valid tickers after backtest", [], [], {}

    # Merge them by date, fill missing with forward/back fill if needed, sum across columns
    combined_df = None
    for t, df_vals in ticker_values_by_date.items():
        if combined_df is None:
            combined_df = df_vals.rename(columns={'Value': t})
        else:
            combined_df = combined_df.join(df_vals.rename(columns={'Value': t}), how='outer')

    # Sort by date, fill missing
    combined_df.sort_index(inplace=True)
    combined_df.fillna(method='ffill', inplace=True)
    combined_df.fillna(method='bfill', inplace=True)

    # Sum across all tickers to get portfolio value
    combined_df['PortfolioValue'] = combined_df.sum(axis=1)

    # Now compute final metrics (Sharpe, etc.) on combined portfolio
    daily_values = combined_df['PortfolioValue'].tolist()
    daily_dates = combined_df.index.tolist()

    final_value = daily_values[-1] if daily_values else initial_capital
    final_return = (final_value - initial_capital) / initial_capital * 100.0
    final_return_str = f"{final_return:.2f}%"

    if len(daily_values) <= 1:
        # Not enough data
        metrics = {
            'FinalValue': f"{final_value:.2f}",
            'PercentReturn': final_return_str,
            'SharpeRatio': "N/A",
            'MaxDrawdown': "N/A"
        }
        return final_value, final_return_str, daily_dates, daily_values, metrics

    # Compute daily returns
    daily_returns = []
    for i in range(1, len(daily_values)):
        ret = (daily_values[i] - daily_values[i-1]) / (daily_values[i-1] + 1e-9)
        daily_returns.append(ret)

    if len(daily_returns) > 1:
        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns, ddof=1)
        sharpe = (mean_ret / (std_ret + 1e-9)) * np.sqrt(252*390)  # for 1-min intraday
    else:
        sharpe = 0.0

    running_max = -np.inf
    drawdowns = []
    for val in daily_values:
        if val > running_max:
            running_max = val
        dd = (val - running_max) / (running_max + 1e-9)
        drawdowns.append(dd)
    max_drawdown = min(drawdowns)
    max_drawdown_str = f"{max_drawdown * 100:.2f}%"

    metrics = {
        'FinalValue': f"{final_value:.2f}",
        'PercentReturn': final_return_str,
        'SharpeRatio': f"{sharpe:.3f}",
        'MaxDrawdown': max_drawdown_str
    }

    return final_value, final_return_str, daily_dates, daily_values, metrics

def advanced_backtest(ticker, model_name,
                      initial_capital=10000,
                      stop_loss_percent=0.05,
                      partial_sell_ratio=0.5,
                      prob_threshold=0.6,
                      trailing_stop=True,
                      take_profit_percent=0.2):
    """
    Each row = 1 minute. The strategy can place multiple intraday trades
    as signals occur. Minimal changes from the daily approach.
    """
    if ticker not in MODELS or ticker not in SCALERS:
        return None, "No models found for this ticker.", None, None, {}

    if model_name not in MODELS[ticker]:
        return None, f"Model {model_name} not found for {ticker}.", None, None, {}

    df = load_data_for_ticker(ticker)
    df = compute_indicators(df)
    df = create_labels(df)

    classifier = MODELS[ticker][model_name]
    feature_cols = [
        'Close', 'High', 'Low', 'Open', 'Volume',
        'MA_10', 'MA_50', 'MA_200',
        'Daily_Return', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist'
    ]
    X = df[feature_cols].values
    X_scaled = SCALERS[ticker].transform(X)

    has_proba = hasattr(classifier, "predict_proba")
    if has_proba:
        probas = classifier.predict_proba(X_scaled)
        predicted_actions = classifier.predict(X_scaled)
    else:
        predicted_actions = classifier.predict(X_scaled)
        probas = None

    df['Prediction'] = np.roll(predicted_actions, 1, axis=0)
    if probas is not None:
        shifted_probas = np.roll(probas, 1, axis=0)
    else:
        shifted_probas = None

    positions = []
    capital = float(initial_capital)
    daily_values = []
    daily_dates = []

    for i, (date_idx, row) in enumerate(df.iterrows()):
        current_price = row['Close']
        daily_portfolio_value = capital + sum(pos['shares'] * current_price for pos in positions)
        daily_values.append(daily_portfolio_value)
        daily_dates.append(date_idx)

        # Update trailing stops
        for pos in positions:
            if trailing_stop:
                if current_price > pos['highest_price']:
                    pos['highest_price'] = current_price
                new_sl = pos['highest_price'] * (1 - stop_loss_percent)
                if new_sl > pos['stop_loss_price']:
                    pos['stop_loss_price'] = new_sl

        action = row['Prediction']
        buy_confidence = shifted_probas[i, 2] if shifted_probas is not None and i < len(shifted_probas) else 0.0
        sell_confidence = shifted_probas[i, 0] if shifted_probas is not None and i < len(shifted_probas) else 0.0

        updated_positions = []
        for pos in positions:
            if current_price <= pos['stop_loss_price']:
                capital += pos['shares'] * current_price
            elif (take_profit_percent > 0.0 and
                  current_price >= pos['entry_price'] * (1 + take_profit_percent)):
                shares_to_sell = int(pos['shares'] * partial_sell_ratio)
                if shares_to_sell > 0:
                    capital += shares_to_sell * current_price
                    pos['shares'] -= shares_to_sell
                if pos['shares'] > 0:
                    updated_positions.append(pos)
            else:
                updated_positions.append(pos)

        positions = updated_positions

        if action == 2 and buy_confidence >= prob_threshold:
            funds_to_spend = capital * buy_confidence
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

        if action == 0 and sell_confidence >= prob_threshold:
            for pos in positions:
                shares_to_sell = int(pos['shares'] * partial_sell_ratio)
                if shares_to_sell > 0:
                    capital += shares_to_sell * current_price
                    pos['shares'] -= shares_to_sell
            positions = [p for p in positions if p['shares'] > 0]

    final_value = capital
    if len(positions) > 0:
        last_price = df.iloc[-1]['Close']
        for pos in positions:
            final_value += pos['shares'] * last_price

    final_return = (final_value - float(initial_capital)) / float(initial_capital) * 100.0
    final_return_str = f"{final_return:.2f}%"

    if not daily_values:
        return final_val, final_return_str, [], [], {
            'FinalValue': f"{final_value:.2f}",
            'PercentReturn': final_return_str,
            'SharpeRatio': "N/A",
            'MaxDrawdown': "N/A"
        }

    daily_returns = [(daily_values[i] - daily_values[i - 1]) / (daily_values[i - 1] + 1e-9)
                     for i in range(1, len(daily_values))]

    if len(daily_returns) > 1:
        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns, ddof=1)
        sharpe = (mean_ret / (std_ret + 1e-9)) * np.sqrt(252 * 390)  # Adjusted for intraday
    else:
        sharpe = 0.0

    running_max = -np.inf
    drawdowns = []
    for val in daily_values:
        if val > running_max:
            running_max = val
        drawdowns.append((val - running_max) / (running_max + 1e-9))

    max_drawdown = min(drawdowns) if drawdowns else 0.0
    max_drawdown_str = f"{max_drawdown * 100:.2f}%"

    metrics = {
        'FinalValue': f"{final_value:.2f}",
        'PercentReturn': final_return_str,
        'SharpeRatio': f"{sharpe:.3f}",
        'MaxDrawdown': max_drawdown_str
    }

    return final_value, final_return_str, daily_dates, daily_values, metrics


###############################################################################
# 5. FLASK ROUTES
###############################################################################

@app.route('/')
def index():
    html = """
    <h1>Welcome to the Advanced Algorithmic Trading App</h1>
    <ul>
      <li><a href="{{ url_for('train') }}">Train Models (SSE Progress)</a></li>
      <li><a href="{{ url_for('select_backtest_advanced') }}">Run Advanced Backtest (Single Ticker)</a></li>
      <li><a href="{{ url_for('select_backtest_portfolio') }}">Run Portfolio Backtest (Multiple Tickers)</a></li>
      <li><a href="{{ url_for('predict_next_day') }}">Predict Next Day O/H/L/C</a></li>
    </ul>
    """
    return render_template_string(html)


# ----------------------------------------------------------------------------
#  Training Page (SSE-based)
# ----------------------------------------------------------------------------
@app.route('/train', methods=['GET'])
def train():
    html = """
    <h1>Train Models for All Tickers</h1>
    <button onclick="startTraining()">Start Training</button>
    <div id="status"></div>
    <div style="width:300px; background:#ccc;">
      <div id="progressbar" style="width:0px; background:green; height:20px;"></div>
    </div>
    <p><a href="{{ url_for('index') }}">Back to Home</a></p>

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
            statusDiv.innerHTML = "Training in progress... " + current + "/" + total;
          } else if(status === 'done'){
            progressBar.style.width = "100%";
            statusDiv.innerHTML = "Training complete!";
            evtSource.close();
          }
        };
      }
    </script>
    """
    return render_template_string(html)


@app.route('/start_training', methods=['GET'])
def start_training():
    with progress_lock:
        if progress_data['status'] == 'training':
            return jsonify({"status": "already_training"})

        thread = threading.Thread(target=train_all_tickers_with_progress)
        thread.start()

    return jsonify({"status": "ok"})


@app.route('/train_progress')
def train_progress():
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


# ----------------------------------------------------------------------------
#  Advanced Backtest Selection Page
# ----------------------------------------------------------------------------
@app.route('/select_backtest_portfolio', methods=['GET', 'POST'])
def select_backtest_portfolio():
    """
    Displays a more user-friendly form for picking multiple tickers,
    plus all the other model/capital parameters.
    """
    tickers = sorted(list_tickers())
    model_names = ['LogisticRegression', 'RandomForest', 'MLP']

    if request.method == 'POST':
        # The hidden field "selected_tickers" holds the final comma-separated list
        selected_tickers_str = request.form.get('selected_tickers', '')
        selected_tickers = [t.strip() for t in selected_tickers_str.split(',') if t.strip()]

        model_name = request.form.get('model_name')
        initial_cap = float(request.form.get('initial_capital', '10000'))
        stop_loss_percent = float(request.form.get('stop_loss_percent', '0.05'))
        partial_sell_ratio = float(request.form.get('partial_sell_ratio', '0.5'))
        prob_threshold = float(request.form.get('prob_threshold', '0.6'))
        trailing_stop = (request.form.get('trailing_stop', 'off') == 'on')
        take_profit_percent = float(request.form.get('take_profit_percent', '0.2'))

        tickers_str = ",".join(selected_tickers)
        return redirect(url_for('backtest_portfolio',
                                tickers=tickers_str,
                                model_name=model_name,
                                initial_capital=initial_cap,
                                stop_loss_percent=stop_loss_percent,
                                partial_sell_ratio=partial_sell_ratio,
                                prob_threshold=prob_threshold,
                                trailing_stop='1' if trailing_stop else '0',
                                take_profit_percent=take_profit_percent))

    # GET => show the HTML form for multiple ticker selection
    html = """
    <h1>Portfolio Backtesting Setup</h1>
    <p>Type or search a ticker below and click "Add" to move it to your selected list. 
       Then "Remove" to remove from your list if needed.</p>

    <div>
      <label for="ticker_search">Search Ticker:</label>
      <input type="text" id="ticker_search" onkeyup="filterTickers()" placeholder="Type to filter...">
    </div>

    <br>
    <div style="display:flex; gap:20px;">
      <div>
        <h4>All Tickers</h4>
        <select id="all_tickers" size="10" style="min-width:150px;">
          {% for t in tickers %}
          <option value="{{t}}">{{t}}</option>
          {% endfor %}
        </select>
        <br>
        <button type="button" onclick="addTicker()">Add &raquo;</button>
      </div>

      <div>
        <h4>Selected Tickers</h4>
        <select id="selected_tickers_list" size="10" style="min-width:150px;">
        </select>
        <br>
        <button type="button" onclick="removeTicker()">&laquo; Remove</button>
      </div>
    </div>

    <br><br>
    <form method="POST" id="portfolioForm">
      <input type="hidden" name="selected_tickers" id="hidden_selected_tickers">

      <label>Model:</label>
      <select name="model_name">
        {% for m in model_names %}
        <option value="{{m}}">{{m}}</option>
        {% endfor %}
      </select>
      <br><br>

      <label>Initial Capital:</label>
      <input type="number" name="initial_capital" value="10000" step="100" />
      <br><br>

      <label>Stop-Loss % (e.g. 0.05=5%):</label>
      <input type="number" name="stop_loss_percent" value="0.05" step="0.01" />
      <br><br>

      <label>Partial Sell Ratio (e.g. 0.5=50%):</label>
      <input type="number" name="partial_sell_ratio" value="0.5" step="0.1" />
      <br><br>

      <label>Probability Threshold (0 to 1):</label>
      <input type="number" name="prob_threshold" value="0.6" step="0.05" />
      <br><br>

      <label>Trailing Stop?</label>
      <input type="checkbox" name="trailing_stop" />
      <br><br>

      <label>Take Profit % (e.g. 0.2=20%):</label>
      <input type="number" name="take_profit_percent" value="0.2" step="0.05" />
      <br><br>

      <button type="submit" onclick="prepareSelectedTickers()">Run Portfolio Backtest</button>
    </form>

    <p><a href="{{ url_for('index') }}">Back to Home</a></p>

    <script>
      function addTicker(){
        const allList = document.getElementById('all_tickers');
        const selectedList = document.getElementById('selected_tickers_list');
        if(allList.selectedIndex >= 0){
          let opt = allList.options[allList.selectedIndex];
          // create new <option> for selectedList
          let newOpt = document.createElement('option');
          newOpt.value = opt.value;
          newOpt.text = opt.text;
          selectedList.add(newOpt);
        }
      }

      function removeTicker(){
        const selectedList = document.getElementById('selected_tickers_list');
        if(selectedList.selectedIndex >= 0){
          selectedList.remove(selectedList.selectedIndex);
        }
      }

      function prepareSelectedTickers(){
        const selectedList = document.getElementById('selected_tickers_list');
        const hiddenField = document.getElementById('hidden_selected_tickers');
        let values = [];
        for(let i=0; i<selectedList.options.length; i++){
          values.push(selectedList.options[i].value);
        }
        hiddenField.value = values.join(',');
      }

      function filterTickers(){
        let input = document.getElementById('ticker_search');
        let filter = input.value.toUpperCase();
        let allList = document.getElementById('all_tickers');
        for(let i=0; i<allList.options.length; i++){
          let txt = allList.options[i].text.toUpperCase();
          if(txt.indexOf(filter) > -1){
            allList.options[i].style.display = "";
          } else {
            allList.options[i].style.display = "none";
          }
        }
      }
    </script>
    """
    return render_template_string(html, tickers=tickers, model_names=model_names)

@app.route('/backtest_portfolio')
def backtest_portfolio():
    """
    Executes a portfolio-level backtest given multiple tickers in the URL (comma-separated).
    Also shows a re-run form and a 'Back to Portfolio Setup' link.
    """
    tickers_str = request.args.get('tickers', '')
    model_name = request.args.get('model_name')
    initial_capital = float(request.args.get('initial_capital', '10000'))
    stop_loss_percent = float(request.args.get('stop_loss_percent', '0.05'))
    partial_sell_ratio = float(request.args.get('partial_sell_ratio', '0.5'))
    prob_threshold = float(request.args.get('prob_threshold', '0.6'))
    trailing_stop = (request.args.get('trailing_stop', '0') == '1')
    take_profit_percent = float(request.args.get('take_profit_percent', '0.2'))

    if tickers_str.strip():
        selected_tickers = [t.strip() for t in tickers_str.split(',') if t.strip()]
    else:
        return "<p>Error: No tickers provided.</p>"

    # Reuse your 'advanced_backtest_portfolio' logic or however you're combining results
    (final_val, final_ret_str, daily_dates, daily_values, metrics) = advanced_backtest_portfolio(
        selected_tickers, model_name,
        initial_capital=initial_capital,
        stop_loss_percent=stop_loss_percent,
        partial_sell_ratio=partial_sell_ratio,
        prob_threshold=prob_threshold,
        trailing_stop=trailing_stop,
        take_profit_percent=take_profit_percent
    )

    if final_val is None:
        return f"<p>Error: {final_ret_str}</p>"

    # Plot the combined daily_values
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(daily_dates, daily_values, label='Portfolio Value')
    ax.set_title(f"Portfolio Backtest: {', '.join(selected_tickers)} - {model_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend()
    fig.tight_layout()

    pngImage = io.BytesIO()
    fig.savefig(pngImage, format='png')
    pngImage.seek(0)
    encoded = base64.b64encode(pngImage.getvalue()).decode('ascii')
    plt.close(fig)

    img_html = f'<img src="data:image/png;base64,{encoded}" alt="Portfolio Chart"/>'

    result_html = f"""
    <p>Final Capital: {metrics['FinalValue']}</p>
    <p>Percent Return: {metrics['PercentReturn']}</p>
    <p>Sharpe Ratio: {metrics['SharpeRatio']}</p>
    <p>Max Drawdown: {metrics['MaxDrawdown']}</p>
    """

    # Build a re-run form just like single ticker,
    # but including the multi-ticker info and all relevant fields
    trailing_stop_checked = 'checked' if trailing_stop else ''
    re_run_form = f"""
    <hr>
    <h2>Refine Your Portfolio Backtest</h2>
    <form method="GET" action="{ url_for('backtest_portfolio') }">
      <input type="hidden" name="tickers" value="{tickers_str}" />

      <label>Model:</label>
      <select name="model_name">
        <option value="LogisticRegression" {"selected" if model_name == "LogisticRegression" else ""}>LogisticRegression</option>
        <option value="RandomForest" {"selected" if model_name == "RandomForest" else ""}>RandomForest</option>
        <option value="MLP" {"selected" if model_name == "MLP" else ""}>MLP</option>
      </select>
      <br><br>

      <label>Initial Capital:</label>
      <input type="number" name="initial_capital" value="{initial_capital}" step="100" />
      <br><br>

      <label>Stop-Loss % (e.g., 0.05=5%):</label>
      <input type="number" name="stop_loss_percent" value="{stop_loss_percent}" step="0.01" />
      <br><br>

      <label>Partial Sell Ratio (e.g., 0.5=50%):</label>
      <input type="number" name="partial_sell_ratio" value="{partial_sell_ratio}" step="0.1" />
      <br><br>

      <label>Probability Threshold (0 to 1):</label>
      <input type="number" name="prob_threshold" value="{prob_threshold}" step="0.05" />
      <br><br>

      <label>Trailing Stop?</label>
      <input type="checkbox" name="trailing_stop" {trailing_stop_checked} />
      <br><br>

      <label>Take Profit % (e.g., 0.2=20%):</label>
      <input type="number" name="take_profit_percent" value="{take_profit_percent}" step="0.05" />
      <br><br>

      <button type="submit">Re-Run Portfolio Backtest</button>
    </form>
    """

    page_html = f"""
    <h1>Portfolio Backtest Results</h1>
    <h2>Tickers: {', '.join(selected_tickers)} - {model_name}</h2>
    {result_html}
    {img_html}

    {re_run_form}

    <hr>
    <!-- Fixing the link: now it references 'select_backtest_portfolio' properly -->
    <p><a href="{{{{ url_for('select_backtest_portfolio') }}}}">Back to Portfolio Setup</a></p>
    """
    return render_template_string(page_html)

@app.route('/select_backtest_advanced', methods=['GET', 'POST'])
def select_backtest_advanced():
    tickers = sorted(list_tickers())
    model_names = ['LogisticRegression', 'RandomForest', 'MLP']

    if request.method == 'POST':
        ticker = request.form.get('ticker')
        model_name = request.form.get('model_name')
        initial_cap = float(request.form.get('initial_capital', '10000'))
        stop_loss_percent = float(request.form.get('stop_loss_percent', '0.05'))
        partial_sell_ratio = float(request.form.get('partial_sell_ratio', '0.5'))
        prob_threshold = float(request.form.get('prob_threshold', '0.6'))
        trailing_stop = (request.form.get('trailing_stop', 'off') == 'on')
        take_profit_percent = float(request.form.get('take_profit_percent', '0.2'))

        return redirect(url_for('backtest_advanced',
                                ticker=ticker,
                                model_name=model_name,
                                initial_capital=initial_cap,
                                stop_loss_percent=stop_loss_percent,
                                partial_sell_ratio=partial_sell_ratio,
                                prob_threshold=prob_threshold,
                                trailing_stop='1' if trailing_stop else '0',
                                take_profit_percent=take_profit_percent))

    html = """
    <h1>Advanced Backtesting Setup</h1>
    <form method="POST">
      <label>Ticker:</label>
      <select name="ticker">
        {% for t in tickers %}
        <option value="{{t}}">{{t}}</option>
        {% endfor %}
      </select>
      <br><br>

      <label>Model:</label>
      <select name="model_name">
        {% for m in model_names %}
        <option value="{{m}}">{{m}}</option>
        {% endfor %}
      </select>
      <br><br>

      <label>Initial Capital:</label>
      <input type="number" name="initial_capital" value="10000" step="100" />
      <br><br>

      <label>Stop-Loss % (e.g. 0.05=5%):</label>
      <input type="number" name="stop_loss_percent" value="0.05" step="0.01" />
      <br><br>

      <label>Partial Sell Ratio (e.g. 0.5=50%):</label>
      <input type="number" name="partial_sell_ratio" value="0.5" step="0.1" />
      <br><br>

      <label>Probability Threshold (0~1):</label>
      <input type="number" name="prob_threshold" value="0.6" step="0.05" />
      <br><br>

      <label>Trailing Stop?</label>
      <input type="checkbox" name="trailing_stop" />
      <br><br>

      <label>Take Profit % (e.g. 0.2=20%):</label>
      <input type="number" name="take_profit_percent" value="0.2" step="0.05" />
      <br><br>

      <button type="submit">Run Advanced Backtest</button>
    </form>

    <p><a href="{{ url_for('index') }}">Back to Home</a></p>
    """
    return render_template_string(html, tickers=tickers, model_names=model_names)


# ----------------------------------------------------------------------------
#  The Advanced Backtest Route
# ----------------------------------------------------------------------------
@app.route('/backtest_advanced')
def backtest_advanced():
    ticker = request.args.get('ticker')
    model_name = request.args.get('model_name')
    initial_capital = float(request.args.get('initial_capital', '10000'))
    stop_loss_percent = float(request.args.get('stop_loss_percent', '0.05'))
    partial_sell_ratio = float(request.args.get('partial_sell_ratio', '0.5'))
    prob_threshold = float(request.args.get('prob_threshold', '0.6'))
    trailing_stop = (request.args.get('trailing_stop', '0') == '1')
    take_profit_percent = float(request.args.get('take_profit_percent', '0.2'))

    final_val, final_ret_str, daily_dates, daily_values, metrics = advanced_backtest(
        ticker, model_name,
        initial_capital=initial_capital,
        stop_loss_percent=stop_loss_percent,
        partial_sell_ratio=partial_sell_ratio,
        prob_threshold=prob_threshold,
        trailing_stop=trailing_stop,
        take_profit_percent=take_profit_percent
    )

    if final_val is None:
        result_html = f"<p>Error: {final_ret_str}</p>"
        img_html = ""
    else:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(daily_dates, daily_values, label='Portfolio Value')
        ax.set_title(f"Advanced Backtest: {ticker} - {model_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value ($)")
        ax.legend()
        fig.tight_layout()

        pngImage = io.BytesIO()
        fig.savefig(pngImage, format='png')
        pngImage.seek(0)
        encoded = base64.b64encode(pngImage.getvalue()).decode('ascii')
        plt.close(fig)

        img_html = f'<img src="data:image/png;base64,{encoded}" alt="Backtest Chart"/>'

        result_html = f"""
        <p>Final Capital: {metrics['FinalValue']}</p>
        <p>Percent Return: {metrics['PercentReturn']}</p>
        <p>Sharpe Ratio: {metrics['SharpeRatio']}</p>
        <p>Max Drawdown: {metrics['MaxDrawdown']}</p>
        """

    tickers = sorted(list_tickers())
    model_names_list = ['LogisticRegression', 'RandomForest', 'MLP']

    re_run_form = f"""
    <hr>
    <h2>Refine Your Backtest</h2>
    <form method="GET" action="{url_for('backtest_advanced')}">
      <label for="ticker">Ticker:</label>
      <select name="ticker">
        {"".join(
        f'<option value="{t}" {"selected" if t == ticker else ""}>{t}</option>'
        for t in tickers
    )}
      </select>
      <br><br>

      <label for="model_name">Model:</label>
      <select name="model_name">
        {"".join(
        f'<option value="{m}" {"selected" if m == model_name else ""}>{m}</option>'
        for m in model_names_list
    )}
      </select>
      <br><br>

      <label>Initial Capital:</label>
      <input type="number" name="initial_capital" value="{initial_capital}" step="100" />
      <br><br>

      <label>Stop-Loss % (e.g., 0.05 = 5%):</label>
      <input type="number" name="stop_loss_percent" value="{stop_loss_percent}" step="0.01" />
      <br><br>

      <label>Partial Sell Ratio (e.g., 0.5 = 50%):</label>
      <input type="number" name="partial_sell_ratio" value="{partial_sell_ratio}" step="0.1" />
      <br><br>

      <label>Probability Threshold (0 to 1):</label>
      <input type="number" name="prob_threshold" value="{prob_threshold}" step="0.05" />
      <br><br>

      <label>Trailing Stop?</label>
      <input type="checkbox" name="trailing_stop" {"checked" if trailing_stop else ""} />
      <br><br>

      <label>Take Profit % (e.g., 0.2 = 20%):</label>
      <input type="number" name="take_profit_percent" value="{take_profit_percent}" step="0.05" />
      <br><br>

      <button type="submit">Re-Run Backtest</button>
    </form>
    """

    page_html = f"""
    <h1>Advanced Backtest Results</h1>
    <h2>{ticker} - {model_name}</h2>
    {result_html}
    {img_html}

    {re_run_form}

    <hr>
    <p><a href="{{{{ url_for('select_backtest_advanced') }}}}">Go to Full Advanced Setup Page</a></p>
    """
    return render_template_string(page_html)


# ----------------------------------------------------------------------------
# 6. NEW ROUTE: PREDICT NEXT-DAY O/H/L/C
# ----------------------------------------------------------------------------
@app.route('/predict_next_day', methods=['GET', 'POST'])
def predict_next_day():
    """
    Lets user pick a ticker, then predicts tomorrow's OPEN/HIGH/LOW/CLOSE
    using the newly added regression models. Suggest a simple BUY/HOLD/SELL
    based on predicted close vs current close.
    """
    tickers = sorted(list_tickers())

    if request.method == 'POST':
        ticker = request.form.get('ticker')
        if ticker not in MODELS:
            return render_template_string("<p>Error: No model for this ticker. Please train first.</p>")

        # Load & compute indicators
        df = load_data_for_ticker(ticker)
        df = compute_indicators(df)
        if df.empty:
            return render_template_string("<p>Error: No data available.</p>")

        last_row = df.iloc[[-1]].copy()
        last_row.dropna(inplace=True)
        if last_row.empty:
            return render_template_string("<p>Error: Not enough data to predict.</p>")

        feature_cols = [
            'Close', 'High', 'Low', 'Open', 'Volume',
            'MA_10', 'MA_50', 'MA_200',
            'Daily_Return', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist'
        ]
        X_last = last_row[feature_cols].values

        # Scale using the same scaler
        scaler = SCALERS[ticker]
        X_last_scaled = scaler.transform(X_last)

        all_models = MODELS[ticker]
        # Retrieve the 4 regression models for next day
        open_reg = all_models['NextOpenReg']
        high_reg = all_models['NextHighReg']
        low_reg = all_models['NextLowReg']
        close_reg = all_models['NextCloseReg']

        pred_open = open_reg.predict(X_last_scaled)[0]
        pred_high = high_reg.predict(X_last_scaled)[0]
        pred_low = low_reg.predict(X_last_scaled)[0]
        pred_close = close_reg.predict(X_last_scaled)[0]

        current_close = last_row['Close'].values[0]
        pct_diff = (pred_close - current_close) / current_close * 100.0

        if pct_diff > 2.0:
            suggestion = "BUY (Predicted close is 2%+ above current)"
        elif pct_diff < -2.0:
            suggestion = "SELL (Predicted close is 2%+ below current)"
        else:
            suggestion = "HOLD (Predicted move is within ±2%)"

        result_html = f"""
        <h3>Predictions for Ticker: {ticker}</h3>
        <p>Tomorrow's Predicted OPEN:  {pred_open:.2f}</p>
        <p>Tomorrow's Predicted HIGH:  {pred_high:.2f}</p>
        <p>Tomorrow's Predicted LOW:   {pred_low:.2f}</p>
        <p>Tomorrow's Predicted CLOSE: {pred_close:.2f}</p>
        <p>Current Close: {current_close:.2f}</p>
        <p>Predicted % Diff vs Current Close: {pct_diff:.2f}%</p>
        <h4>Suggested Action: {suggestion}</h4>
        <p><a href='{url_for('predict_next_day')}'>Back to Predict Page</a></p>
        """
        return render_template_string(result_html)

    # If GET => show minimal form
    html = """
    <h1>Predict Next Day O/H/L/C</h1>
    <form method="POST">
      <label>Ticker:</label>
      <select name="ticker">
        {% for t in tickers %}
        <option value="{{t}}">{{t}}</option>
        {% endfor %}
      </select>
      <br><br>
      <button type="submit">Predict Next Day</button>
    </form>
    <p><a href="{{ url_for('index') }}">Back to Home</a></p>
    """
    return render_template_string(html, tickers=tickers)


###############################################################################
# MAIN
###############################################################################
if __name__ == '__main__':
    load_models()
    app.run(debug=True)