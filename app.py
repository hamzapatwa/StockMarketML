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

    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df.set_index('Date', inplace=True, drop=True)
    return df


def compute_indicators(df):
    df = df.copy()
    df.sort_index(inplace=True)

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

    class_counts = df['Action'].value_counts()
    print(f"Class distribution for threshold {threshold}:")
    print(class_counts)
    return df


###############################################################################
# 2. TRAINING MODELS
###############################################################################
def train_models_for_ticker(ticker, df):
    feature_cols = [
        'Close', 'High', 'Low', 'Open', 'Volume',
        'MA_10', 'MA_50', 'MA_200',
        'Daily_Return', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist'
    ]
    X_class = df[feature_cols].values
    y_class = df['Action'].values

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

    df_reg = df.copy()
    df_reg['Next_Open'] = df_reg['Open'].shift(-1)
    df_reg['Next_High'] = df_reg['High'].shift(-1)
    df_reg['Next_Low'] = df_reg['Low'].shift(-1)
    df_reg['Next_Close'] = df_reg['Close'].shift(-1)
    df_reg.dropna(inplace=True)

    X_reg = df_reg[feature_cols].values
    y_open = df_reg['Next_Open'].values
    y_high = df_reg['Next_High'].values
    y_low = df_reg['Next_Low'].values
    y_close = df_reg['Next_Close'].values

    X_train_r, X_test_r = X_reg[:len(X_train_c)], X_reg[len(X_train_c):]
    X_train_r_scaled = scaler.transform(X_train_r)

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

    all_models = {**classification_models, **regression_models}
    return all_models, scaler


def train_all_tickers_with_progress():
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

        time.sleep(0.5)

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

def advanced_backtest(ticker, model_name,
                      initial_capital=10000,
                      stop_loss_percent=0.05,
                      partial_sell_ratio=0.5,
                      prob_threshold=0.6,
                      trailing_stop=True,
                      take_profit_percent=0.2):
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
        sharpe = (mean_ret / (std_ret + 1e-9)) * np.sqrt(252 * 390)
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


def advanced_backtest_portfolio(tickers, model_name,
                                initial_capital=10000,
                                stop_loss_percent=0.05,
                                partial_sell_ratio=0.5,
                                prob_threshold=0.6,
                                trailing_stop=True,
                                take_profit_percent=0.2):
    if not tickers:
        return None, "No tickers selected!", [], [], {}

    n = len(tickers)
    capital_per_ticker = initial_capital / n
    ticker_values_by_date = {}

    for t in tickers:
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
            print(f"Skipping {t} due to error: {final_ret_str}")
            continue

        df_vals = pd.DataFrame({
            'Date': daily_dates,
            'Value': daily_vals
        }).set_index('Date')
        ticker_values_by_date[t] = df_vals

    if not ticker_values_by_date:
        return None, "No valid tickers after backtest", [], [], {}

    combined_df = None
    for t, df_vals in ticker_values_by_date.items():
        if combined_df is None:
            combined_df = df_vals.rename(columns={'Value': t})
        else:
            combined_df = combined_df.join(df_vals.rename(columns={'Value': t}), how='outer')

    combined_df.sort_index(inplace=True)
    combined_df.fillna(method='ffill', inplace=True)
    combined_df.fillna(method='bfill', inplace=True)

    combined_df['PortfolioValue'] = combined_df.sum(axis=1)

    daily_values = combined_df['PortfolioValue'].tolist()
    daily_dates = combined_df.index.tolist()

    final_value = daily_values[-1] if daily_values else initial_capital
    final_return = (final_value - initial_capital) / initial_capital * 100.0
    final_return_str = f"{final_return:.2f}%"

    if len(daily_values) <= 1:
        metrics = {
            'FinalValue': f"{final_value:.2f}",
            'PercentReturn': final_return_str,
            'SharpeRatio': "N/A",
            'MaxDrawdown': "N/A"
        }
        return final_value, final_return_str, daily_dates, daily_values, metrics

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


###############################################################################
# 5. FLASK ROUTES WITH BOOTSTRAP
###############################################################################
@app.route('/train')
def train():
    # Our SSE training page, now with Bootstrap styling
    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Train Models</title>
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
      <style>
        .container { max-width: 800px; margin-top: 40px; }
      </style>
    </head>
    <body class="bg-light">
    <div class="container">
      <h1 class="mt-4 mb-3">Train Models for All Tickers</h1>
      <p>Click the button below to start training models.</p>
      <button class="btn btn-primary" onclick="startTraining()">Start Training</button>

      <div id="status" class="mt-3"></div>
      <div class="progress" style="width:300px; background:#ccc; margin-top:15px;">
        <div id="progressbar" class="progress-bar bg-success" role="progressbar" style="width:0%;">
        </div>
      </div>
      <p class="mt-3"><a href="{{ url_for('index') }}" class="btn btn-secondary">Back to Home</a></p>
    </div>

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
            progressBar.innerHTML = Math.floor(pct) + "%";
            statusDiv.innerHTML = "Training in progress... " + current + "/" + total;
          } else if(status === 'done'){
            progressBar.style.width = "100%";
            progressBar.innerHTML = "100%";
            statusDiv.innerHTML = "Training complete!";
            evtSource.close();
          }
        };
      }
    </script>
    </body>
    </html>
    """
    return render_template_string(html)


@app.route('/start_training')
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


@app.route('/select_backtest_portfolio', methods=['GET', 'POST'])
def select_backtest_portfolio():
    """
    Displays a more user-friendly form for picking multiple tickers,
    plus model/capital parameters.
    """
    tickers = sorted(list_tickers())
    model_names = ['LogisticRegression', 'RandomForest', 'MLP']

    if request.method == 'POST':
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

    # GET => show a more user-friendly form
    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Portfolio Backtesting Setup</title>
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
      <style>
        .container { max-width: 900px; margin-top: 40px; }
        .ticker-box { min-width: 180px; }
      </style>
    </head>
    <body class="bg-light">
    <div class="container">
      <h1 class="mb-4">Portfolio Backtesting Setup</h1>
      <p>Type or search a ticker below and click "Add" to move it to your selected list. 
         Then "Remove" to remove from your list if needed.</p>

      <div class="row mb-3">
        <div class="col-12 col-md-6 mb-2">
          <label for="ticker_search" class="form-label">Search Ticker:</label>
          <input type="text" id="ticker_search" onkeyup="filterTickers()" 
                 class="form-control" placeholder="Type to filter...">
        </div>
      </div>

      <div class="row">
        <div class="col-12 col-md-5">
          <h5>All Tickers</h5>
          <select id="all_tickers" size="10" class="form-select ticker-box">
            {% for t in tickers %}
            <option value="{{t}}">{{t}}</option>
            {% endfor %}
          </select>
          <br>
          <button type="button" class="btn btn-sm btn-primary" onclick="addTicker()">Add &raquo;</button>
        </div>

        <div class="col-12 col-md-5">
          <h5>Selected Tickers</h5>
          <select id="selected_tickers_list" size="10" class="form-select ticker-box">
          </select>
          <br>
          <button type="button" class="btn btn-sm btn-danger" onclick="removeTicker()">&laquo; Remove</button>
        </div>
      </div>

      <hr class="my-4">

      <form method="POST" id="portfolioForm" class="row g-3">
        <input type="hidden" name="selected_tickers" id="hidden_selected_tickers">

        <div class="col-md-4">
          <label class="form-label">Model:</label>
          <select name="model_name" class="form-select">
            {% for m in model_names %}
            <option value="{{m}}">{{m}}</option>
            {% endfor %}
          </select>
        </div>

        <div class="col-md-4">
          <label class="form-label">Initial Capital:</label>
          <input type="number" name="initial_capital" value="10000" step="100" class="form-control" />
        </div>

        <div class="col-md-4">
          <label class="form-label">Stop-Loss %:</label>
          <input type="number" name="stop_loss_percent" value="0.05" step="0.01" class="form-control" />
        </div>

        <div class="col-md-4">
          <label class="form-label">Partial Sell Ratio:</label>
          <input type="number" name="partial_sell_ratio" value="0.5" step="0.1" class="form-control" />
        </div>

        <div class="col-md-4">
          <label class="form-label">Probability Threshold (0~1):</label>
          <input type="number" name="prob_threshold" value="0.6" step="0.05" class="form-control" />
        </div>

        <div class="col-md-4">
          <div class="form-check mt-4">
            <input type="checkbox" name="trailing_stop" class="form-check-input" id="trailingStopCheck" />
            <label for="trailingStopCheck" class="form-check-label">Trailing Stop?</label>
          </div>
        </div>

        <div class="col-md-4">
          <label class="form-label">Take Profit %:</label>
          <input type="number" name="take_profit_percent" value="0.2" step="0.05" class="form-control" />
        </div>

        <div class="col-12">
          <button type="submit" onclick="prepareSelectedTickers()" class="btn btn-success">Run Portfolio Backtest</button>
          <a href="{{ url_for('index') }}" class="btn btn-secondary">Back to Home</a>
        </div>
      </form>
    </div>

    <script>
      function addTicker(){
        const allList = document.getElementById('all_tickers');
        const selectedList = document.getElementById('selected_tickers_list');
        if(allList.selectedIndex >= 0){
          let opt = allList.options[allList.selectedIndex];
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
    </body>
    </html>
    """
    return render_template_string(html, tickers=tickers, model_names=model_names)


@app.route('/backtest_portfolio')
def backtest_portfolio():
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

    img_html = f'<img src="data:image/png;base64,{encoded}" class="img-fluid" alt="Portfolio Chart"/>'

    result_html = f"""
    <p><strong>Final Capital:</strong> {metrics['FinalValue']}</p>
    <p><strong>Percent Return:</strong> {metrics['PercentReturn']}</p>
    <p><strong>Sharpe Ratio:</strong> {metrics['SharpeRatio']}</p>
    <p><strong>Max Drawdown:</strong> {metrics['MaxDrawdown']}</p>
    """

    trailing_stop_checked = 'checked' if trailing_stop else ''
    re_run_form = f"""
    <hr>
    <h3>Refine Your Portfolio Backtest</h3>
    <form method="GET" action="{url_for('backtest_portfolio')}" class="row g-3 mt-2">
      <input type="hidden" name="tickers" value="{tickers_str}" />

      <div class="col-md-4">
        <label class="form-label">Model:</label>
        <select name="model_name" class="form-select">
          <option value="LogisticRegression" {"selected" if model_name == "LogisticRegression" else ""}>LogisticRegression</option>
          <option value="RandomForest" {"selected" if model_name == "RandomForest" else ""}>RandomForest</option>
          <option value="MLP" {"selected" if model_name == "MLP" else ""}>MLP</option>
        </select>
      </div>

      <div class="col-md-4">
        <label class="form-label">Initial Capital:</label>
        <input type="number" name="initial_capital" value="{initial_capital}" step="100" class="form-control"/>
      </div>

      <div class="col-md-4">
        <label class="form-label">Stop-Loss %:</label>
        <input type="number" name="stop_loss_percent" value="{stop_loss_percent}" step="0.01" class="form-control"/>
      </div>

      <div class="col-md-4">
        <label class="form-label">Partial Sell Ratio:</label>
        <input type="number" name="partial_sell_ratio" value="{partial_sell_ratio}" step="0.1" class="form-control"/>
      </div>

      <div class="col-md-4">
        <label class="form-label">Probability Threshold:</label>
        <input type="number" name="prob_threshold" value="{prob_threshold}" step="0.05" class="form-control"/>
      </div>

      <div class="col-md-4 d-flex align-items-end">
        <div class="form-check">
          <input class="form-check-input" type="checkbox" name="trailing_stop" {trailing_stop_checked} id="trailingStopCheck2">
          <label class="form-check-label" for="trailingStopCheck2">Trailing Stop?</label>
        </div>
      </div>

      <div class="col-md-4">
        <label class="form-label">Take Profit %:</label>
        <input type="number" name="take_profit_percent" value="{take_profit_percent}" step="0.05" class="form-control"/>
      </div>

      <div class="col-12">
        <button type="submit" class="btn btn-primary">Re-Run Portfolio Backtest</button>
      </div>
    </form>
    """

    page_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <title>Portfolio Backtest Results</title>
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
      <style>
        .container {{ max-width: 900px; margin-top: 40px; }}
      </style>
    </head>
    <body class="bg-light">
    <div class="container">
      <h1>Portfolio Backtest Results</h1>
      <h5>Tickers: {', '.join(selected_tickers)} - {model_name}</h5>
      <div class="mt-3">
        {result_html}
      </div>
      <div class="mt-4">
        {img_html}
      </div>

      <div class="mt-5">
        {re_run_form}
      </div>

      <hr class="my-4">
      <p><a href="{{{{ url_for('select_backtest_portfolio') }}}}" class="btn btn-secondary">Back to Portfolio Setup</a></p>
    </div>
    </body>
    </html>
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
    <!DOCTYPE html>
    <html>
    <head>
      <title>Advanced Backtesting Setup</title>
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
      <style>
        .container { max-width: 700px; margin-top: 40px; }
      </style>
    </head>
    <body class="bg-light">
    <div class="container">
      <h1>Advanced Backtesting Setup</h1>
      <form method="POST" class="row g-3">
        <div class="col-md-6">
          <label class="form-label">Ticker:</label>
          <select name="ticker" class="form-select">
            {% for t in tickers %}
            <option value="{{t}}">{{t}}</option>
            {% endfor %}
          </select>
        </div>
        <div class="col-md-6">
          <label class="form-label">Model:</label>
          <select name="model_name" class="form-select">
            {% for m in model_names %}
            <option value="{{m}}">{{m}}</option>
            {% endfor %}
          </select>
        </div>

        <div class="col-md-4">
          <label class="form-label">Initial Capital:</label>
          <input type="number" name="initial_capital" value="10000" step="100" class="form-control"/>
        </div>
        <div class="col-md-4">
          <label class="form-label">Stop-Loss % (e.g. 0.05=5%):</label>
          <input type="number" name="stop_loss_percent" value="0.05" step="0.01" class="form-control"/>
        </div>
        <div class="col-md-4">
          <label class="form-label">Partial Sell Ratio (e.g. 0.5=50%):</label>
          <input type="number" name="partial_sell_ratio" value="0.5" step="0.1" class="form-control"/>
        </div>
        <div class="col-md-4">
          <label class="form-label">Probability Threshold (0~1):</label>
          <input type="number" name="prob_threshold" value="0.6" step="0.05" class="form-control"/>
        </div>
        <div class="col-md-4">
          <label class="form-label">Take Profit % (e.g. 0.2=20%):</label>
          <input type="number" name="take_profit_percent" value="0.2" step="0.05" class="form-control"/>
        </div>
        <div class="col-md-4 d-flex align-items-end">
          <div class="form-check">
            <input type="checkbox" name="trailing_stop" class="form-check-input" id="trailingStopCheckSingle"/>
            <label for="trailingStopCheckSingle" class="form-check-label">Trailing Stop?</label>
          </div>
        </div>

        <div class="col-12">
          <button type="submit" class="btn btn-primary">Run Advanced Backtest</button>
          <a href="{{ url_for('index') }}" class="btn btn-secondary">Back to Home</a>
        </div>
      </form>
    </div>
    </body>
    </html>
    """
    return render_template_string(html, tickers=tickers, model_names=model_names)


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

        img_html = f'<img src="data:image/png;base64,{encoded}" class="img-fluid" alt="Backtest Chart"/>'

        result_html = f"""
        <p><strong>Final Capital:</strong> {metrics['FinalValue']}</p>
        <p><strong>Percent Return:</strong> {metrics['PercentReturn']}</p>
        <p><strong>Sharpe Ratio:</strong> {metrics['SharpeRatio']}</p>
        <p><strong>Max Drawdown:</strong> {metrics['MaxDrawdown']}</p>
        """

    # Build the re-run form just as before
    tickers = sorted(list_tickers())
    model_names_list = ['LogisticRegression', 'RandomForest', 'MLP']
    trailing_stop_checked = 'checked' if trailing_stop else ''

    re_run_form = f"""
    <hr>
    <h3>Refine Your Backtest</h3>
    <form method="GET" action="{url_for('backtest_advanced')}" class="row g-3 mt-2">
      <div class="col-md-6">
        <label for="ticker" class="form-label">Ticker:</label>
        <select name="ticker" class="form-select">
          {"".join(
        f'<option value="{t}" {"selected" if t == ticker else ""}>{t}</option>'
        for t in tickers
    )}
        </select>
      </div>

      <div class="col-md-6">
        <label for="model_name" class="form-label">Model:</label>
        <select name="model_name" class="form-select">
          {"".join(
        f'<option value="{m}" {"selected" if m == model_name else ""}>{m}</option>'
        for m in model_names_list
    )}
        </select>
      </div>

      <div class="col-md-4">
        <label class="form-label">Initial Capital:</label>
        <input type="number" name="initial_capital" value="{initial_capital}" step="100" class="form-control"/>
      </div>
      <div class="col-md-4">
        <label class="form-label">Stop-Loss %:</label>
        <input type="number" name="stop_loss_percent" value="{stop_loss_percent}" step="0.01" class="form-control"/>
      </div>
      <div class="col-md-4">
        <label class="form-label">Partial Sell Ratio:</label>
        <input type="number" name="partial_sell_ratio" value="{partial_sell_ratio}" step="0.1" class="form-control"/>
      </div>
      <div class="col-md-4">
        <label class="form-label">Probability Threshold:</label>
        <input type="number" name="prob_threshold" value="{prob_threshold}" step="0.05" class="form-control"/>
      </div>
      <div class="col-md-4">
        <label class="form-label">Take Profit %:</label>
        <input type="number" name="take_profit_percent" value="{take_profit_percent}" step="0.05" class="form-control"/>
      </div>
      <div class="col-md-4 d-flex align-items-end">
        <div class="form-check">
          <input class="form-check-input" type="checkbox" name="trailing_stop" {trailing_stop_checked} id="trailingStopCheckSingle2">
          <label class="form-check-label" for="trailingStopCheckSingle2">Trailing Stop?</label>
        </div>
      </div>

      <div class="col-12">
        <button type="submit" class="btn btn-primary">Re-Run Backtest</button>
      </div>
    </form>
    """

    page_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <title>Advanced Backtest Results</title>
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
      <style>
        .container {{ max-width: 900px; margin-top: 40px; }}
      </style>
    </head>
    <body class="bg-light">
    <div class="container">
      <h1>Advanced Backtest Results</h1>
      <h5>{ticker} - {model_name}</h5>
      <div class="mt-3">
        {result_html}
      </div>
      <div class="mt-4">
        {img_html}
      </div>

      <div class="mt-5">
        {re_run_form}
      </div>

      <hr class="my-4">
      <p><a href="{{{{ url_for('select_backtest_advanced') }}}}" class="btn btn-secondary">Go to Full Advanced Setup Page</a></p>
    </div>
    </body>
    </html>
    """
    return render_template_string(page_html)


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
@app.route('/')
def index():
    # The main index with bootstrap styling
    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Algorithmic Trading App</title>
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
      <style>
        .container {
          max-width: 800px; 
          margin-top: 40px;
        }
      </style>
    </head>
    <body class="bg-light">
      <div class="container">
        <h1 class="mb-4">Welcome to the Advanced Algorithmic Trading App</h1>
        <ul class="list-group">
          <li class="list-group-item">
            <a href="{{ url_for('train') }}" class="btn btn-link">Train Models (SSE Progress)</a>
          </li>
          <li class="list-group-item">
            <a href="{{ url_for('select_backtest_advanced') }}" class="btn btn-link">Run Advanced Backtest (Single Ticker)</a>
          </li>
          <li class="list-group-item">
            <a href="{{ url_for('select_backtest_portfolio') }}" class="btn btn-link">Run Portfolio Backtest (Multiple Tickers)</a>
          </li>
          <li class="list-group-item">
            <a href="{{ url_for('predict_next_day') }}" class="btn btn-link">Predict Next Day O/H/L/C</a>
          </li>
        </ul>
      </div>
    </body>
    </html>
    """
    return render_template_string(html)


if __name__ == '__main__':
    load_models()
    app.run(debug=True)