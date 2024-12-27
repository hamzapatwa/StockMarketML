import os
import glob
import pickle
import numpy as np
import pandas as pd

import time
import threading
from flask import Flask, request, render_template_string, redirect, url_for, Response, jsonify
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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

    # Adjust skiprows if your CSV structure differs
    df = pd.read_csv(file_path, skiprows=[1, 2])
    df.rename(columns={
        "Price": "Date",
        "Close": "Close",
        "High":  "High",
        "Low":   "Low",
        "Open":  "Open",
        "Volume":"Volume"
    }, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
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

def create_labels(df, threshold=0.01):
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
    X = df[feature_cols].values
    y = df['Action'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(multi_class='multinomial', max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train_scaled, y_train)

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000)
    mlp.fit(X_train_scaled, y_train)

    models = {
        'LogisticRegression': lr,
        'RandomForest': rf,
        'MLP': mlp
    }
    return models, scaler


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
        df = load_data_for_ticker(ticker)
        df = compute_indicators(df)
        df = create_labels(df)

        models_dict, scaler = train_models_for_ticker(ticker, df)
        MODELS[ticker] = models_dict
        SCALERS[ticker] = scaler

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
# 4. ADVANCED BACKTESTING
#    (Multiple positions, partial sells, dynamic stop-loss, probabilities, etc.)
###############################################################################
def advanced_backtest(ticker, model_name,
                      initial_capital=10000,
                      stop_loss_percent=0.05,
                      partial_sell_ratio=0.5,
                      prob_threshold=0.6,
                      trailing_stop=True,
                      take_profit_percent=0.2):
    """
    A demonstration of advanced strategy logic:
    - multiple positions
    - partial sells
    - dynamic trailing stop
    - probability-based weighting
    - additional metrics (Sharpe Ratio, Max Drawdown)
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
        'Close','High','Low','Open','Volume',
        'MA_10','MA_50','MA_200',
        'Daily_Return','RSI_14','MACD','MACD_Signal','MACD_Hist'
    ]
    X = df[feature_cols].values
    X_scaled = SCALERS[ticker].transform(X)

    # Attempt predict_proba
    has_proba = hasattr(classifier, "predict_proba")
    if has_proba:
        probas = classifier.predict_proba(X_scaled)
        predicted_actions = classifier.predict(X_scaled)
    else:
        predicted_actions = classifier.predict(X_scaled)
        probas = None

    # Shift by 1 day
    df['Prediction'] = np.roll(predicted_actions, 1, axis=0)
    if probas is not None:
        shifted_probas = np.roll(probas, 1, axis=0)
    else:
        shifted_probas = None

    # We'll keep multiple positions
    positions = []
    capital = float(initial_capital)
    daily_values = []
    daily_dates = []

    for i, (date_idx, row) in enumerate(df.iterrows()):
        current_price = row['Close']
        # Portfolio value before today's action
        daily_portfolio_value = capital + sum(pos['shares'] * current_price for pos in positions)
        daily_values.append(daily_portfolio_value)
        daily_dates.append(date_idx)

        # Update trailing stops
        for pos in positions:
            # If trailing_stop is on, move stop_loss_price up if new high
            if trailing_stop:
                if current_price > pos['highest_price']:
                    pos['highest_price'] = current_price
                # Re-calc trailing
                new_sl = pos['highest_price'] * (1 - stop_loss_percent)
                if new_sl > pos['stop_loss_price']:
                    pos['stop_loss_price'] = new_sl

        action = row['Prediction']
        buy_confidence = 0.0
        sell_confidence = 0.0
        if shifted_probas is not None and i < len(shifted_probas):
            buy_confidence = shifted_probas[i, 2]  # label=2 => BUY
            sell_confidence = shifted_probas[i, 0] # label=0 => SELL

        # 1) Check existing positions for stop-loss or take-profit
        updated_positions = []
        for pos in positions:
            # If price < stop_loss_price => SELL all
            if current_price <= pos['stop_loss_price']:
                capital += pos['shares'] * current_price
                # do not keep
            else:
                # check take profit
                if (take_profit_percent > 0.0 and
                    current_price >= pos['entry_price'] * (1 + take_profit_percent)):
                    # partial sell
                    shares_to_sell = int(pos['shares'] * partial_sell_ratio)
                    if shares_to_sell > 0:
                        capital += shares_to_sell * current_price
                        pos['shares'] -= shares_to_sell
                    # if pos still has shares, keep it
                    if pos['shares'] > 0:
                        updated_positions.append(pos)
                else:
                    updated_positions.append(pos)

        positions = updated_positions

        # 2) Process model's buy/sell signals with probability threshold
        if action == 2 and buy_confidence >= prob_threshold:
            # Weighted buy => e.g. buy_confidence * capital
            funds_to_spend = capital * buy_confidence
            shares_to_buy = int(funds_to_spend // current_price)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price
                capital -= cost
                new_pos = {
                    'shares': shares_to_buy,
                    'entry_price': current_price,
                    'highest_price': current_price,
                    'stop_loss_price': current_price * (1 - stop_loss_percent)
                }
                positions.append(new_pos)

        if action == 0 and sell_confidence >= prob_threshold:
            # partial sell from each position
            for pos in positions:
                shares_to_sell = int(pos['shares'] * partial_sell_ratio)
                if shares_to_sell > 0:
                    capital += shares_to_sell * current_price
                    pos['shares'] -= shares_to_sell
            # remove empty positions
            positions = [p for p in positions if p['shares'] > 0]

    # End => liquidate
    final_value = capital
    if len(positions) > 0:
        last_price = df.iloc[-1]['Close']
        for pos in positions:
            final_value += pos['shares'] * last_price

    final_return = (final_value - float(initial_capital)) / float(initial_capital) * 100.0
    final_return_str = f"{final_return:.2f}%"

    # Additional metrics => Sharpe ratio, Max drawdown
    daily_returns = []
    for idx in range(1, len(daily_values)):
        ret = (daily_values[idx] - daily_values[idx - 1]) / (daily_values[idx - 1] + 1e-9)
        daily_returns.append(ret)

    if len(daily_returns) > 1:
        mean_ret = np.mean(daily_returns)
        std_ret = np.std(daily_returns, ddof=1)
        sharpe = (mean_ret / (std_ret + 1e-9)) * np.sqrt(252)
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
# 5. FLASK ROUTES
###############################################################################

@app.route('/')
def index():
    """
    Simple home page with links
    """
    html = """
    <h1>Welcome to the Advanced Algorithmic Trading App</h1>
    <ul>
      <li><a href="{{ url_for('train') }}">Train Models (SSE Progress)</a></li>
      <li><a href="{{ url_for('select_backtest_advanced') }}">Run Advanced Backtest</a></li>
    </ul>
    """
    return render_template_string(html)

# ----------------------------------------------------------------------------
#  Training Page (SSE-based)
# ----------------------------------------------------------------------------
@app.route('/train', methods=['GET'])
def train():
    """
    Show a page with a button to start training
    and a progress bar that updates via SSE.
    """
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

# ----------------------------------------------------------------------------
#  Advanced Backtest Selection Page
# ----------------------------------------------------------------------------
@app.route('/select_backtest_advanced', methods=['GET','POST'])
def select_backtest_advanced():
    """
    A page that allows the user to specify advanced parameters for backtesting.
    """
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
    # 1. Grab all parameters from request.args
    ticker = request.args.get('ticker')
    model_name = request.args.get('model_name')
    initial_capital = float(request.args.get('initial_capital', '10000'))
    stop_loss_percent = float(request.args.get('stop_loss_percent', '0.05'))
    partial_sell_ratio = float(request.args.get('partial_sell_ratio', '0.5'))
    prob_threshold = float(request.args.get('prob_threshold', '0.6'))
    trailing_stop = (request.args.get('trailing_stop', '0') == '1')
    take_profit_percent = float(request.args.get('take_profit_percent', '0.2'))

    # 2. Run your advanced backtest logic
    #    This function returns final_val, final_ret_str, daily_dates, daily_values, metrics
    final_val, final_ret_str, daily_dates, daily_values, metrics = advanced_backtest(
        ticker, model_name,
        initial_capital=initial_capital,
        stop_loss_percent=stop_loss_percent,
        partial_sell_ratio=partial_sell_ratio,
        prob_threshold=prob_threshold,
        trailing_stop=trailing_stop,
        take_profit_percent=take_profit_percent
    )

    # 3. If there's an error (e.g. model not found), handle gracefully
    if final_val is None:
        result_html = f"<p>Error: {final_ret_str}</p>"
        img_html = ""
    else:
        # Plotting the daily portfolio values
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

        # Additional metrics (Sharpe, MaxDrawdown, etc.) from `metrics`
        result_html = f"""
        <p>Final Capital: {metrics['FinalValue']}</p>
        <p>Percent Return: {metrics['PercentReturn']}</p>
        <p>Sharpe Ratio: {metrics['SharpeRatio']}</p>
        <p>Max Drawdown: {metrics['MaxDrawdown']}</p>
        """

    # 4. Re-display the advanced form at the bottom, pre-filled with last inputs
    #    We need to re-fetch the tickers & model names if you want to let the user switch them
    tickers = sorted(list_tickers())
    model_names_list = ['LogisticRegression', 'RandomForest', 'MLP']

    # Build a mini form that includes all fields (ticker, model_name, capital, etc.)
    # We use a GET form that re-calls this same endpoint (`backtest_advanced_route`)
    # so the user can tweak & re-run
    re_run_form = f"""
    <hr>
    <h2>Refine Your Backtest</h2>
    <form method="GET" action="{url_for('backtest_advanced')}">
      <!-- Ticker dropdown (selects the last used ticker by default) -->
      <label for="ticker">Ticker:</label>
      <select name="ticker">
        {"".join(
          f'<option value="{t}" {"selected" if t==ticker else ""}>{t}</option>'
          for t in tickers
        )}
      </select>
      <br><br>

      <!-- Model dropdown (selects the last used model by default) -->
      <label for="model_name">Model:</label>
      <select name="model_name">
        {"".join(
          f'<option value="{m}" {"selected" if m==model_name else ""}>{m}</option>'
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

    # 5. Combine everything into final page output
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


###############################################################################
# MAIN
###############################################################################
if __name__ == '__main__':
    load_models()
    app.run(debug=True)