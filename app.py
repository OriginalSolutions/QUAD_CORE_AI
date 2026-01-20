import sys
import os
import time
import json
import csv
import math
import random
import threading
import requests
import gc
import copy
import io
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
from flask import Flask, jsonify, render_template, send_file, make_response
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# === CONFIGURATION & PATHS ===
# ==========================================
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  #  (Wyłącza cache przeglądarki dla CSS/JS)
CORS(app)

PNL_FILE_PATH = os.path.join(app.root_path, 'static', 'strategy_pnl.csv')
csv_file_lock = threading.Lock()

BASE_SEED = 42
DEVICE = torch.device("cpu") 
EPOCHS = 120
LEARNING_RATE = 0.0001
KAN_EPOCHS = 90
KAN_LEARNING_RATE = 0.01
Q_LEARNING_RATE = 0.005
LOOKBACK_WINDOW = 3 
TRAIN_SPLIT_INDEX = 990
TOTAL_DATA_POINTS = 1000
HIDDEN_DIM = 64
TEMPERATURE = 0.5
NUM_LAYERS = 5
RSI_PERIOD = 14
SR_LOOKAHEAD = 60
FUTURE_PREDICTION_STEPS = 2 
HISTORY_SHOW = 5
PROB_LOOKAHEAD_MINS = 60
MC_ITERATIONS = 50
MC_TEMP_MULT = 9.0
MC_TREND_SUPPRESSION = 0.3
RF_ESTIMATORS = 100
RF_LOOKAHEAD = 60
RF_RSI_PERIOD = 14
RF_ROC_PERIOD = 5
RF_VOLATILITY_WINDOW = 10
RF_BB_WINDOW = 60
RF_BB_STD = 2.0
INITIAL_BALANCE = 100000.0  
POSITION_SIZE_BTC = 1.0     
data_lock = threading.Lock()

# ==========================================
# === GLOBAL STATE ===
# ==========================================
class ServerState:
    STATUS = "STARTUP"
    LAST_UPDATE = None
    TRAINED_Q_UP = 0.9
    TRAINED_Q_LOW = 0.1
    RF_ACCURACY = 0.50
    MODEL_WEIGHTS = {"mc": 1.0, "rf": 1.0, "kan": 1.0, "net": 1.0}
    LAST_PREDICTIONS = {"mc": None, "rf": None, "kan": None, "net": None}
    LAST_PRICE = None
    # Startujemy od -1 (Strategia Kontra)
    STRATEGY_MULT = -1.0 

CACHE = {
    "dates": [], "timestamps": [], "history": [],
    "forecast_dates": [], "stoch": [], "trend": [], "res": [], "sup": [],
    "last_real_price": 0, "prob_val": 50.0, "rf_prob_up": 50.0, "rf_raw_prob": 50.0, 
    "rf_acc_view": 0.50, "rf_calc_steps": {"diff": 0, "trust": 0},
    "kan_val": 50.0, "neural_val": 50.0, "consensus_val": 50.0, "consensus_signal": "NEUTRAL",
    "pnl": {"times": [], "balance": []},
    "strategy_mult": -1.0 
}

# ==========================================
# === PAPER TRADER ===
# ==========================================
class PaperTrader:
    def __init__(self, filepath, initial_balance):
        self.filepath = filepath
        self.balance = initial_balance
        self.current_position = None
        self.equity_curve = [initial_balance]
        self.returns = []
        self.peak_balance = initial_balance
        
        with csv_file_lock:
            if not os.path.exists(self.filepath):
                try:
                    with open(self.filepath, mode='w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            "Open_Time", "Close_Time", "Type", "Size_BTC", 
                            "Entry_Price", "Exit_Price", "PnL_USDT", 
                            "Total_Balance", "", "Sharpe_Ratio", "Max_Drawdown_%"
                        ])
                except IOError as e:
                    print(f"!!! Error creating CSV: {e}")
            else:
                try:
                    df = pd.read_csv(self.filepath)
                    if not df.empty:
                        self.balance = float(df.iloc[-1]["Total_Balance"])
                        self.peak_balance = df["Total_Balance"].max()
                        balances = df["Total_Balance"].tolist()
                        self.equity_curve = balances
                        self.returns = [ (balances[i] - balances[i-1]) for i in range(1, len(balances)) ]
                        print(f">>> [Trader] Resumed balance: {self.balance:.2f} USDT")
                except Exception:
                    pass

    def close_position(self, exit_price, exit_time_str):
        if self.current_position is None:
            return

        pos = self.current_position
        pnl = 0.0
        
        if pos['type'] == 'LONG':
            pnl = (exit_price - pos['entry']) * POSITION_SIZE_BTC
        elif pos['type'] == 'SHORT':
            pnl = (pos['entry'] - exit_price) * POSITION_SIZE_BTC
            
        self.balance += pnl
        self.equity_curve.append(self.balance)
        self.returns.append(pnl)
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
            
        sharpe = 0.0
        if len(self.returns) > 1:
            avg_ret = np.mean(self.returns)
            std_ret = np.std(self.returns)
            if std_ret > 0:
                sharpe = avg_ret / std_ret 

        drawdown = 0.0
        if self.peak_balance > 0:
            drawdown = (self.peak_balance - self.balance) / self.peak_balance * 100.0

        with csv_file_lock:
            try:
                with open(self.filepath, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        pos['time'], exit_time_str, pos['type'], POSITION_SIZE_BTC, 
                        f"{pos['entry']:.2f}", f"{exit_price:.2f}", f"{pnl:.2f}", 
                        f"{self.balance:.2f}", "", f"{sharpe:.4f}", f"{drawdown:.2f}"
                    ])
            except IOError:
                print("!!! [Trader] CSV Write Error")

        print(f">>> [Trader] Closed {pos['type']} | PnL: {pnl:.2f} | Bal: {self.balance:.2f}")
        self.current_position = None

    def open_position(self, direction, entry_price, entry_time_str):
        self.current_position = {
            'type': direction,
            'entry': entry_price,
            'time': entry_time_str
        }
        print(f">>> [Trader] Opened {direction} (1 BTC) at {entry_price:.2f}")

TRADER = PaperTrader(PNL_FILE_PATH, INITIAL_BALANCE)

# ==========================================
# === NEURAL NETWORKS ===
# ==========================================
class AdaptiveQuantiles(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_q_up = nn.Parameter(torch.tensor(2.5))
        self.raw_q_low = nn.Parameter(torch.tensor(-2.5))
    def get_q(self):
        q_up = 0.70 + (torch.sigmoid(self.raw_q_up) * 0.29)
        q_low = 0.01 + (torch.sigmoid(self.raw_q_low) * 0.29)
        return q_up, q_low

class TrendNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=0.3)
        self.head = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]) 

class NoiseNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))   
    def forward(self, x):
        out, _ = self.rnn(x)
        raw = self.head(out[:, -1, :])
        return nn.functional.softplus(raw) + 0.1 

class QuantileNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2), nn.Softplus())
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]) 

class DualModeNetwork(nn.Module):
    def __init__(self, input_dim=4, d_model=24, n_layers=2): 
        super().__init__()
        self.trend = TrendNet(input_dim, d_model, n_layers)
        self.noise = NoiseNet(input_dim, d_model)
        self.level = QuantileNet(input_dim, d_model)
        self.adaptive_q = AdaptiveQuantiles() 
    def forward(self, x):
        return self.trend(x), self.noise(x), self.level(x)

class AdvancedKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, grid_size=5):
        super().__init__()
        self.grid_size = grid_size
        self.layernorm = nn.LayerNorm(input_dim)
        self.base_weight = nn.Parameter(torch.Tensor(output_dim, input_dim))
        self.spline_weight = nn.Parameter(torch.Tensor(output_dim, input_dim, grid_size))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.1)
        self.base_activation = nn.SiLU()
    def forward(self, x):
        x = self.layernorm(x)
        base_output = nn.functional.linear(self.base_activation(x), self.base_weight)
        x_norm = torch.tanh(x) 
        spline_output = 0
        for i in range(self.grid_size):
            basis = torch.cos(math.pi * (i + 1) * x_norm)
            term = torch.matmul(basis, self.spline_weight[:, :, i].T)
            spline_output += term
        return base_output + spline_output

class TemporalKAN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.kan_head = AdvancedKANLayer(hidden_dim, 1, grid_size=6)
    def forward(self, x):
        _, hn = self.gru(x) 
        context_vector = hn[-1] 
        context_vector = self.dropout(context_vector)
        out = self.kan_head(context_vector)
        return torch.sigmoid(out)

# ==========================================
# === HELPER FUNCTIONS ===
# ==========================================
def log_msg(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")
    sys.stdout.flush()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def quantile_loss(preds, target, quantile):
    errors = target - preds
    loss = torch.max((quantile - 1) * errors, quantile * errors)
    return torch.abs(loss).mean()

def calculate_rsi(prices, period=14):
    series = pd.Series(prices)
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return (100 - (100 / (1 + rs))).fillna(50).values 

def calculate_volatility(prices, window=10):
    return pd.Series(prices).diff().rolling(window=window).std().fillna(0).values

def calculate_roc(prices, period=5):
    return (pd.Series(prices).pct_change(periods=period).fillna(0).values) * 100 

def get_data_server():
    log_msg(">>> [Data] Fetching market data from Binance API...")
    url = "https://api.binance.com/api/v3/klines"
    limit = TOTAL_DATA_POINTS + RSI_PERIOD + SR_LOOKAHEAD + 100 
    try:
        r = requests.get(url, params={"symbol": "BTCUSDT", "interval": "1m", "limit": limit}, timeout=10).json()
        if isinstance(r, dict) and "code" in r: return [], [], []
        if not isinstance(r, list): return [], [], []
        df = pd.DataFrame(r, columns=["t", "o", "h", "l", "c", "v", "x", "y", "z", "a", "b", "d"])
        df = df.iloc[:-1] 
        timestamps = [int(ts) for ts in df["t"]]
        times = [datetime.fromtimestamp(ts/1000) for ts in timestamps]
        prices = df["c"].astype(float).values
        return times, prices, timestamps
    except Exception as e:
        log_msg(f"!!! [Data] API Error: {e}")
        return [], [], []

def prepare_data(prices):
    rsi = calculate_rsi(prices, RSI_PERIOD)
    vol = calculate_volatility(prices, window=10)
    roc = calculate_roc(prices, period=5)
    diffs = np.zeros_like(prices)
    diffs[1:] = np.diff(prices)
    rsi = np.nan_to_num(rsi, nan=50.0)
    vol = np.nan_to_num(vol, nan=0.0)
    roc = np.nan_to_num(roc, nan=0.0)
    diffs = np.nan_to_num(diffs, nan=0.0)
    start_idx = RSI_PERIOD + 15
    c_diffs, c_rsi = diffs[start_idx:], rsi[start_idx:]
    c_vol, c_roc = vol[start_idx:], roc[start_idx:]
    c_prices = prices[start_idx:]
    s_diff = StandardScaler()
    s_rsi = MinMaxScaler((-1, 1))
    s_vol = MinMaxScaler((0, 1))
    s_roc = StandardScaler()
    dataset = np.hstack((
        s_diff.fit_transform(c_diffs.reshape(-1, 1)),
        s_rsi.fit_transform(c_rsi.reshape(-1, 1)),
        s_vol.fit_transform(c_vol.reshape(-1, 1)),
        s_roc.fit_transform(c_roc.reshape(-1, 1))
    ))
    train_size = TRAIN_SPLIT_INDEX - start_idx
    train_data = dataset[:train_size]
    X, y = [], []
    for i in range(len(train_data) - LOOKBACK_WINDOW):
        X.append(train_data[i : i+LOOKBACK_WINDOW])
        if i + LOOKBACK_WINDOW < len(train_data):
            y.append(train_data[i+LOOKBACK_WINDOW, 0])
        else:
            y.append(0.0)
    return (
        torch.tensor(np.array(X), dtype=torch.float32).to(DEVICE),
        torch.tensor(np.array(y), dtype=torch.float32).to(DEVICE),
        None, s_diff, s_rsi, s_vol, s_roc, c_prices, start_idx, prices
    )

def prepare_rf_data(prices):
    df = pd.DataFrame({"close": prices})
    df['rsi'] = calculate_rsi(prices, RF_RSI_PERIOD)
    df['roc'] = calculate_roc(prices, RF_ROC_PERIOD)
    df['vol'] = calculate_volatility(prices, RF_VOLATILITY_WINDOW)
    df['sma_bb'] = df['close'].rolling(window=RF_BB_WINDOW).mean()
    std_dev = df['close'].rolling(window=RF_BB_WINDOW).std()
    df['bb_position'] = (df['close'] - df['sma_bb']) / (RF_BB_STD * std_dev)
    df['dist_sma'] = (df['close'] - df['sma_bb']) / df['sma_bb']
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df['future_close'] = df['close'].shift(-RF_LOOKAHEAD)
    df['target'] = (df['future_close'] > df['close']).astype(int)
    train_df = df.dropna()
    features = ['rsi', 'roc', 'vol', 'dist_sma', 'bb_position']
    X = train_df[features].values
    y = train_df['target'].values
    last_row = df.iloc[-1][features].values.reshape(1, -1)
    return X, y, last_row

# ==========================================
# === TRAINING & INFERENCE ===
# ==========================================
def train_model(model, X, y, _):
    q_params = list(model.adaptive_q.parameters())
    net_params = list(model.trend.parameters()) + list(model.noise.parameters()) + list(model.level.parameters())
    opt = optim.AdamW([{'params': net_params, 'lr': LEARNING_RATE}, {'params': q_params, 'lr': Q_LEARNING_RATE} ], weight_decay=1e-4)
    loss_fn_t = nn.GaussianNLLLoss()
    model.train()
    print(f"    [Net] Starting Training ({EPOCHS} epochs)...")
    for e in range(EPOCHS):
        opt.zero_grad()
        mu, sigma, quantiles_pred = model(X)
        curr_q_up, curr_q_low = model.adaptive_q.get_q()
        pred_res = mu + quantiles_pred[:, 0].unsqueeze(1)
        pred_sup = mu - quantiles_pred[:, 1].unsqueeze(1)
        loss_trend = loss_fn_t(mu, y, sigma)
        loss_dir = (1.0 - torch.mean(torch.tanh(mu) * torch.tanh(y)))
        loss_q_upper = quantile_loss(pred_res, y, curr_q_up)
        loss_q_lower = quantile_loss(pred_sup, y, curr_q_low)
        channel_width = curr_q_up - curr_q_low
        loss_penalty = torch.pow(channel_width, 2) * 1.0 
        loss = loss_trend + (loss_dir * 1.0) + (loss_q_upper * 5.0) + (loss_q_lower * 5.0) + loss_penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) 
        opt.step()
        if (e+1) % 20 == 0:
            print(f"    [Net] Epoch {e+1}/{EPOCHS} | Loss: {loss.item():.4f}", flush=True)

    with torch.no_grad():
        final_q_up, final_q_low = model.adaptive_q.get_q()
        ServerState.TRAINED_Q_UP = final_q_up.item()
        ServerState.TRAINED_Q_LOW = final_q_low.item()

def forecast(model, start_win, steps, s_diff, s_rsi, s_vol, s_roc, last_price_val, history_prices, temp_override=None, suppression=None):
    model.eval()
    curr_input = torch.tensor(start_win, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    current_temp = temp_override if temp_override is not None else TEMPERATURE
    is_monte_carlo = temp_override is not None and temp_override > 1.0
    trend_suppression = suppression if suppression is not None else 1.0
    preds_stoch, preds_trend, preds_res, preds_sup = [], [], [], []
    sim_history = [last_price_val] * (RSI_PERIOD + 20) 
    recent_volatility = np.std(history_prices[-15:]) if len(history_prices) > 15 else last_price_val * 0.005
    initial_offset_res = recent_volatility * 2.0
    initial_offset_sup = recent_volatility * 2.0
    start_jitter = 0
    if is_monte_carlo: start_jitter = np.random.normal(0, recent_volatility * 0.2)
    pos_trend = last_price_val + start_jitter
    vel_trend = 0.0; pos_stoch = last_price_val + start_jitter; vel_stoch = 0.0
    pos_res = last_price_val + initial_offset_res; vel_res = 0.0
    pos_sup = last_price_val - initial_offset_sup; vel_sup = 0.0
    INERTIA_TREND = 0.60; FORCE_TREND = 0.25; MAX_VELOCITY = recent_volatility * 0.9 
    INERTIA_STOCH = 0.50; FORCE_STOCH = 0.70
    TETHER_STRENGTH = 0.0015 * (abs(trend_suppression) if trend_suppression < 1.0 else 1.0)
    drift_res_factor = random.uniform(0.9, 1.1)
    drift_sup_factor = random.uniform(0.9, 1.1)
    with torch.no_grad():
        for i in range(steps):
            mu, sigma, quantiles = model(curr_input)
            raw_mu = mu.item(); raw_sigma = sigma.item()
            real_trend_delta = s_diff.inverse_transform([[raw_mu]])[0][0]
            real_sigma_delta = raw_sigma * s_diff.scale_[0] 
            if abs(trend_suppression) < 0.1: avg_drift = s_diff.mean_[0]; real_trend_delta -= avg_drift
            trend_component = real_trend_delta * trend_suppression
            noise_component = 0.0
            if is_monte_carlo: noise_component = real_sigma_delta * torch.randn(1).item() * current_temp
            final_delta_usd = trend_component + noise_component
            max_allowed_move = recent_volatility * 3.0
            delta_trend = np.clip(final_delta_usd, -max_allowed_move, max_allowed_move)
            target_trend = pos_trend + delta_trend
            raw_width_res = quantiles[0, 0].item() * s_diff.scale_[0] * 8.0
            raw_width_sup = quantiles[0, 1].item() * s_diff.scale_[0] * 8.0
            current_rsi_norm = curr_input[0, -1, 1].item()
            gravity = 0.0
            if current_rsi_norm > 0.8: gravity = -recent_volatility * 0.1
            elif current_rsi_norm < -0.8: gravity = recent_volatility * 0.1
            gravity *= abs(trend_suppression)
            vel_trend = (vel_trend * INERTIA_TREND) + ((target_trend - pos_trend) * FORCE_TREND) + gravity
            vel_trend = np.clip(vel_trend, -MAX_VELOCITY, MAX_VELOCITY)
            pos_trend += vel_trend
            local_vol_factor = np.std(s_diff.inverse_transform(start_win[:, 0].reshape(-1,1)))
            chaos = 1.0 + (abs(current_rsi_norm) * 2.0)
            stoch_noise_delta = (real_sigma_delta * chaos * torch.randn(1).item() * current_temp * 1.5)
            target_stoch = pos_trend + stoch_noise_delta
            tether_pull = (pos_trend - pos_stoch) * TETHER_STRENGTH
            vel_stoch = (vel_stoch * INERTIA_STOCH) + ((target_stoch - pos_stoch) * FORCE_STOCH) + tether_pull
            vel_stoch = np.clip(vel_stoch, -MAX_VELOCITY*3.0, MAX_VELOCITY*3.0)
            pos_stoch += vel_stoch
            if current_rsi_norm > 0.5: res_stiffness = 0.05
            else: res_stiffness = 0.15 
            if current_rsi_norm < -0.5: sup_stiffness = 0.05
            else: sup_stiffness = 0.15
            target_res = (pos_trend + raw_width_res * drift_res_factor)
            target_sup = (pos_trend - raw_width_sup * drift_sup_factor)
            vel_res = (vel_res * 0.85) + ((target_res - pos_res) * res_stiffness)
            pos_res += vel_res
            vel_sup = (vel_sup * 0.85) + ((target_sup - pos_sup) * sup_stiffness)
            pos_sup += vel_sup
            preds_trend.append(pos_trend); preds_stoch.append(pos_stoch)
            preds_res.append(pos_res); preds_sup.append(pos_sup)
            sim_history.append(pos_stoch)
            full_arr = np.array(sim_history)
            new_rsi_sc = s_rsi.transform([[calculate_rsi(full_arr, RSI_PERIOD)[-1]]])[0][0]
            diff_arr = np.diff(full_arr)
            new_vol_sc = s_vol.transform([[np.std(diff_arr[-10:]) if len(diff_arr)>10 else 0]])[0][0]
            pct_chg = (full_arr[-1]-full_arr[-6])/(full_arr[-6]+1e-5)*100 if len(full_arr)>6 else 0
            new_roc_sc = s_roc.transform([[pct_chg]])[0][0]
            new_roc_sc = np.clip(new_roc_sc, -3.0, 3.0) 
            real_delta = vel_stoch 
            scaled_delta = s_diff.transform([[real_delta]])[0][0]
            scaled_delta = np.clip(scaled_delta, -3.0, 3.0)
            feat = torch.tensor([[[scaled_delta, new_rsi_sc, new_vol_sc, new_roc_sc]]], dtype=torch.float32).to(DEVICE)
            curr_input = torch.cat((curr_input[:, 1:, :], feat), dim=1)
    return preds_stoch, preds_trend, preds_res, preds_sup

def calculate_monte_carlo_probability(model, start_win, s_diff, s_rsi, s_vol, s_roc, last_price, history_prices):
    log_msg(f">>> [Probability] Running Monte Carlo ({MC_ITERATIONS} runs)...")
    up_count = 0
    mc_temp = TEMPERATURE * MC_TEMP_MULT 
    for _ in range(MC_ITERATIONS):
        p_stoch, _, _, _ = forecast(model, start_win, PROB_LOOKAHEAD_MINS, s_diff, s_rsi, s_vol, s_roc, last_price, history_prices, temp_override=mc_temp, suppression=MC_TREND_SUPPRESSION)
        if p_stoch[-1] > last_price: up_count += 1
    probability_up = (up_count / MC_ITERATIONS) * 100.0
    return probability_up

def adjust_weights(current_price):
    if ServerState.LAST_PRICE is None or ServerState.LAST_PREDICTIONS["mc"] is None: return
    actual_direction_up = current_price > ServerState.LAST_PRICE
    direction_str = "UP" if actual_direction_up else "DOWN"
    log_msg(f">>> [Adaptive] Checking performance. Market went {direction_str}")
    for model_name, prev_prob in ServerState.LAST_PREDICTIONS.items():
        if prev_prob is None: continue
        model_said_up = prev_prob > 50.0
        is_correct = (model_said_up == actual_direction_up)
        step = 0.05
        if is_correct: ServerState.MODEL_WEIGHTS[model_name] += step
        else: ServerState.MODEL_WEIGHTS[model_name] -= step
        ServerState.MODEL_WEIGHTS[model_name] = max(0.2, min(2.0, ServerState.MODEL_WEIGHTS[model_name]))
    
    w = ServerState.MODEL_WEIGHTS
    log_msg(f">>> [Adaptive] NEW WEIGHTS: MC={w['mc']:.2f}, RF={w['rf']:.2f}, KAN={w['kan']:.2f}, NET={w['net']:.2f}")

# ==========================================
# === MAIN LOGIC ===
# ==========================================
def run_ai_training_sequence():
    try:
        ServerState.STATUS = "TRAINING"
        print("\n" + "="*40 + "\n   STARTING QUAD-CORE AI TRAINING SEQUENCE\n" + "="*40)
        cleanup_memory()
        set_seed(int(time.time())) 
        times_all, prices_all, timestamps_all = get_data_server()
        if len(prices_all) < 200:
            log_msg("!!! Error: Not enough data points.")
            ServerState.STATUS = "ERROR"
            return
        def get_live_execution_data():
            try:
                r = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=3).json()
                price = float(r['price'])
            except: price = prices_all[-1]
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return price, time_str
        exec_close_price, exec_close_time = get_live_execution_data()
        if TRADER.current_position is not None:
            TRADER.close_position(exec_close_price, exec_close_time)
        adjust_weights(exec_close_price)
        X, y, _, s_diff, s_rsi, s_vol, s_roc, clean_prices, offset, raw_all_prices = prepare_data(prices_all)
        raw_diffs = np.diff(raw_all_prices).reshape(-1,1)[-LOOKBACK_WINDOW:]
        in_diff = s_diff.transform(raw_diffs)
        raw_rsi = calculate_rsi(raw_all_prices, RSI_PERIOD)[-LOOKBACK_WINDOW:]
        in_rsi = s_rsi.transform(raw_rsi.reshape(-1,1))
        raw_vol = calculate_volatility(raw_all_prices, window=10)[-LOOKBACK_WINDOW:]
        in_vol = s_vol.transform(raw_vol.reshape(-1,1))
        raw_roc = calculate_roc(raw_all_prices, period=5)[-LOOKBACK_WINDOW:]
        in_roc = s_roc.transform(raw_roc.reshape(-1,1))
        start_win = np.hstack((in_diff, in_rsi, in_vol, in_roc))
        
        # --- MODELS ---
        model = DualModeNetwork(input_dim=4, d_model=HIDDEN_DIM, n_layers=NUM_LAYERS).to(DEVICE)
        train_model(model, X, y, None) 
        prob_up_mc = calculate_monte_carlo_probability(model, start_win, s_diff, s_rsi, s_vol, s_roc, raw_all_prices[-1], raw_all_prices)
        log_msg(f">>> [1/4] Monte Carlo: {prob_up_mc:.1f}%")
        
        rf_X, rf_y, rf_last_row = prepare_rf_data(raw_all_prices)
        split = int(len(rf_X) * 0.8)
        X_train, X_test, y_train, y_test = rf_X[:split], rf_X[split:], rf_y[:split], rf_y[split:]
        rf_model = RandomForestClassifier(n_estimators=RF_ESTIMATORS, max_depth=5, min_samples_split=10, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_acc = accuracy_score(y_test, rf_model.predict(X_test))
        ServerState.RF_ACCURACY = rf_acc
        rf_model.fit(rf_X, rf_y)
        rf_raw_prob_float = rf_model.predict_proba(rf_last_row)[0][1] * 100.0
        raw_diff = rf_raw_prob_float - 50.0
        trust_factor = (rf_acc - 0.50) * 2.0 
        rf_final_val = round(50.0 + (raw_diff * trust_factor), 2)
        log_msg(f">>> [2/4] Random Forest: {rf_final_val:.2f}%")
        
        log_msg(f">>> [T-KAN] Training Advanced Spline Network ({KAN_EPOCHS} epochs)...")
        y_binary = (y > 0).float().unsqueeze(1) 
        kan_model = TemporalKAN(input_dim=4, hidden_dim=32).to(DEVICE)
        kan_opt = optim.AdamW(kan_model.parameters(), lr=KAN_LEARNING_RATE, weight_decay=1e-4)
        kan_loss_fn = nn.BCELoss()
        kan_model.train()
        for ke in range(KAN_EPOCHS): 
            kan_opt.zero_grad()
            out_kan = kan_model(X)
            loss_kan = kan_loss_fn(out_kan, y_binary)
            loss_kan.backward()
            kan_opt.step()
            if (ke+1) % 10 == 0:
                print(f"    [T-KAN] Epoch {ke+1}/{KAN_EPOCHS} | Loss: {loss_kan.item():.4f}", flush=True)

        kan_model.eval()
        with torch.no_grad():
            kan_input = torch.tensor(start_win, dtype=torch.float32).unsqueeze(0).to(DEVICE)
            raw_kan_prob = kan_model(kan_input).item() * 100.0
            kan_prob = 50.0 + ((raw_kan_prob - 50.0) * 0.25)
        log_msg(f">>> [3/4] T-KAN: {kan_prob:.2f}%")

        set_seed(int(time.time()) + 1)       
        p_stoch, p_trend, p_res, p_sup = forecast(model, start_win, FUTURE_PREDICTION_STEPS, s_diff, s_rsi, s_vol, s_roc, raw_all_prices[-1], raw_all_prices)
        start_price = raw_all_prices[-1]
        end_price = p_trend[-1]
        trend_change_pct = (end_price - start_price) / start_price * 100.0
        raw_trend_score = (1 / (1 + np.exp(-trend_change_pct * 50.0))) * 100.0
        log_msg(f">>> [4/4] Neural Trend: {raw_trend_score:.1f}%")

        w_mc, w_rf, w_kan, w_net = ServerState.MODEL_WEIGHTS["mc"], ServerState.MODEL_WEIGHTS["rf"], ServerState.MODEL_WEIGHTS["kan"], ServerState.MODEL_WEIGHTS["net"]
        if ServerState.RF_ACCURACY > 0.70: w_rf += 0.4
        
        log_msg(f"    [Consensus Weights] MC:{w_mc:.2f} RF:{w_rf:.2f} KAN:{w_kan:.2f} NET:{w_net:.2f}")

        weighted_sum = (prob_up_mc * w_mc) + (rf_final_val * w_rf) + (kan_prob * w_kan) + (raw_trend_score * w_net)
        total_weight = w_mc + w_rf + w_kan + w_net
        avg_raw_full = weighted_sum / total_weight if total_weight > 0 else 50.0
        
        # --- UŻYCIE OBECNEGO MNOŻNIKA ---
        # 1. Pobieramy mnożnik, który zaraz zostanie użyty do decyzji
        used_mult = ServerState.STRATEGY_MULT
        
        consensus_text = "NEUTRAL"
        trade_action = None
        if avg_raw_full > 50.10: 
            consensus_text = "BUY"
            # Logika handlu: BUY * 1 -> LONG, BUY * -1 -> SHORT
            trade_action = "LONG" if used_mult > 0 else "SHORT"
        elif avg_raw_full < 49.90: 
            consensus_text = "SELL"
            # Logika handlu: SELL * 1 -> SHORT, SELL * -1 -> LONG
            trade_action = "SHORT" if used_mult > 0 else "LONG"
            
        log_msg(f">>> [Consensus] {consensus_text} ({avg_raw_full:.2f}%) [Used Mult: {used_mult}]")

        exec_open_price, exec_open_time = get_live_execution_data()
        if trade_action:
            TRADER.open_position(trade_action, exec_open_price, exec_open_time)
        ServerState.LAST_PRICE = exec_open_price
        ServerState.LAST_PREDICTIONS = {"mc": prob_up_mc, "rf": rf_final_val, "kan": kan_prob, "net": raw_trend_score}

        with data_lock:
            CACHE.update({
                "prob_val": round(prob_up_mc, 1),
                "prob_minutes": PROB_LOOKAHEAD_MINS,
                "rf_prob_up": rf_final_val,
                "rf_raw_prob": round(rf_raw_prob_float, 1),
                "rf_acc_view": round(rf_acc, 2),
                "rf_calc_steps": {"diff": round(raw_diff, 1), "trust": round(trust_factor, 2)},
                "kan_val": round(kan_prob, 2),
                "neural_val": round(raw_trend_score, 1),
                "consensus_val": round(avg_raw_full, 2),
                "consensus_signal": consensus_text, 
                "last_real_price": raw_all_prices[-1],
                # WAŻNE: Frontend dostaje "used_mult", czyli to co widział system podczas otwierania
                "strategy_mult": used_mult 
            })
            times_pnl = []
            balance_pnl = []
            
            with csv_file_lock:
                try:
                    if os.path.exists(PNL_FILE_PATH):
                        df_pnl = pd.read_csv(PNL_FILE_PATH)
                        if not df_pnl.empty:
                            times_pnl = df_pnl["Close_Time"].tolist()
                            balance_pnl = df_pnl["Total_Balance"].tolist()
                except: pass
            
            CACHE["pnl"] = {"times": times_pnl, "balance": balance_pnl}
            last_date = times_all[-1]
            future_dates = [last_date + timedelta(minutes=i) for i in range(1, FUTURE_PREDICTION_STEPS + 1)]
            history_slice_idx = -HISTORY_SHOW if len(raw_all_prices) > HISTORY_SHOW else 0
            CACHE["dates"] = [t.strftime("%Y-%m-%d %H:%M:%S") for t in times_all[history_slice_idx:]] 
            CACHE["timestamps"] = timestamps_all[history_slice_idx:] 
            CACHE["history"] = raw_all_prices[history_slice_idx:].tolist()
            last_hist_val = raw_all_prices[-1]
            recent_vol = np.std(raw_all_prices[-15:]) if len(raw_all_prices)>15 else last_hist_val*0.005
            init_res_vis = last_hist_val + (recent_vol * 2.0)
            init_sup_vis = last_hist_val - (recent_vol * 2.0)
            CACHE["forecast_dates"] = [last_date.strftime("%Y-%m-%d %H:%M:%S")] + [t.strftime("%Y-%m-%d %H:%M:%S") for t in future_dates]
            CACHE["stoch"] = [last_hist_val] + p_stoch
            CACHE["trend"] = [last_hist_val] + p_trend
            CACHE["res"] = [init_res_vis] + p_res 
            CACHE["sup"] = [init_sup_vis] + p_sup 
        
        # --- ZMIANA NA PRZECIWNY DOPIERO PO ZAPISANIU CACHE ---
        ServerState.STRATEGY_MULT = used_mult * -1.0
        log_msg(f">>> [Strategy] Toggled Multiplier to: {ServerState.STRATEGY_MULT} for NEXT cycle.")

        log_msg(f">>> [Complete] All models updated.")
        ServerState.LAST_UPDATE = datetime.now()
        ServerState.STATUS = "READY"
        del model, kan_model, rf_model
        del X, y, raw_diffs, in_diff, rf_X, rf_y
        cleanup_memory()
    except Exception as e:
        log_msg(f"!!! CRITICAL ERROR IN TRAINING LOOP: {e}")
        ServerState.STATUS = "ERROR"
        cleanup_memory()

def background_worker():
    log_msg(f">>> [System] Worker Init. Interval: {FUTURE_PREDICTION_STEPS} min.")
    now = datetime.now()
    sec = 60 - now.second + 1
    if sec > 60: sec -= 60
    time.sleep(sec)
    while True:
        cycle_start_time = datetime.now()
        run_ai_training_sequence()
        next_target_time = cycle_start_time + timedelta(minutes=FUTURE_PREDICTION_STEPS)
        next_target_time = next_target_time.replace(second=1, microsecond=0)
        current_time_after_training = datetime.now()
        sleep_duration = (next_target_time - current_time_after_training).total_seconds()
        if sleep_duration > 0: time.sleep(sleep_duration)

# ==========================================
# === FLASK ROUTES ===
# ==========================================
@app.route('/')
def index():
    # Odczyt prostego pliku disclaimer.txt bez doklejania czegokolwiek
    disclaimer_path = os.path.join(app.root_path, 'templates', 'disclaimer.txt')
    disclaimer_content = "Disclaimer info not available."
    try:
        if os.path.exists(disclaimer_path):
            with open(disclaimer_path, 'r', encoding='utf-8') as f:
                disclaimer_content = f.read()
    except Exception: 
        pass
    
    return render_template('index.html', disclaimer=disclaimer_content)

@app.route('/download-csv')
def download_csv():
    with csv_file_lock:
        if not os.path.exists(PNL_FILE_PATH):
            return "File not found or not yet generated.", 404
        with open(PNL_FILE_PATH, 'rb') as f:
            data = io.BytesIO(f.read())
    return send_file(
        data,
        mimetype='text/csv',
        as_attachment=True,
        download_name='strategy_pnl.csv'
    )

@app.route('/api/init')
def api_init():
    with data_lock:
        data = copy.deepcopy(CACHE)
    data['models'] = {
        'rf_acc': ServerState.RF_ACCURACY, 'rf_prob': data['rf_prob_up'], 
        'mc_prob': data['prob_val'], 'kan_prob': data.get('kan_val', 50.0),
        'neural_prob': data.get('neural_val', 50.0), 'consensus_val': data.get('consensus_val', 50.0),
        'rf_raw': data['rf_raw_prob'], 'rf_acc_view': data['rf_acc_view'],
        'rf_steps': data.get('rf_calc_steps', {}), 'weights': ServerState.MODEL_WEIGHTS,
        'mult': data.get('strategy_mult', -1.0),
        'config': {'win': LOOKBACK_WINDOW, 'ahead': PROB_LOOKAHEAD_MINS, 'iter': MC_ITERATIONS, 'temp': MC_TEMP_MULT, 'sup': MC_TREND_SUPPRESSION }
    }
    resp = make_response(jsonify(data))
    resp.headers['Cache-Control'] = 'no-store'
    return resp









@app.route('/api/current_price')
def api_current():
    try:
        # Próba pobrania ceny z Binance (limit czasu 2s)
        ticker = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=2).json()
        live_price = float(ticker['price'])
        
        # Próba pobrania świecy
        klines = requests.get("https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=1", timeout=2).json()
        closed_candle = klines[0]
        
        # Jeśli się uda, zapisz jako ostatnią udaną cenę
        ServerState.LAST_PRICE = live_price
        
        return jsonify({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            "price": live_price,
            "closed_candle": {
                "time": datetime.fromtimestamp(int(closed_candle[0])/1000).strftime("%Y-%m-%d %H:%M:%S"),
                "price": float(closed_candle[4]), "ts": int(closed_candle[0])
            },
            "status": ServerState.STATUS
        })

    except Exception as e:
        # === SYSTEM AUTONAPRAWY ===
        # Logujemy błąd w konsoli (żebyś wiedział, że coś jest nie tak z siecią)
        print(f"!!! [API Warning] Binance lag: {e}. Using cached price.")
        
        # Używamy ostatniej zapamiętanej ceny, żeby nie psuć frontendu
        fallback_price = ServerState.LAST_PRICE if ServerState.LAST_PRICE else 0.0
        
        # Zwracamy kod 200 (OK) zamiast 500, ale z informacją o statusie
        return jsonify({
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "price": fallback_price,
            "closed_candle": None, 
            "status": "RECONNECTING..." 
        })



# ==========================================
# TO MUSI BYĆ PRZED SEKCJĄ "if __name__"
# ==========================================

# 1. Sprawdzamy/Tworzymy plik PnL
if not os.path.exists(PNL_FILE_PATH):
    with open(PNL_FILE_PATH, 'w') as f:
        f.write("Open_Time,Close_Time,Type,Size_BTC,Entry_Price,Exit_Price,PnL_USDT,Total_Balance,Sharpe_Ratio,Max_Drawdown_%\n")

# 2. Uruchamiamy Twój wątek w tle
# Ponieważ kod jest w głównym ciele pliku, Gunicorn wykona go przy starcie
t = threading.Thread(target=background_worker, daemon=True)
t.start()
print(">>> Background Worker Started (Gunicorn ready)")

# ==========================================
# KONIEC - Sekcji uruchamialnej
# ==========================================



if __name__ == "__main__":
    if not os.path.exists(PNL_FILE_PATH):
        with open(PNL_FILE_PATH, 'w') as f:
            f.write("Open_Time,Close_Time,Type,Size_BTC,Entry_Price,Exit_Price,PnL_USDT,Total_Balance,,Sharpe_Ratio,Max_Drawdown_%\n")

    t = threading.Thread(target=background_worker, daemon=True)
    t.start()
    print(">>> Server running on http://127.0.0.1:8050")
    app.run(host='0.0.0.0', port=8050, debug=False, use_reloader=False)
