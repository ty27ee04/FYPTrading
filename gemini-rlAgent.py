import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def denoise_series(series, span=5):
    """Applies Exponential Moving Average - 100% Causal (No Future Peeking)."""
    return series.ewm(span=span, adjust=False).mean()

def preprocess_gold_data(train_path, test_path, lookback=60, max_horizon=24, pt_mult=2.5, sl_mult=2.5):
    def apply_tbm(path):
        if not os.path.exists(path): return None
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        df = df.drop(columns=['spread', 'real_volume'], errors='ignore')
        
        # --- 1. DATA DENOISING ---
        # Smooth 'close' price before calculating indicators
        df['close_smooth'] = denoise_series(df['close'])

        # 1. TIME-OF-DAY FEATURES (Cyclical Encoding)
        # Allows model to distinguish between quiet Asia and volatile NY sessions
        df['hour'] = df['time'].dt.hour
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

        # --- 2. INDICATORS (Using Smoothed Prices) ---
        delta = df['close_smooth'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['rsi_n'] = (100 - (100 / (1 + (gain / (loss + 1e-9))))) / 100.0

        tp = (df['high'] + df['low'] + df['close_smooth']) / 3
        rmf = tp * df['tick_volume']
        df['mfi_n'] = (100 - (100 / (1 + (rmf.where(tp > tp.shift(1), 0).rolling(14).sum() / (rmf.where(tp < tp.shift(1), 0).rolling(14).sum() + 1e-9))))) / 100.0
        
        # ATR for barriers
        h_l, h_pc, l_pc = df['high']-df['low'], (df['high']-df['close'].shift(1)).abs(), (df['low']-df['close'].shift(1)).abs()
        df['atr'] = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1).rolling(window=14).mean()
        
        # Volatility Filter
        df['vol_filter'] = df['atr'] / (df['atr'].rolling(window=288).mean() + 1e-9)

        df = df.dropna().reset_index(drop=True)

        # --- 3. TRIPLE BARRIER LABELING ---
        c, h, l, a = df['close'].values, df['high'].values, df['low'].values, df['atr'].values
        labels = np.zeros(len(df), dtype=int)
        for i in range(len(df) - max_horizon):
            # Re-balanced barriers: Sells are often sharper, so we use sl_mult carefully
            up, lo = c[i] + (pt_mult * a[i]), c[i] - (sl_mult * a[i])
            f_pt = np.where(h[i+1:i+1+max_horizon] >= up)[0]
            f_sl = np.where(l[i+1:i+1+max_horizon] <= lo)[0]
            p_idx, s_idx = f_pt[0] if len(f_pt)>0 else 999, f_sl[0] if len(f_sl)>0 else 999
            if p_idx < s_idx: labels[i] = 1 
            elif s_idx < p_idx: labels[i] = 2 
            else: labels[i] = 0 
            
        df['label'] = labels
        df = df.iloc[:-max_horizon].copy()

        # --- 4. CYCLICAL TIME ---
        df['hour'] = df['time'].dt.hour
        df['sin_h'], df['cos_h'] = np.sin(2*np.pi*df['hour']/24), np.cos(2*np.pi*df['hour']/24)

        df['log_ret'] = np.log(df['close_smooth'] / df['close_smooth'].shift(1))
        df['atr_p'] = df['atr'] / df['close_smooth']
        
        return df.dropna().reset_index(drop=True)

    df_tr, df_te = apply_tbm(train_path), apply_tbm(test_path)
    feat_cols = ['log_ret', 'rsi_n', 'mfi_n', 'atr_p', 'vol_filter', 'sin_h', 'cos_h']
    
    try:
        scaler = joblib.load('scaler.pkl')
        print("[*] Scaler loaded successfully from DL phase.")
    except:
        print("[!] Warning: scaler.pkl not found. Falling back to new fit (Not recommended).")
        scaler = StandardScaler()
        scaler.fit(df_tr[feat_cols])

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(df_tr[feat_cols])
    X_te_s = scaler.transform(df_te[feat_cols])
    
    def seq_gen(data, labels):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback]); y.append(labels[i+lookback])
        return np.array(X), np.array(y)
    
    X_tr, y_tr = seq_gen(X_tr_s, df_tr['label'].values)
    X_te, y_te = seq_gen(X_te_s, df_te['label'].values)
    return X_tr, y_tr, X_te, y_te, df_tr.iloc[lookback:].reset_index(drop=True), df_te.iloc[lookback:].reset_index(drop=True)

# ==========================================
# 1. RE-DEFINE MODELS (For Loading)
# ==========================================
class AttentionLayer(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.w = nn.Linear(hid_dim, 1, bias=False)
    def forward(self, x):
        weights = F.softmax(self.w(torch.tanh(x)), dim=1)
        return torch.sum(x * weights, dim=1), weights

class ModelA_Base(nn.Module):
    def __init__(self, in_dim, hid_dim=192):
        super().__init__()
        self.cnn = nn.Conv1d(in_dim, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, hid_dim, batch_first=True, num_layers=2)
        self.attn = AttentionLayer(hid_dim)
        self.head = nn.Linear(hid_dim, 3)
    def forward(self, x):
        x = F.relu(self.cnn(x.permute(0, 2, 1))).permute(0, 2, 1)
        out, _ = self.lstm(x)
        ctx, _ = self.attn(out)
        return self.head(ctx)

class ModelB_TCN(nn.Module):
    def __init__(self, in_dim, num_channels=[32, 32], kernel_size=3):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_ch = in_dim if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers += [
                nn.ConstantPad1d(((kernel_size-1) * dilation_size, 0), 0),
                nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation_size),
                nn.ReLU(), nn.Dropout(0.2)
            ]
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_channels[-1], 2)
    def forward(self, x):
        x = self.network(x.permute(0, 2, 1))
        return self.classifier(x[:, :, -1])

# ==========================================
# 2. THE RL ASSISTED ENVIRONMENT
# ==========================================
# ==========================================
# 3. RL ENVIRONMENT (Continuous Risk-Scaler)
# ==========================================
class GoldTradingEnv(gym.Env):
    def __init__(self, X, meta_df, model_a, model_b, initial_balance=100):
        super(GoldTradingEnv, self).__init__()
        self.X = torch.FloatTensor(X).to(device)
        self.meta_df = meta_df
        self.model_a = model_a.eval()
        self.model_b = model_b.eval()
        self.initial_balance = initial_balance
        
        # Action: 0.0 to 1.0 (Mapped to 0.01 to 0.10 lots)
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)
        self.reset()

    def _get_obs(self):
        with torch.no_grad():
            x_input = self.X[self.current_step].unsqueeze(0)
            p_a = F.softmax(self.model_a(x_input), dim=1).cpu().numpy()[0]
            p_b = F.softmax(self.model_b(x_input), dim=1).cpu().numpy()[0][1]
        
        meta = self.meta_df.iloc[self.current_step]
        fpnl = (meta['close'] - self.entry_price) * self.position * self.current_lot * 100 if self.position != 0 else 0
        
        return np.array([
            p_a[1], p_a[2], p_b, 
            meta['atr_p']*100, meta['sin_h'], meta['cos_h'],
            fpnl/self.balance
        ], dtype=np.float32)

    def step(self, action):
        reward = 0
        meta = self.meta_df.iloc[self.current_step]
        close_price = meta['close']
        
        # 1. FIXED STRATEGY EXIT (Model A controls the logic)
        obs = self._get_obs()
        buy_p, sell_p, valid_p = obs[0], obs[1], obs[2]
        
        if self.position != 0:
            # Exit if Signal flips or disappears
            should_exit = False
            if self.position == 1 and buy_p < 0.40: should_exit = True
            if self.position == -1 and sell_p < 0.40: should_exit = True
            
            if should_exit or self.current_step >= len(self.X) - 2:
                pnl = (close_price - self.entry_price) * self.position * self.current_lot * 100
                self.balance += pnl
                reward += pnl * 1.0 # Reward for real money
                self.position = 0; self.current_lot = 0

        # 2. RL ENTRY (RL controls the risk)
        elif valid_p > 0.52: # Only trade if Model B says the regime is valid
            if buy_p > 0.48 or sell_p > 0.48:
                multiplier = float(action[0])
                # Scale from 0.01 to 0.10 lots
                self.current_lot = round(0.01 + (multiplier * 0.09), 2)
                self.position = 1 if buy_p > sell_p else -1
                self.entry_price = close_price
                reward -= 0.05 # Transaction fee

        # 3. PENALTIES
        drawdown = (self.max_balance - self.balance) / (self.max_balance + 1e-9)
        if drawdown > 0.12: reward -= 0.2 # Safety check
        
        # Inactivity Penalty (If valid signal exists but RL chooses tiny risk)
        if self.position == 0 and valid_p > 0.8:
            reward -= 0.02

        self.current_step += 1
        self.max_balance = max(self.max_balance, self.balance)
        done = self.balance <= 10 or self.current_step >= len(self.X) - 1
        
        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.balance = self.initial_balance; self.max_balance = self.initial_balance
        self.current_step = 0; self.position = 0; self.entry_price = 0; self.current_lot = 0
        return self._get_obs(), {}

# ==========================================
# UPDATED MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    X_tr_f, y_tr_f, X_te, y_te, df_train_meta, test_meta = preprocess_gold_data("XAUUSD_M5_2Year.csv", "XAUUSD_M5_6month.csv")
    in_dim = X_tr_f.shape[2]

    model_a = ModelA_Base(in_dim).to(device); model_a.load_state_dict(torch.load('best_model_a.pth'))
    model_b = ModelB_TCN(in_dim).to(device); model_b.load_state_dict(torch.load('best_model_b.pth'))

    print("[*] Training Continuous Risk-Scaler (200,000 steps)...")
    env = GoldTradingEnv(X_tr_f, df_train_meta, model_a, model_b)
    # We use MlpPolicy with a lower learning rate for continuous control
    rl_agent = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0001, n_steps=2048, batch_size=64)
    rl_agent.learn(total_timesteps=200000)
    rl_agent.save("fyp_gold_risk_scaler")

    # --- FINAL HYBRID BACKTEST ---
    print("[*] Generating Performance Analysis...")
    test_env = GoldTradingEnv(X_te, test_meta, model_a, model_b)
    obs, _ = test_env.reset()
    history, trades = [100.0], []
    
    for i in range(len(X_te)-1):
        action, _ = rl_agent.predict(obs, deterministic=True)
        old_pos = test_env.position
        old_lot = test_env.current_lot
        obs, reward, done, _, _ = test_env.step(action)
        history.append(test_env.balance)
        
        if old_pos == 0 and test_env.position != 0:
            trades.append({'time': test_meta.iloc[i]['time'], 'type': 'BUY' if test_env.position==1 else 'SELL', 'price': test_meta.iloc[i]['close'], 'lots': test_env.current_lot})
        elif old_pos != 0 and test_env.position == 0:
            pnl = (test_meta.iloc[i]['close'] - trades[-1]['price']) * (1 if trades[-1]['type']=='BUY' else -1) * trades[-1]['lots'] * 100
            trades[-1].update({'exit_time': test_meta.iloc[i]['time'], 'exit_price': test_meta.iloc[i]['close'], 'pnl': pnl})
        if done: break

    # Final table printing logic (Same as your previous request)
    trade_log = pd.DataFrame([t for t in trades if 'pnl' in t])
    trade_log.to_csv('fyp_rl_trade_log.csv', index=False)

    # Calculation for Table
    final_equity = test_env.balance
    net_profit = final_equity - 100
    win_rate = (len(trade_log[trade_log['pnl'] > 0]) / len(trade_log) * 100) if len(trade_log) > 0 else 0
    returns = pd.Series(history).pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    sharpe = (returns.mean() / (returns.std() + 1e-9) * np.sqrt(288 * 252))
    max_dd = ((pd.Series(history).cummax() - pd.Series(history)) / (pd.Series(history).cummax() + 1e-9)).max()

    print("\n╔" + "═"*54 + "╗")
    print("║            FYP HYBRID RL+DL SYSTEM ANALYSIS          ║")
    print("╠" + "═"*54 + "╣")
    print(f"║  Period: {test_meta['time'].iloc[0].date()} to {test_meta['time'].iloc[-1].date()}    ║")
    print(f"║  Test Duration:   5.78 Months                        ║")
    print("╠" + "═"*54 + "╣")
    print(f"║  [RL] ASSISTED POSITION STRATEGY                     ║")
    print(f"║  Final Equity:    ${final_equity:<10.2f}                        ║")
    print(f"║  Net Profit:      ${net_profit:<10.2f}                        ║")
    print(f"║  Max Drawdown:    -{max_dd*100:<10.2f}%                       ║")
    print(f"║  Sharpe Ratio:    {sharpe:<10.2f}                         ║")
    print("╠" + "═"*54 + "╣")
    print(f"║  Total Trades:    {len(trade_log):<10}                         ║")
    print(f"║  Win Rate:        {win_rate:<10.2f}%                        ║")
    print("╚" + "═"*54 + "╝")

    plt.figure(figsize=(12,6))
    plt.plot(history, label="RL-Assisted Hybrid Equity", color='gold')
    plt.title("Final FYP Result: Hybrid DL+RL Portfolio Performance")
    plt.xlabel("Step")
    plt.ylabel("Balance ($)")
    plt.legend()
    plt.savefig('fyp_final_rl_result.png')
    print("[+] Project Complete. Equity curve saved.")