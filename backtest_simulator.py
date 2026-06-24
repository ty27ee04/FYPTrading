import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import joblib
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Initializing Simulator on device: {device}")

# ==========================================
# 1. CORE ARCHITECTURE (REQUIRED FOR LOADING)
# ==========================================
class AttentionLayer(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.w = nn.Linear(hid_dim, 1, bias=False)
    def forward(self, x):
        weights = F.softmax(self.w(torch.tanh(x)), dim=1)
        return torch.sum(x * weights, dim=1), weights

class ModelA_Base(nn.Module):
    def __init__(self, in_dim, hid_dim):
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
                nn.ConstantPad1d(( (kernel_size-1) * dilation_size, 0), 0),
                nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ]
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(num_channels[-1], 2)
    def forward(self, x):
        x = self.network(x.permute(0, 2, 1))
        return self.classifier(x[:, :, -1])

# ==========================================
# 2. INFERENCE PREPROCESSING 
# ==========================================
def denoise_series(series, span=5):
    return series.ewm(span=span, adjust=False).mean()

def load_and_preprocess_test_data(test_path, lookback=60, max_horizon=24, pt_mult=3.0, sl_mult=2.0):
    if not os.path.exists(test_path): 
        raise FileNotFoundError(f"Test file {test_path} not found.")
        
    df = pd.read_csv(test_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    df = df.drop(columns=['spread', 'real_volume'], errors='ignore')
    
    df['close_smooth'] = denoise_series(df['close'])
    df['hour'] = df['time'].dt.hour
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

    delta = df['close_smooth'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['rsi_n'] = (100 - (100 / (1 + (gain / (loss + 1e-9))))) / 100.0

    tp = (df['high'] + df['low'] + df['close_smooth']) / 3
    rmf = tp * df['tick_volume']
    df['mfi_n'] = (100 - (100 / (1 + (rmf.where(tp > tp.shift(1), 0).rolling(14).sum() / (rmf.where(tp < tp.shift(1), 0).rolling(14).sum() + 1e-9))))) / 100.0
    
    h_l, h_pc, l_pc = df['high']-df['low'], (df['high']-df['close'].shift(1)).abs(), (df['low']-df['close'].shift(1)).abs()
    df['atr'] = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1).rolling(window=14).mean()
    df['vol_filter'] = df['atr'] / (df['atr'].rolling(window=288).mean() + 1e-9)

    df['ma_h1'] = df['close_smooth'].rolling(window=12).mean()
    df['h1_trend_slope'] = (df['ma_h1'] - df['ma_h1'].shift(12)) / (df['ma_h1'].shift(12) + 1e-9)
    gain_h1 = (delta.where(delta > 0, 0)).rolling(window=168).mean()
    loss_h1 = (-delta.where(delta < 0, 0)).rolling(window=168).mean()
    df['rsi_h1'] = (100 - (100 / (1 + (gain_h1 / (loss_h1 + 1e-9))))) / 100.0

    df = df.dropna().reset_index(drop=True)

    df['log_ret'] = np.log(df['close_smooth'] / df['close_smooth'].shift(1))
    df['atr_p'] = df['atr'] / df['close_smooth']
    df['sin_h'], df['cos_h'] = np.sin(2*np.pi*df['hour']/24), np.cos(2*np.pi*df['hour']/24)
    
    df = df.dropna().reset_index(drop=True)
    
    # Load existing scaler (Do NOT fit a new one)
    feat_cols = ['log_ret', 'rsi_n', 'mfi_n', 'atr_p', 'vol_filter', 'sin_h', 'cos_h', 'h1_trend_slope', 'rsi_h1']
    scaler = joblib.load('scaler.pkl')
    X_te_s = scaler.transform(df[feat_cols])
    
    X = []
    for i in range(len(X_te_s) - lookback):
        X.append(X_te_s[i:i+lookback])
        
    # We don't need actual labels for pure backtesting simulation
    test_meta = df.iloc[lookback:].reset_index(drop=True)
    return np.array(X), test_meta, len(feat_cols)

# ==========================================
# 3. UPGRADED BACKTEST ENGINE
# ==========================================
def run_scenario_backtest(df, preds, scenario_name, initial_equity, strategy_type, 
                          fixed_lot=0.01, dyn_step_equity=100, dyn_step_lot=0.01, 
                          pt_mult=3.0, sl_mult=2.0, max_horizon=24, spread_penalty=0.20):
    """
    strategy_type: 'FIXED' or 'DYNAMIC'
    dyn_step_equity & dyn_step_lot: Example: for every $100, trade 0.01 lots.
    """
    df = df.copy()
    contract_size = 100 
    equity = initial_equity
    equity_history = [initial_equity]
    
    trades = []
    in_trade = False
    trade_type, entry_price, entry_idx, entry_atr, lot_size_at_entry = 0, 0, 0, 0, 0
    
    for i in range(1, len(df)):
        if in_trade:
            high, low, close = df['high'].iloc[i], df['low'].iloc[i], df['close'].iloc[i]
            bars_held = i - entry_idx
            
            exit_triggered, exit_price, exit_reason = False, 0, ""
            
            if trade_type == 1: 
                tp, sl = entry_price + (pt_mult * entry_atr), entry_price - (sl_mult * entry_atr)
                if high >= tp and low <= sl:
                    exit_triggered, exit_price, exit_reason = True, sl, "Stop Loss (Same Bar Conflict)"
                elif high >= tp: exit_triggered, exit_price, exit_reason = True, tp, "Take Profit"
                elif low <= sl: exit_triggered, exit_price, exit_reason = True, sl, "Stop Loss"
                elif bars_held >= max_horizon: exit_triggered, exit_price, exit_reason = True, close, "Time Stop"
                    
            elif trade_type == -1: 
                tp, sl = entry_price - (pt_mult * entry_atr), entry_price + (sl_mult * entry_atr)
                if low <= tp and high >= sl:
                    exit_triggered, exit_price, exit_reason = True, sl, "Stop Loss (Same Bar Conflict)"
                elif low <= tp: exit_triggered, exit_price, exit_reason = True, tp, "Take Profit"
                elif high >= sl: exit_triggered, exit_price, exit_reason = True, sl, "Stop Loss"
                elif bars_held >= max_horizon: exit_triggered, exit_price, exit_reason = True, close, "Time Stop"

            if exit_triggered:
                p_diff_raw = (exit_price - entry_price) * trade_type
                p_diff_net = p_diff_raw - spread_penalty
                
                spread_cost = spread_penalty * lot_size_at_entry * contract_size
                pnl = p_diff_net * lot_size_at_entry * contract_size
                
                equity += pnl
                
                trades.append({
                    'Scenario': scenario_name,
                    'Entry_Time': df['time'].iloc[entry_idx], 'Exit_Time': df['time'].iloc[i],
                    'Direction': 'Long' if trade_type==1 else 'Short',
                    'Lot_Size': lot_size_at_entry, 'Net_PnL': pnl, 'Running_Equity': equity
                })
                in_trade = False
                
        if not in_trade:
            signal = preds[i]
            if signal == 1 or signal == 2:
                # If equity drops below 0, margin call (account blown)
                if equity <= 0:
                    break 

                in_trade = True
                trade_type = 1 if signal == 1 else -1
                entry_price = df['open'].iloc[i] 
                entry_idx = i
                entry_atr = df['atr'].iloc[i-1] 
                
                if strategy_type == 'FIXED':
                    lot_size_at_entry = fixed_lot
                else:
                    # DYNAMIC LOT CALCULATION
                    raw_dyn_lot = (equity / dyn_step_equity) * dyn_step_lot
                    lot_size_at_entry = np.clip(round(raw_dyn_lot, 2), 0.01, 10.0)

        equity_history.append(equity)

    trade_log = pd.DataFrame(trades)
    
    # Calculate Metrics
    if len(equity_history) > 0:
        cum_max = np.maximum.accumulate(equity_history)
        drawdown = (equity_history - cum_max) / (cum_max + 1e-9)
        max_dd = max(drawdown.min(), -1.0) * 100
    else:
        max_dd = 0
        
    def calculate_sharpe(eq_list):
        rets = pd.Series(eq_list).pct_change().dropna()
        if len(rets) == 0 or np.std(rets) == 0: return 0
        return (np.mean(rets) / np.std(rets)) * np.sqrt(288 * 252)

    win_rate = (len(trade_log[trade_log['Net_PnL'] > 0]) / len(trade_log) * 100) if len(trade_log) > 0 else 0
    net_profit = equity - initial_equity
    pct_increase = (net_profit / initial_equity) * 100

    return {
        'Scenario': scenario_name,
        'Initial_Eq': initial_equity,
        'Final_Eq': round(equity, 2),
        'Net_Profit': round(net_profit, 2),
        'Pct_Increase': round(pct_increase, 2),
        'Max_Drawdown_%': round(max_dd, 2),
        'Sharpe_Ratio': round(calculate_sharpe(equity_history), 2),
        'Win_Rate': round(win_rate, 2),
        'Total_Trades': len(trade_log),
        'Times': df['time'].values[:len(equity_history)], # Added for graphing
        'Equity_Curve': equity_history                    # Added for graphing
    }

# ==========================================
# 4. SCENARIO EXECUTION
# ==========================================
if __name__ == "__main__":
    print("[*] Processing Test Data (Fast Mode)...")
    X_te, test_meta, in_dim = load_and_preprocess_test_data("XAUUSD_M5_6month.csv")
    
    print("[*] Loading Pre-trained Models...")
    # Load Models (Must match your Optuna best hidden dimensions. Assume 64 based on your last run)
    model_a = ModelA_Base(in_dim, hid_dim=64).to(device)
    model_b = ModelB_TCN(in_dim).to(device)
    
    model_a.load_state_dict(torch.load('best_model_a.pth'))
    model_b.load_state_dict(torch.load('best_model_b.pth'))
    model_a.eval(); model_b.eval()

    print("[*] Generating AI Predictions...")
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_te)), batch_size=256, shuffle=False)
    final_preds = []
    
    with torch.no_grad():
        for batch in test_loader:
            bx = batch[0].to(device)
            sig_a = torch.argmax(model_a(bx), dim=1).cpu().numpy()
            prob_b = F.softmax(model_b(bx), dim=1)[:, 1].cpu().numpy()
            sig_final = np.where(prob_b > 0.52, sig_a, 0)
            final_preds.extend(sig_final)

    test_preds = np.array(final_preds)
    
    # --- DEFINE YOUR CUSTOM SCENARIOS HERE ---
    SCENARIOS = [
        {
            "name": "$1K Base (Fixed 0.10)", 
            "initial": 1000, "type": "FIXED", "fixed_lot": 0.10
        },
        {
            "name": "$100 Base (Fixed 0.01)", 
            "initial": 100, "type": "FIXED", "fixed_lot": 0.01
        },
        {
            "name": "$100 Base (Dynamic every $100 = 0.01 lots)", 
            "initial": 100, "type": "DYNAMIC", "dyn_equity": 100, "dyn_lot": 0.01
        }        
        # {
        #     "name": "$100 Micro (Fixed 0.01)", 
        #     "initial": 100, "type": "FIXED", "fixed_lot": 0.01
        # },
        # {
        #     "name": "$100 Aggressive Compounding", 
        #     "initial": 100, "type": "DYNAMIC", 
        #     "dyn_equity": 100, "dyn_lot": 0.01 # Every $100 = 0.01 lots
        # },
        # {
        #     "name": "$100 Safe Compounding", 
        #     "initial": 100, "type": "DYNAMIC", 
        #     "dyn_equity": 200, "dyn_lot": 0.01 # Every $200 = 0.01 lots
        # },
        # {
        #     "name": "$1000 Compounding", 
        #     "initial": 1000, "type": "DYNAMIC", 
        #     "dyn_equity": 1000, "dyn_lot": 0.01 # Every $1000 = 0.01 lots
        # },
        # {
        #     "name": "$1000 Aggressive Compounding", 
        #     "initial": 1000, "type": "DYNAMIC", 
        #     "dyn_equity": 100, "dyn_lot": 0.01 # Every $100 = 0.01 lots
        # },
        # {
        #     "name": "$200 Safe Compounding", 
        #     "initial": 200, "type": "DYNAMIC", 
        #     "dyn_equity": 200, "dyn_lot": 0.01 # Every $200 = 0.01 lots
        # }
    ]

    print("\n[*] Running Backtest Scenarios & Generating Graphs...")
    results = []
    
    for s in SCENARIOS:
        res = run_scenario_backtest(
            df=test_meta, preds=test_preds, scenario_name=s["name"], 
            initial_equity=s["initial"], strategy_type=s["type"],
            fixed_lot=s.get("fixed_lot", 0.01),
            dyn_step_equity=s.get("dyn_equity", 100),
            dyn_step_lot=s.get("dyn_lot", 0.01)
        )
        results.append(res)
        
        # --- DYNAMIC FILE NAMING LOGIC ---
        if s["type"] == "FIXED":
            # e.g., equity_100_f_0.01.png
            filename = f"equity_{s['initial']}_f_{s.get('fixed_lot', 0.01)}.png"
        else:
            # e.g., equity_100_d_200-0.01.png
            filename = f"equity_{s['initial']}_d_{s.get('dyn_equity', 100)}-{s.get('dyn_lot', 0.01)}.png"
            
        # --- GENERATE EQUITY CURVE GRAPH ---
        plt.figure(figsize=(10, 5))
        plt.plot(res['Times'], res['Equity_Curve'], color='blue' if s['type'] == 'FIXED' else 'orange', linewidth=2)
        plt.title(f"Equity Curve: {s['name']}")
        plt.xlabel('Time')
        plt.ylabel('Account Balance ($)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        print(f"[+] Saved graph: {filename}")
        
    # --- PRINT ENHANCED ANALYSIS TABLE ---
    results_df = pd.DataFrame(results)
    print("\n" + "="*125)
    print(f"{'SCENARIO NAME':<30} | {'START $':<8} | {'FINAL $':<10} | {'PROFIT $':<10} | {'INC %':<10} | {'MAX DD %':<8} | {'SHARPE':<7} | {'WIN %':<7}")
    print("-" * 125)
    for index, row in results_df.iterrows():
        profit_str = f"${row['Net_Profit']}"
        inc_str = f"+{row['Pct_Increase']}%"
        print(f"{row['Scenario']:<30} | ${row['Initial_Eq']:<7} | ${row['Final_Eq']:<9} | {profit_str:<10} | {inc_str:<10} | {row['Max_Drawdown_%']:<8} | {row['Sharpe_Ratio']:<7} | {row['Win_Rate']:<7}")
    print("="*125)

    # ==========================================
    # 5. MT5 STRATEGY TESTER INJECTOR EXPORT
    # ==========================================
    print("[*] Generating Signal Injector CSV for MT5 Strategy Tester...")
    
    mt5_signals = []
    for i in range(len(test_meta)):
        signal = test_preds[i]
        if signal == 1 or signal == 2:
            # We add 5 minutes to the timestamp because the signal is generated at the CLOSE 
            # of the current bar, and we want MT5 to execute on the OPEN of the next bar.
            exec_time = test_meta['time'].iloc[i] + pd.Timedelta(minutes=5)
            
            mt5_signals.append({
                'Time': exec_time.strftime('%Y.%m.%d %H:%M'),
                'Signal': 1 if signal == 1 else -1
            })

    mt5_df = pd.DataFrame(mt5_signals)
    mt5_df.to_csv('AI_Signals_XAUUSD.csv', index=False)
    print(f"[+] Saved {len(mt5_df)} signals to 'AI_Signals_XAUUSD.csv'")