import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from scipy.signal import savgol_filter
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import optuna
import time
from datetime import datetime

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. UTILITIES & DENOISING
# ==========================================
def denoise_series(series, span=5):
    """Applies Exponential Moving Average - 100% Causal (No Future Peeking)."""
    return series.ewm(span=span, adjust=False).mean()

# --- UPGRADE 1: FOCAL LOSS (Math Refined) ---
class FocalLoss(nn.Module):
    """
    Dynamically scales loss based on prediction confidence.
    Down-weights the easily classified 'Hold' (Class 0) signals to force 
    the model to focus on hard-to-predict Buy/Sell reversals.
    """
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha 

    def forward(self, inputs, targets):
        # Calculate raw cross entropy without weights first to get accurate probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        # Apply class weights mathematically AFTER the focal modulation
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# ==========================================
# 2. PREPROCESSING 
# ==========================================
# --- UPGRADE 2: ASYMMETRIC BARRIERS (pt_mult=3.0, sl_mult=2.0) ---
def preprocess_gold_data(train_path, test_path, lookback=60, max_horizon=24, pt_mult=3.0, sl_mult=2.0):
    def apply_tbm(path):
        if not os.path.exists(path): return None
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        df = df.drop(columns=['spread', 'real_volume'], errors='ignore')
        
        # Smooth 'close' price before calculating indicators
        df['close_smooth'] = denoise_series(df['close'])

        # Time-Of-Day Cyclical Encoding
        df['hour'] = df['time'].dt.hour
        df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Base Indicators
        delta = df['close_smooth'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['rsi_n'] = (100 - (100 / (1 + (gain / (loss + 1e-9))))) / 100.0

        tp = (df['high'] + df['low'] + df['close_smooth']) / 3
        rmf = tp * df['tick_volume']
        df['mfi_n'] = (100 - (100 / (1 + (rmf.where(tp > tp.shift(1), 0).rolling(14).sum() / (rmf.where(tp < tp.shift(1), 0).rolling(14).sum() + 1e-9))))) / 100.0
        
        # ATR 
        h_l, h_pc, l_pc = df['high']-df['low'], (df['high']-df['close'].shift(1)).abs(), (df['low']-df['close'].shift(1)).abs()
        df['atr'] = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1).rolling(window=14).mean()
        df['vol_filter'] = df['atr'] / (df['atr'].rolling(window=288).mean() + 1e-9)

        # --- UPGRADE 3: MULTI-TIMEFRAME (MTF) INJECTION ---
        # 1 Hour = 12 M5 bars. We calculate pseudo-H1 metrics to give the model macro-context.
        
        # 1. H1 Trend Moving Average & Slope
        df['ma_h1'] = df['close_smooth'].rolling(window=12).mean()
        df['h1_trend_slope'] = (df['ma_h1'] - df['ma_h1'].shift(12)) / (df['ma_h1'].shift(12) + 1e-9)

        # 2. H1 RSI Approximation (14 Hours = 168 M5 bars)
        gain_h1 = (delta.where(delta > 0, 0)).rolling(window=168).mean()
        loss_h1 = (-delta.where(delta < 0, 0)).rolling(window=168).mean()
        df['rsi_h1'] = (100 - (100 / (1 + (gain_h1 / (loss_h1 + 1e-9))))) / 100.0

        df = df.dropna().reset_index(drop=True)

        # Triple Barrier Labeling (Now mathematically asymmetric)
        # Triple Barrier Labeling (Corrected Asymmetric Logic)
        c, h, l, a = df['close'].values, df['high'].values, df['low'].values, df['atr'].values
        labels = np.zeros(len(df), dtype=int)
        
        for i in range(len(df) - max_horizon):
            # BUY SETUP: Profit Target = UP (+3 ATR), Stop Loss = DOWN (-2 ATR)
            buy_pt = c[i] + (pt_mult * a[i])
            buy_sl = c[i] - (sl_mult * a[i])
            
            # SELL SETUP: Profit Target = DOWN (-3 ATR), Stop Loss = UP (+2 ATR)
            sell_pt = c[i] - (pt_mult * a[i])
            sell_sl = c[i] + (sl_mult * a[i])
            
            # Find the exact moments the barriers are touched
            hit_buy_pt = np.where(h[i+1:i+1+max_horizon] >= buy_pt)[0]
            hit_buy_sl = np.where(l[i+1:i+1+max_horizon] <= buy_sl)[0]
            
            hit_sell_pt = np.where(l[i+1:i+1+max_horizon] <= sell_pt)[0]
            hit_sell_sl = np.where(h[i+1:i+1+max_horizon] >= sell_sl)[0]
            
            # Get the index of the first touch (999 if it never touches within horizon)
            idx_b_pt = hit_buy_pt[0] if len(hit_buy_pt) > 0 else 999
            idx_b_sl = hit_buy_sl[0] if len(hit_buy_sl) > 0 else 999
            
            idx_s_pt = hit_sell_pt[0] if len(hit_sell_pt) > 0 else 999
            idx_s_sl = hit_sell_sl[0] if len(hit_sell_sl) > 0 else 999
            
            # A success means hitting the Profit Target BEFORE the Stop Loss
            buy_success = idx_b_pt < idx_b_sl
            sell_success = idx_s_pt < idx_s_sl
            
            if buy_success and not sell_success:
                labels[i] = 1
            elif sell_success and not buy_success:
                labels[i] = 2
            elif buy_success and sell_success:
                # If market volatility is so high it hits both targets, favor the one that was hit FIRST
                labels[i] = 1 if idx_b_pt < idx_s_pt else 2
            else:
                labels[i] = 0
                
        df['label'] = labels
        df = df.iloc[:-max_horizon].copy()

        df['hour'] = df['time'].dt.hour
        df['sin_h'], df['cos_h'] = np.sin(2*np.pi*df['hour']/24), np.cos(2*np.pi*df['hour']/24)
        df['log_ret'] = np.log(df['close_smooth'] / df['close_smooth'].shift(1))
        df['atr_p'] = df['atr'] / df['close_smooth']
        
        return df.dropna().reset_index(drop=True)

    df_tr, df_te = apply_tbm(train_path), apply_tbm(test_path)
    
    # Feature list updated with MTF context
    feat_cols = ['log_ret', 'rsi_n', 'mfi_n', 'atr_p', 'vol_filter', 'sin_h', 'cos_h', 'h1_trend_slope', 'rsi_h1']
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(df_tr[feat_cols])
    X_te_s = scaler.transform(df_te[feat_cols])

    joblib.dump(scaler, 'scaler.pkl')
    print("[*] Scaler saved successfully from DL preprocessing.")
    
    def seq_gen(data, labels):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i+lookback]); y.append(labels[i+lookback])
        return np.array(X), np.array(y)
    
    X_tr, y_tr = seq_gen(X_tr_s, df_tr['label'].values)
    X_te, y_te = seq_gen(X_te_s, df_te['label'].values)
    return X_tr, y_tr, X_te, y_te, df_te.iloc[lookback:].reset_index(drop=True)

# ==========================================
# 3. MODELS: BASE (CNN-LSTM) & META (TCN)
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
# 4. AUTOMATED TUNING (OPTUNA)
# ==========================================
def objective(trial):
    hid_dim = trial.suggest_int('hid_dim', 64, 256, step=64)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    
    model = ModelA_Base(in_dim_global, hid_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Using Focal Loss for Optuna evaluation too
    class_weights = torch.tensor([0.2, 1.0, 1.0]).to(device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    model.train()
    for _ in range(2):
        for bx, by in t_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward(); optimizer.step()
            
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for vx, vy in v_loader:
            vx, vy = vx.to(device), vy.to(device)
            val_loss += criterion(model(vx), vy).item()
            
    return val_loss / len(v_loader)

# ==========================================
# 5. BACKTEST ENGINE
# ==========================================
def run_detailed_backtest(df, preds, initial_equity=10000, fixed_lot=0.10, pt_mult=3.0, sl_mult=2.0, max_horizon=24, spread_penalty=0.20):
    df = df.copy()
    contract_size = 100 
    
    equity_fixed = initial_equity
    equity_dynamic = initial_equity
    
    fixed_history = [initial_equity]
    dynamic_history = [initial_equity]
    
    trades = []
    
    # State tracking
    in_trade = False
    trade_type = 0 # 1 for Buy, -1 for Sell
    entry_price = 0
    entry_idx = 0
    entry_atr = 0
    dyn_lot_at_entry = 0
    
    for i in range(1, len(df)):
        # 1. CHECK EXITS IF WE ARE IN A TRADE
        if in_trade:
            high, low, close = df['high'].iloc[i], df['low'].iloc[i], df['close'].iloc[i]
            bars_held = i - entry_idx
            
            exit_triggered = False
            exit_price = 0
            exit_reason = ""
            
            if trade_type == 1: # LONG POSITIONS
                tp = entry_price + (pt_mult * entry_atr)
                sl = entry_price - (sl_mult * entry_atr)
                if high >= tp and low <= sl:
                    exit_triggered, exit_price, exit_reason = True, sl, "Stop Loss (Same Bar Conflict)"
                elif high >= tp: exit_triggered, exit_price, exit_reason = True, tp, "Take Profit"
                elif low <= sl: exit_triggered, exit_price, exit_reason = True, sl, "Stop Loss"
                elif bars_held >= max_horizon: exit_triggered, exit_price, exit_reason = True, close, "Time Stop"
                    
            elif trade_type == -1: # SHORT POSITIONS
                tp = entry_price - (pt_mult * entry_atr)
                sl = entry_price + (sl_mult * entry_atr)
                if low <= tp and high >= sl:
                    exit_triggered, exit_price, exit_reason = True, sl, "Stop Loss (Same Bar Conflict)"
                elif low <= tp: exit_triggered, exit_price, exit_reason = True, tp, "Take Profit"
                elif high >= sl: exit_triggered, exit_price, exit_reason = True, sl, "Stop Loss"
                elif bars_held >= max_horizon: exit_triggered, exit_price, exit_reason = True, close, "Time Stop"

            if exit_triggered:
                # Raw diffs and Spread deduction
                p_diff_raw = (exit_price - entry_price) * trade_type
                p_diff_net = p_diff_raw - spread_penalty
                
                # Calculate actual spread cost in Dollars
                spread_cost_dynamic = spread_penalty * dyn_lot_at_entry * contract_size
                
                # Calculate Net PnL
                pnl_fixed = p_diff_net * fixed_lot * contract_size
                pnl_dynamic = p_diff_net * dyn_lot_at_entry * contract_size
                
                equity_fixed += pnl_fixed
                equity_dynamic += pnl_dynamic
                
                # ENHANCED TRADE LOGGING
                trades.append({
                    'Trade_ID': len(trades) + 1,
                    'Entry_Time': df['time'].iloc[entry_idx], 
                    'Exit_Time': df['time'].iloc[i],
                    'Direction': 'Long' if trade_type==1 else 'Short',
                    'Entry_Price': round(entry_price, 3), 
                    'Exit_Price': round(exit_price, 3),
                    'Exit_Reason': exit_reason,
                    'Dynamic_Lot_Size': dyn_lot_at_entry,
                    'Spread_Charge_USD': round(spread_cost_dynamic, 2),
                    'Net_PnL_USD': round(pnl_dynamic, 2),
                    'Running_Equity_USD': round(equity_dynamic, 2)
                })
                in_trade = False
                
        # 2. CHECK ENTRIES IF WE ARE NOT IN A TRADE
        if not in_trade:
            signal = preds[i]
            if signal == 1 or signal == 2:
                in_trade = True
                trade_type = 1 if signal == 1 else -1
                entry_price = df['open'].iloc[i] # Enter on open of next bar
                entry_idx = i
                entry_atr = df['atr'].iloc[i-1] # Use ATR from signal bar
                
                # Calculate Dynamic Lot
                raw_dyn_lot = (equity_dynamic / 10000) * 0.1
                dyn_lot_at_entry = np.clip(round(raw_dyn_lot, 2), 0.01, 10.0)

        fixed_history.append(equity_fixed)
        dynamic_history.append(equity_dynamic)

    df['equity_fixed'] = fixed_history[:len(df)]
    df['equity_dynamic'] = dynamic_history[:len(df)]
    
    trade_log = pd.DataFrame(trades)
    
    def calculate_sharpe(equity_series):
        rets = equity_series.pct_change().dropna()
        if len(rets) == 0 or np.std(rets) == 0: return 0
        return (np.mean(rets) / np.std(rets)) * np.sqrt(288 * 252)

    def get_max_dd(series):
        cum_max = series.cummax()
        drawdown = ((series - cum_max) / (cum_max + 1e-9))
        return max(drawdown.min(), -1.0) 

    win_rate = (len(trade_log[trade_log['Net_PnL_USD'] > 0]) / len(trade_log) * 100) if len(trade_log) > 0 else 0
    
    return df, trade_log, {
        'initial': initial_equity,
        'final_fixed': equity_fixed,
        'final_dynamic': equity_dynamic,
        'max_dd_fixed': get_max_dd(df['equity_fixed']),
        'max_dd_dynamic': get_max_dd(df['equity_dynamic']),
        'sharpe_fixed': calculate_sharpe(df['equity_fixed']),
        'sharpe_dynamic': calculate_sharpe(df['equity_dynamic']),
        'num_trades': len(trade_log),
        'win_rate': win_rate
    }

# ==========================================
# 6. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    script_start_time = time.time()
    start_datetime = datetime.now()
    print(f"\n[*] Pipeline Execution Started: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")

    try:
        X_tr_f, y_tr_f, X_te, y_te, test_meta = preprocess_gold_data(
        "XAUUSD_M5_2Year.csv", "XAUUSD_M5_6month.csv"
        )
        in_dim_global = X_tr_f.shape[2]
    except Exception as e:
        print(f"[!] Error: {e}")
        exit()

    dataset = TensorDataset(torch.FloatTensor(X_tr_f), torch.LongTensor(y_tr_f))

    purge_gap = 24  
    train_idx = int(0.8 * len(dataset))

    t_set = torch.utils.data.Subset(dataset, range(0, train_idx - purge_gap))
    v_set = torch.utils.data.Subset(dataset, range(train_idx, len(dataset)))
    
    t_loader = DataLoader(t_set, batch_size=128, shuffle=True)
    v_loader = DataLoader(v_set, batch_size=128, shuffle=False)

    print("[*] Starting Optuna Study...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30) # Reduced to 30 for speed with MTF features
    print(f"[*] Best Hyperparams: {study.best_params}")

    torch.cuda.empty_cache() 

    # --- PHASE 2: TRAIN BASE MODEL (MODEL A) ---
    best_params = study.best_params
    model_a = ModelA_Base(in_dim_global, best_params['hid_dim']).to(device)
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=best_params['lr'], weight_decay=1e-4)
    
    # Applying Focal Loss natively during training
    class_weights = torch.tensor([0.2, 1.0, 1.0]).to(device)
    criterion_a = FocalLoss(alpha=class_weights, gamma=2.0)
    
    stopper = EarlyStopping(patience=15)

    print("[*] Training Model A (with Focal Loss)...")
    best_val_loss = float('inf')
    for epoch in range(100):
        model_a.train()
        for bx, by in t_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer_a.zero_grad()
            criterion_a(model_a(bx), by).backward(); optimizer_a.step()
        
        model_a.eval(); v_l = 0
        with torch.no_grad():
            for vx, vy in v_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_l += criterion_a(model_a(vx), vy).item()
        val_loss = v_l/len(v_loader)
        
        print(f"Epoch {epoch+1} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model_a.state_dict(), 'best_model_a.pth')
            print(f"[*] New Best Model A Saved (Loss: {val_loss:.4f})")

        stopper(val_loss)
        if stopper.early_stop: 
            print(f"[*] Early Stopping at Epoch {epoch+1}")
            break

    torch.cuda.empty_cache() 

    # --- PHASE 3: META-LABELING (MODEL B) ---
    model_a.load_state_dict(torch.load('best_model_a.pth'))
    model_a.eval()
    
    print("[*] Generating Meta-Labels for TCN in batches...")
    train_preds_list = []
    meta_label_gen_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr_f)), batch_size=512, shuffle=False)
    
    with torch.no_grad():
        for batch in meta_label_gen_loader:
            bx = batch[0].to(device)
            logits = model_a(bx)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            train_preds_list.extend(preds)
    
    train_preds = np.array(train_preds_list)
    meta_y = ((train_preds == y_tr_f) & (train_preds != 0)).astype(int)
    
    torch.cuda.empty_cache()

    meta_dataset = TensorDataset(torch.FloatTensor(X_tr_f), torch.LongTensor(meta_y))
    meta_loader = DataLoader(meta_dataset, batch_size=128, shuffle=True)
    
    model_b = ModelB_TCN(in_dim_global).to(device)
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=0.001)
    
    print(f"[*] Training TCN Gatekeeper (Model B) | Meta-Samples: {len(meta_y)}")
    meta_stopper = EarlyStopping(patience=8, min_delta=0.0005)
    best_meta_loss = float('inf') 
    
    for epoch in range(100):
        model_b.train()
        epoch_loss = 0
        for bx, by in meta_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer_b.zero_grad()
            loss = F.cross_entropy(model_b(bx), by)
            loss.backward()
            optimizer_b.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(meta_loader)
        print(f"Epoch {epoch+1} | Meta Loss: {avg_loss:.4f}")

        if avg_loss < best_meta_loss:
            best_meta_loss = avg_loss
            torch.save(model_b.state_dict(), 'best_model_b.pth')
            print(f"[*] Best Model B Saved (Loss: {avg_loss:.4f})")
        
        meta_stopper(avg_loss)
        if meta_stopper.early_stop:
            print(f"[*] Model B Early Stopping at Epoch {epoch}")
            break

    torch.cuda.empty_cache() 

    # --- PHASE 4: FINAL INFERENCE (HIERARCHICAL) ---
    print("[*] Loading Best Weights for Hierarchical Backtest...")
    model_a.load_state_dict(torch.load('best_model_a.pth'))
    model_b.load_state_dict(torch.load('best_model_b.pth')) 
    model_a.eval(); model_b.eval()
    
    final_preds = []
    test_loader = DataLoader(TensorDataset(torch.FloatTensor(X_te)), batch_size=256, shuffle=False)

    print("[*] Generating predictions ...")
    with torch.no_grad():
        for batch in test_loader:
            bx = batch[0].to(device)
            sig_a = torch.argmax(model_a(bx), dim=1).cpu().numpy()
            prob_b = F.softmax(model_b(bx), dim=1)[:, 1].cpu().numpy()
            
            sig_final = np.where(prob_b > 0.52, sig_a, 0)
            final_preds.extend(sig_final)

    test_preds = np.array(final_preds)
    print("[+] Test predictions complete.")
    
    print("\n[+] Classification Report:")
    print(classification_report(y_te, test_preds, target_names=['Hold', 'Buy', 'Sell'], zero_division=0))

    cm = confusion_matrix(y_te, test_preds)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['H','B','S'], yticklabels=['H','B','S'])
    plt.title('FYP: Confusion Matrix')
    plt.xlabel('Predicted Signal (AI Guess)')
    plt.ylabel('Original Signal (Market Actual)')
    plt.savefig('fyp_cm.png')

    res_df, trade_log, stats = run_detailed_backtest(test_meta, test_preds)

    start_d, end_d = res_df['time'].iloc[0], res_df['time'].iloc[-1]
    duration_months = (end_d - start_d).days / 30.44

    print("\n" + "╔══════════════════════════════════════════════════════╗")
    print("║            FYP HYBRID SIGNAL SYSTEM ANALYSIS         ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Period: {start_d.strftime('%Y-%m-%d')} to {end_d.strftime('%Y-%m-%d')}    ║")
    print(f"║  Test Duration:   {duration_months:.2f} Months                         ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  [1] FIXED LOT STRATEGY (Constant {0.1} Lots)         ║")
    print(f"║  Final Equity:    ${stats['final_fixed']:<10,.2f}                       ║")
    print(f"║  Net Profit:      ${(stats['final_fixed'] - stats['initial']):<10,.2f}                       ║")
    print(f"║  Max Drawdown:    {stats['max_dd_fixed']*100:<10.2f}%                       ║")
    print(f"║  Sharpe Ratio: {stats['sharpe_fixed']:<10.2f}                       ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  [2] AI DYNAMIC STRATEGY (Compounding)               ║")
    print(f"║  Final Equity:    ${stats['final_dynamic']:<10,.2f}                       ║")
    print(f"║  Net Profit:      ${(stats['final_dynamic'] - stats['initial']):<10,.2f}                       ║")
    print(f"║  Max Drawdown:    {stats['max_dd_dynamic']*100:<10.2f}%                       ║")
    print(f"║  Sharpe Ratio: {stats['sharpe_dynamic']:<10.2f}                       ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Total Trades:    {stats['num_trades']:<10}                       ║")
    print(f"║  Win Rate:        {stats['win_rate']:<10.2f}%                       ║")
    print("╚══════════════════════════════════════════════════════╝")

    if not trade_log.empty:
            trade_log.to_csv('fyp_xauusd_trade_log.csv', index=False)
            print(f"[*] Trade Log ({len(trade_log)} trades) saved to 'fyp_xauusd_trade_log.csv'")
    else:
        print("[!] Warning: No trades were recorded!")

    # ==========================================
    # 8. LIVE DEPLOYMENT EXPORT (ONNX)
    # ==========================================
    print("[*] Compiling Neural Networks to ONNX for Live Deployment...")
    
    # Create a dummy input tensor matching the M5 sequence shape (Batch, Lookback, Features)
    dummy_input = torch.randn(1, 60, in_dim_global).to(device)
    
    # Export the Base Model (CNN-LSTM-Attention)
    torch.onnx.export(model_a, dummy_input, "best_model_a_live.onnx", 
                      export_params=True, opset_version=18, 
                      do_constant_folding=True, 
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
                      
    # Export the Gatekeeper Model (TCN)
    torch.onnx.export(model_b, dummy_input, "best_model_b_live.onnx", 
                      export_params=True, opset_version=18, 
                      do_constant_folding=True, 
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
                      
    print("[+] ONNX export complete. Models are isolated and ready for MetaTrader5 / VPS integration.")

    print("[*] DL Execution Complete. Ready for RL pipeline.")

    # ==========================================
    # FINAL VISUALIZATION SUITE (Replacement Block)
    # ==========================================
    
    # 1. Helper Function for Dual-Axis Plots (Equity + Price)
    def plot_equity_vs_price(df, equity_col, title, filename, color='orange'):
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Axis 1: Strategy Equity (Left)
        ax1.set_xlabel('Date/Time')
        ax1.set_ylabel('Account Balance ($)', color=color, fontsize=12, fontweight='bold')
        ax1.plot(df['time'], df[equity_col], color=color, linewidth=2, label='Strategy Equity')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--', alpha=0.3)

        # Axis 2: Underlying Price (Right)
        ax2 = ax1.twinx() 
        ax2.set_ylabel('XAUUSD Price', color='gray', fontsize=12)
        ax2.plot(df['time'], df['close'], color='gray', alpha=0.4, label='XAUUSD Price')
        ax2.tick_params(axis='y', labelcolor='gray')

        plt.title(title, fontsize=14)
        fig.tight_layout()
        plt.savefig(filename)
        plt.close()

    # 2. Generate Chart 1: Fixed Lot vs Price Movement
    plot_equity_vs_price(res_df, 'equity_fixed', 
                         f'Fixed Lot Strategy (0.1) vs. Gold Price', 
                         'fyp_fixed_vs_price.png', color='blue')

    # 3. Generate Chart 2: Dynamic Lot vs Price Movement
    plot_equity_vs_price(res_df, 'equity_dynamic', 
                         'AI Dynamic Strategy vs. Gold Price', 
                         'fyp_dynamic_vs_price.png', color='orange')

    # 4. Generate Chart 3: Original Dual Equity (Log Scale for Comparison)
    plt.figure(figsize=(12, 6))
    plt.plot(res_df['time'], res_df['equity_fixed'], label=f'Fixed Lot (0.1)', color='blue', alpha=0.8)
    plt.plot(res_df['time'], res_df['equity_dynamic'], label='AI Dynamic (Compounding)', color='orange', linewidth=2)
    
    # Use Log Scale because Dynamic ($10M) is too large for Linear Scale
    plt.yscale('log') 
    
    plt.title('Final Performance Comparison (Log Scale)', fontsize=14)
    plt.ylabel('Account Balance ($) - Logarithmic Scale')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig('fyp_dual_equity.png')

    print("[*] All artifacts (3 charts, 1 log, 1 CM) saved successfully. Project Complete.")

    # PRO-TIP: Save the final stats too for your records
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv('fyp_final_stats.csv', index=False)

    print("[*] All artifacts saved successfully. Project Complete.")

    # --- END PIPELINE TIMER & SUMMARY ---
    script_end_time = time.time()
    end_datetime = datetime.now()
    elapsed_time = script_end_time - script_start_time
    
    # Calculate Hours, Minutes, Seconds
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║              PIPELINE EXECUTION SUMMARY              ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Start Time:    {start_datetime.strftime('%Y-%m-%d %H:%M:%S'):<36} ║")
    print(f"║  End Time:      {end_datetime.strftime('%Y-%m-%d %H:%M:%S'):<36} ║")
    print(f"║  Total Time:    {int(hours):02d}h {int(minutes):02d}m {seconds:05.2f}s{' ' * 22} ║")
    print("╚══════════════════════════════════════════════════════╝\n")