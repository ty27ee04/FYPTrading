import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import joblib
import optuna
import time
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. UTILITIES & METRICS
# ==========================================
def denoise_series(series, span=5):
    return series.ewm(span=span, adjust=False).mean()

def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

def calculate_mda(y_true, y_pred, y_baseline):
    true_direction = np.sign(y_true - y_baseline)
    pred_direction = np.sign(y_pred - y_baseline)
    valid_indices = true_direction != 0
    return np.mean(true_direction[valid_indices] == pred_direction[valid_indices]) * 100

def format_duration(seconds):
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{int(hours):02d}h {int(minutes):02d}m {secs:05.2f}s"

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
# 2. FIXED PREPROCESSING PIPELINE
# ==========================================
def preprocess_gold_regression(train_path, test_path, lookback=60, forecast_horizon=1):
    def extract_features(path):
        if not os.path.exists(path): 
            raise FileNotFoundError(f"Target data file not found at: {path}")
        
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)
        df = df.drop(columns=['spread', 'real_volume', 'label'], errors='ignore')
        
        df['close_smooth'] = denoise_series(df['close'])

        # Time Encoding
        df['hour'] = df['time'].dt.hour
        df['sin_h'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['cos_h'] = np.cos(2 * np.pi * df['hour'] / 24)

        # Technical Indicators
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
        df['atr_p'] = df['atr'] / df['close_smooth']
        df['log_ret'] = np.log(df['close_smooth'] / df['close_smooth'].shift(1))

        df['ma_h1'] = df['close_smooth'].rolling(window=12).mean()
        df['h1_trend_slope'] = (df['ma_h1'] - df['ma_h1'].shift(12)) / (df['ma_h1'].shift(12) + 1e-9)

        gain_h1 = (delta.where(delta > 0, 0)).rolling(window=168).mean()
        loss_h1 = (-delta.where(delta < 0, 0)).rolling(window=168).mean()
        df['rsi_h1'] = (100 - (100 / (1 + (gain_h1 / (loss_h1 + 1e-9))))) / 100.0

        # --- THE FIX: TARGET IS THE PRICE CHANGE (DELTA) ---
        df['baseline_price'] = df['close']
        df['target_delta'] = df['close'].shift(-forecast_horizon) - df['close']
        
        return df.dropna().reset_index(drop=True)

    df_tr = extract_features(train_path)
    df_te = extract_features(test_path)
    
    df_tr.to_csv("preprocessed_train_audit.csv", index=False)
    df_te.to_csv("preprocessed_test_audit.csv", index=False)
    
    feat_cols = ['log_ret', 'rsi_n', 'mfi_n', 'atr_p', 'vol_filter', 'sin_h', 'cos_h', 'h1_trend_slope', 'rsi_h1']
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_tr_s = scaler_X.fit_transform(df_tr[feat_cols])
    X_te_s = scaler_X.transform(df_te[feat_cols])
    
    y_tr_s = scaler_y.fit_transform(df_tr[['target_delta']]).squeeze(-1)
    y_te_s = scaler_y.transform(df_te[['target_delta']]).squeeze(-1)

    joblib.dump(scaler_X, 'scaler_X_regression.pkl')
    joblib.dump(scaler_y, 'scaler_y_regression.pkl')
    
    def generate_sequences(scaled_features, scaled_targets, baseline_prices):
        X, y, base = [], [], []
        for i in range(len(scaled_features) - lookback):
            X.append(scaled_features[i:i+lookback])
            y.append(scaled_targets[i+lookback])
            base.append(baseline_prices[i+lookback])
        return np.array(X), np.array(y), np.array(base)
    
    X_tr, y_tr, y_base_tr = generate_sequences(X_tr_s, y_tr_s, df_tr['baseline_price'].values)
    X_te, y_te, y_base_te = generate_sequences(X_te_s, y_te_s, df_te['baseline_price'].values)
    
    return X_tr, y_tr, y_base_tr, X_te, y_te, y_base_te, df_tr.iloc[lookback:].reset_index(drop=True), df_te.iloc[lookback:].reset_index(drop=True)

# ==========================================
# 3. ARCHITECTURES
# ==========================================
class AttentionLayer(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.w = nn.Linear(hid_dim, 1, bias=False)
    def forward(self, x):
        weights = F.softmax(self.w(torch.tanh(x)), dim=1)
        return torch.sum(x * weights, dim=1), weights

class ModelA_Regression(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super().__init__()
        self.cnn = nn.Conv1d(in_dim, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(64, hid_dim, batch_first=True, num_layers=2)
        self.attn = AttentionLayer(hid_dim)
        self.head = nn.Linear(hid_dim, 1) 
        
    def forward(self, x):
        x = F.relu(self.cnn(x.permute(0, 2, 1))).permute(0, 2, 1)
        out, _ = self.lstm(x)
        ctx, _ = self.attn(out)
        return self.head(ctx).squeeze(-1) 

class ModelB_TCNMetaErrorGatekeeper(nn.Module):
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
# 4. TUNING ENGINE
# ==========================================
def objective(trial):
    hid_dim = trial.suggest_int('hid_dim', 64, 256, step=64)
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    
    model = ModelA_Regression(in_dim_global, hid_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.HuberLoss() 
    
    model.train()
    for _ in range(2):
        for bx, by in t_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for vx, vy in v_loader:
            vx, vy = vx.to(device), vy.to(device)
            val_loss += criterion(model(vx), vy).item()
            
    return val_loss / len(v_loader)

# ==========================================
# 5. EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    global_start_time = time.time()
    print(f"\n[*] Regression Pipeline Execution Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    X_tr_f, y_tr_f, y_base_tr, X_te, y_te_s, y_base_te, train_meta, test_meta = preprocess_gold_regression(
        "XAUUSD_M5_2Year.csv", "XAUUSD_M5_6month.csv"
    )
    in_dim_global = X_tr_f.shape[2]

    dataset = TensorDataset(torch.FloatTensor(X_tr_f), torch.FloatTensor(y_tr_f))
    purge_gap = 24  
    train_idx = int(0.8 * len(dataset))

    t_set = torch.utils.data.Subset(dataset, range(0, train_idx - purge_gap))
    v_set = torch.utils.data.Subset(dataset, range(train_idx, len(dataset)))
    
    t_loader = DataLoader(t_set, batch_size=128, shuffle=True)
    v_loader = DataLoader(v_set, batch_size=128, shuffle=False)

    # --- PHASE 1: OPTUNA ---
    optuna_start = time.time()
    print("[*] Running Optuna Regression Parameter Sweep...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15)
    optuna_duration = time.time() - optuna_start
    print(f"[*] Best Hyperparams Found: {study.best_params} | Duration: {format_duration(optuna_duration)}")

    # --- PHASE 2: MODEL A TRAIN ---
    model_a_start = time.time()
    best_params = study.best_params
    model_a = ModelA_Regression(in_dim_global, best_params['hid_dim']).to(device)
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=best_params['lr'], weight_decay=1e-4)
    criterion_a = nn.MSELoss() 
    
    stopper = EarlyStopping(patience=15)
    best_val_loss = float('inf')
    
    model_a_train_history = []
    model_a_val_history = []

    print("[*] Training Base Regression Engine (Model A)...")
    for epoch in range(100):
        model_a.train()
        epoch_train_loss = 0
        for bx, by in t_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer_a.zero_grad()
            loss = criterion_a(model_a(bx), by)
            loss.backward()
            optimizer_a.step()
            epoch_train_loss += loss.item()
        
        model_a.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for vx, vy in v_loader:
                vx, vy = vx.to(device), vy.to(device)
                epoch_val_loss += criterion_a(model_a(vx), vy).item()
                
        avg_train_loss = epoch_train_loss / len(t_loader)
        avg_val_loss = epoch_val_loss / len(v_loader)
        
        model_a_train_history.append(avg_train_loss)
        model_a_val_history.append(avg_val_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  > Epoch {epoch+1:02d}/100 | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model_a.state_dict(), 'best_model_a_regression.pth')

        stopper(avg_val_loss)
        if stopper.early_stop: 
            print(f"[*] Early stopping triggered at epoch {epoch+1}")
            break
            
    model_a_duration = time.time() - model_a_start

    # --- PHASE 3: MODEL B TRAIN ---
    model_b_start = time.time()
    model_a.load_state_dict(torch.load('best_model_a_regression.pth'))
    model_a.eval()
    
    print("[*] Extracting training residuals to isolate variance maps...")
    train_preds_list = []
    meta_gen_loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr_f)), batch_size=512, shuffle=False)
    
    with torch.no_grad():
        for batch in meta_gen_loader:
            train_preds_list.extend(model_a(batch[0].to(device)).cpu().numpy())
            
    train_preds = np.array(train_preds_list)
    absolute_residuals = np.abs(y_tr_f - train_preds)
    error_threshold = np.percentile(absolute_residuals, 65)
    
    meta_y = (absolute_residuals <= error_threshold).astype(int)
    meta_dataset = TensorDataset(torch.FloatTensor(X_tr_f), torch.LongTensor(meta_y))
    meta_loader = DataLoader(meta_dataset, batch_size=128, shuffle=True)
    
    model_b = ModelB_TCNMetaErrorGatekeeper(in_dim_global).to(device)
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=0.001)
    
    print(f"[*] Training TCN Error Gatekeeper (Model B) | Samples: {len(meta_y)}")
    best_meta_loss = float('inf')
    meta_stopper = EarlyStopping(patience=8)
    model_b_history = []
    
    for epoch in range(50):
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
        model_b_history.append(avg_loss)
        
        if avg_loss < best_meta_loss:
            best_meta_loss = avg_loss
            torch.save(model_b.state_dict(), 'best_model_b_gatekeeper.pth')
            
        meta_stopper(avg_loss)
        if meta_stopper.early_stop: break
        
    model_b_duration = time.time() - model_b_start

    # --- PHASE 4: INVERSE-TRANSFORM RECONSTRUCTION ---
    model_a.load_state_dict(torch.load('best_model_a_regression.pth'))
    model_a.eval()
    
    scaler_y = joblib.load('scaler_y_regression.pkl')

    # 1. Generate Training Predictions for Visualization
    train_preds_list = []
    train_loader_nodes = DataLoader(TensorDataset(torch.FloatTensor(X_tr_f)), batch_size=256, shuffle=False)
    with torch.no_grad():
        for batch in train_loader_nodes:
            train_preds_list.extend(model_a(batch[0].to(device)).cpu().numpy())
            
    train_deltas_pred = scaler_y.inverse_transform(np.array(train_preds_list).reshape(-1, 1)).squeeze(-1)
    train_deltas_actual = scaler_y.inverse_transform(np.array(y_tr_f).reshape(-1, 1)).squeeze(-1)
    
    y_tr_unscaled = y_base_tr + train_deltas_actual
    train_preds_unscaled = y_base_tr + train_deltas_pred

    # 2. Generate Testing Predictions
    test_preds_list = []
    test_loader_nodes = DataLoader(TensorDataset(torch.FloatTensor(X_te)), batch_size=256, shuffle=False)
    with torch.no_grad():
        for batch in test_loader_nodes:
            test_preds_list.extend(model_a(batch[0].to(device)).cpu().numpy())
            
    test_deltas_pred = scaler_y.inverse_transform(np.array(test_preds_list).reshape(-1, 1)).squeeze(-1)
    test_deltas_actual = scaler_y.inverse_transform(np.array(y_te_s).reshape(-1, 1)).squeeze(-1)
    
    y_te_unscaled = y_base_te + test_deltas_actual
    test_preds_unscaled = y_base_te + test_deltas_pred
    
    # Compute Test Metrics for the summary panel
    r2 = r2_score(y_te_unscaled, test_preds_unscaled)
    mae = mean_absolute_error(y_te_unscaled, test_preds_unscaled)
    rmse = np.sqrt(mean_squared_error(y_te_unscaled, test_preds_unscaled))
    mape = calculate_mape(y_te_unscaled, test_preds_unscaled)
    mda = calculate_mda(y_te_unscaled, test_preds_unscaled, y_base_te)
    
    print("\n" + "╔══════════════════════════════════════════════════════╗")
    print("║          FYP PRICE REGRESSION MODEL EVALUATION       ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  R² Score (Accuracy Metric): {r2:<23.4f} ║")
    print(f"║  Mean Directional Accuracy:  {mda:<22.2f}% ║")
    print(f"║  Mean Absolute Error (MAE):  ${mae:<22.4f} ║")
    print(f"║  Root Mean Squared Error:    ${rmse:<22.4f} ║")
    print(f"║  Mean Absolute Pct Error:   {mape:<23.4f}% ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # --- PHASE 5: LOSS VISUALIZATION CURVES ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(model_a_train_history, label='Train MSE', color='#3498db', linewidth=2)
    ax1.plot(model_a_val_history, label='Val MSE', color='#e74c3c', linestyle='--', linewidth=2)
    ax1.set_title('Model A Regression Optimization Progress')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss Value (Delta Space MSE)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(model_b_history, label='Gatekeeper CE Loss', color='#2ecc71', linewidth=2)
    ax2.set_title('Model B (TCN Error Gatekeeper) Progress')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Cross-Entropy Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fyp_regression_training_progress.png', dpi=300)
    plt.close()

    # --- PHASE 6: FULL CHRONOLOGICAL COMPARATIVE CHART ---
    print("[*] Generating complete actual vs predicted timeline movement chart...")
    
    # Concatenate time series arrays sequentially
    full_time = pd.concat([train_meta['time'], test_meta['time']]).reset_index(drop=True)
    full_actual = np.concatenate([y_tr_unscaled, y_te_unscaled])
    full_predicted = np.concatenate([train_preds_unscaled, test_preds_unscaled])
    
    plt.figure(figsize=(16, 8))
    
    # Plot complete actual market line
    plt.plot(full_time, full_actual, label='Actual XAUUSD Price', color='#2c3e50', linewidth=2, alpha=0.9)
    
    # Plot complete model forecast line
    plt.plot(full_time, full_predicted, label='Hybrid Model Forecast', color='#e67e22', linestyle='--', linewidth=1.2, alpha=0.85)
    
    # Draw vertical boundary line dividing training phase from testing phase
    boundary_idx = len(train_meta)
    boundary_time = full_time.iloc[boundary_idx]
    plt.axvline(x=boundary_time, color='#c0392b', linestyle=':', linewidth=2.5, label='Out-of-Sample Test Boundary')
    
    # Label the two periods directly onto the canvas area
    plt.text(full_time.iloc[int(boundary_idx * 0.4)], max(full_actual) * 0.98, 'TRAINING PERIOD\n(In-Sample Optimization)', 
             fontsize=11, color='#2980b9', fontweight='bold', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.7))
    plt.text(full_time.iloc[int(boundary_idx + (len(test_meta) * 0.4))], max(full_actual) * 0.98, 'TESTING PERIOD\n(Out-of-Sample Evaluation)', 
             fontsize=11, color='#27ae60', fontweight='bold', horizontalalignment='center', bbox=dict(facecolor='white', alpha=0.7))

    plt.title('FYP Regression Target Space: Full Historic Timeline vs. Predicted Movements', fontsize=14, fontweight='bold')
    plt.xlabel('Timeline Sequence Horizon (M5 Intervals)', fontsize=12)
    plt.ylabel('Asset Value (USD Denominated)', fontsize=12)
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig('fyp_regression_comparative_movement.png', dpi=300)
    plt.close()
    print("[+] Complete historic timeline chart saved successfully to 'fyp_regression_comparative_movement.png'")

    global_duration = time.time() - global_start_time
    print("╔══════════════════════════════════════════════════════╗")
    print("║            PIPELINE COMPLETION CHRONOLOGY            ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  [1] Optuna Parameter Sweep:  {format_duration(optuna_duration):<22} ║")
    print(f"║  [2] Model A Train Execution: {format_duration(model_a_duration):<22} ║")
    print(f"║  [3] Model B Train Execution: {format_duration(model_b_duration):<22} ║")
    print(f"║  [4] Total Runtime Overhead:  {format_duration(global_duration):<22} ║")
    print("╚══════════════════════════════════════════════════════╝\n")