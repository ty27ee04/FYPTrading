import time
from datetime import datetime
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import onnxruntime as ort
import joblib
import requests
import logging
import traceback
import os
from dotenv import load_dotenv

# ==========================================
# 0. SYSTEM LOGGING SETUP
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"), # Saves to text file
        logging.StreamHandler()                 # Also prints to the console
    ]
)

logging.getLogger("urllib3").setLevel(logging.WARNING)

# ==========================================
# 1. LIVE TRADING CONFIGURATION (Master Menu)
# ==========================================
SYMBOL = "XAUUSD.m"  # Update this to match your exact broker symbol
TIMEFRAME = mt5.TIMEFRAME_M5
LOOKBACK = 60
REQUIRED_CANDLES = 300 

# --- Strategy Controls ---
STRATEGY_TYPE = "FIXED"            # Options: "FIXED" or "DYNAMIC"
ALLOW_MULTIPLE_TRADES = False      # False = Match Python backtest. True = Pyramiding/Overlapping.

# --- Lot Size Settings ---
FIXED_LOT_SIZE = 0.01              # Active if STRATEGY_TYPE = "FIXED"
DYN_STEP_EQUITY = 100.0            # Active if STRATEGY_TYPE = "DYNAMIC" (Every $X Equity...)
DYN_STEP_LOT = 0.01                # Active if STRATEGY_TYPE = "DYNAMIC" (...Trade Y Lots)
MAX_LOT_SAFETY = 50.0              # Broker Maximum Lot Cap

# --- Risk Management ---
TP_MULT = 3.0
SL_MULT = 2.0
MAGIC_NUMBER = 2026

# ==========================================
# 1.5 TELEGRAM NOTIFICATION SYSTEM
# ==========================================
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    logging.error("CRITICAL: Telegram credentials missing! Please check your .env file.")

active_trade_tickets = set()    

def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"  
    }
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        logging.error(f"Failed to send Telegram alert: {e}")

# ==========================================
# 2. INITIALIZATION & MODEL LOADING
# ==========================================
logging.info("Initializing MT5 Connection...")
if not mt5.initialize():
    logging.error(f"MT5 Initialize failed, error code = {mt5.last_error()}")
    quit()

logging.info(f"Connected to MT5. Broker: {mt5.account_info().company}")

logging.info("Loading AI Models & Scaler...")
try:
    scaler = joblib.load('scaler.pkl')
    session_a = ort.InferenceSession('best_model_a_live.onnx')
    session_b = ort.InferenceSession('best_model_b_live.onnx')
    logging.info("ONNX Models & Scaler loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load files: {e}")
    mt5.shutdown()
    quit()

# ==========================================
# 3. LIVE PREPROCESSING
# ==========================================
def denoise_series(series, span=5):
    return series.ewm(span=span, adjust=False).mean()

def get_live_tensor():
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, REQUIRED_CANDLES)
    if rates is None or len(rates) < REQUIRED_CANDLES:
        logging.warning("Failed to retrieve sufficient live market data.")
        return None, None
        
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    df['close_smooth'] = denoise_series(df['close'])
    df['hour'] = df['time'].dt.hour

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

    df['log_ret'] = np.log(df['close_smooth'] / df['close_smooth'].shift(1))
    df['atr_p'] = df['atr'] / df['close_smooth']
    df['sin_h'], df['cos_h'] = np.sin(2*np.pi*df['hour']/24), np.cos(2*np.pi*df['hour']/24)
    
    df = df.dropna().reset_index(drop=True)
    
    live_atr = df['atr'].iloc[-1]
    
    feat_cols = ['log_ret', 'rsi_n', 'mfi_n', 'atr_p', 'vol_filter', 'sin_h', 'cos_h', 'h1_trend_slope', 'rsi_h1']
    X_scaled = scaler.transform(df[feat_cols])
    
    live_sequence = X_scaled[-LOOKBACK:]
    tensor = np.expand_dims(live_sequence, axis=0).astype(np.float32)
    return tensor, live_atr

# ==========================================
# 3.5 REAL-TIME TRADE MONITOR
# ==========================================
def check_closed_trades():
    global active_trade_tickets
    
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return
        
    current_tickets = {p.ticket for p in positions}
    closed_tickets = active_trade_tickets - current_tickets
    
    for ticket in closed_tickets:
        deals = mt5.history_deals_get(position=ticket)
        if deals and len(deals) > 1:
            out_deal = deals[-1]
            profit = out_deal.profit
            price = out_deal.price
            
            icon = "🟢" if profit > 0 else "🔴"
            result_text = "PROFIT" if profit > 0 else "LOSS"
            
            close_msg = (
                f"{icon} **TRADE CLOSED ({result_text})** {icon}\n\n"
                f"💎 **Asset:** {SYMBOL}\n"
                f"🎫 **Ticket:** `#{ticket}`\n"
                f"💲 **Close Price:** `{price:.2f}`\n"
                f"💵 **Net PnL:** `${profit:.2f}`\n"
                f"🏦 **New Equity:** `${mt5.account_info().equity:,.2f}`"
            )
            logging.info(f"Trade Closed: #{ticket} | PnL: ${profit:.2f}")
            send_telegram_alert(close_msg)
            
    active_trade_tickets = current_tickets

# ==========================================
# 4. EXECUTION & BROKER ROUTING
# ==========================================
def execute_trade(action, atr):
    if not ALLOW_MULTIPLE_TRADES:
        positions = mt5.positions_get(symbol=SYMBOL)
        if positions is not None and len(positions) > 0:
            logging.info("Trade already active (Execution Lock ON). Skipping signal.")
            return

    equity = mt5.account_info().equity

    if STRATEGY_TYPE == "FIXED":
        lot_size = FIXED_LOT_SIZE
    elif STRATEGY_TYPE == "DYNAMIC":
        raw_lot = (equity / DYN_STEP_EQUITY) * DYN_STEP_LOT
        lot_size = round(raw_lot, 2)
    else:
        logging.error("Invalid STRATEGY_TYPE defined. Aborting trade.")
        return
        
    lot_size = max(0.01, min(lot_size, MAX_LOT_SAFETY))
    
    tick = mt5.symbol_info_tick(SYMBOL)
    
    if action == 1: # BUY
        price = tick.ask
        sl = price - (atr * SL_MULT)
        tp = price + (atr * TP_MULT)
        order_type = mt5.ORDER_TYPE_BUY
    elif action == -1: # SELL
        price = tick.bid
        sl = price + (atr * SL_MULT)
        tp = price - (atr * TP_MULT)
        order_type = mt5.ORDER_TYPE_SELL
        
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "AI Hybrid Execution",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.warning(f"Order Failed. MT5 Error Code: {result.retcode}")
        send_telegram_alert(f"⚠️ **TRADE FAILED** ⚠️\nError Code: {result.retcode}")
    else:
        direction = "🟢 BUY" if action == 1 else "🔴 SELL"
        
        alert_msg = (
            f"🚨 **AI TRADE EXECUTED** 🚨\n\n"
            f"📈 **Direction:** {direction}\n"
            f"💎 **Asset:** {SYMBOL}\n"
            f"📊 **Volume:** {lot_size} Lots\n"
            f"💲 **Entry Price:** `{price:.2f}`\n"
            f"🛑 **Stop Loss:** `{sl:.2f}`\n"
            f"🎯 **Take Profit:** `{tp:.2f}`\n"
            f"💰 **Running Equity:** `${equity:,.2f}`\n"
            f"🎫 **Ticket:** `#{result.order}`\n\n"
            f"🤖 *CNN-LSTM + TCN Hybrid System*"
        )
        
        logging.info(f"$$$ {direction} EXECUTED! Ticket: {result.order} | Volume: {lot_size}")
        send_telegram_alert(alert_msg)

# ==========================================
# 5. THE ACTIVE POLLING LOOP 
# ==========================================
logging.info("\n[*] Starting Live AI Execution Loop (With Real-Time Monitoring)...")

try:
    initial_positions = mt5.positions_get(symbol=SYMBOL)
    if initial_positions:
        active_trade_tickets = {p.ticket for p in initial_positions}

    last_processed_candle = None  # <--- MEMORY FOR THE STALE CHECK

    while True:
        current_time = time.time()
        
        check_closed_trades()
        
        seconds_past_candle = current_time % 300
        
        if 2 <= seconds_past_candle < 3: 
            
            # --- STALE CANDLE CHECK (HOLIDAY/WEEKEND PROTECTION) ---
            latest_rate = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 1)
            if latest_rate is not None and len(latest_rate) > 0:
                current_candle_time = latest_rate[0]['time']
                
                if current_candle_time == last_processed_candle:
                    logging.info("Market is closed/paused (Stale candle detected). Sleeping until next cycle...")
                    time.sleep(2)
                    continue  # Abort the loop here and wait for the next 5 minutes
                    
                # If it's a fresh candle, update our memory and proceed!
                last_processed_candle = current_candle_time
            # -------------------------------------------------------

            logging.info("Processing New M5 Candle...")
            
            input_tensor, current_atr = get_live_tensor()
            if input_tensor is not None:
                logits_a = session_a.run(None, {'input': input_tensor})[0]
                sig_a = np.argmax(logits_a, axis=1)[0]
                
                logits_b = session_b.run(None, {'input': input_tensor})[0]
                exp_b = np.exp(logits_b - np.max(logits_b, axis=1, keepdims=True))
                prob_b = (exp_b / np.sum(exp_b, axis=1, keepdims=True))[0][1]
                
                logging.info(f"    -> Model A: {sig_a} | Model B Gatekeeper: {prob_b*100:.2f}%")
                
                if prob_b > 0.52 and sig_a in [1, 2]:
                    trade_action = 1 if sig_a == 1 else -1
                    logging.info("    -> AI THRESHOLD MET. Initiating Trade Routing...")
                    execute_trade(trade_action, current_atr)
                else:
                    logging.info("    -> Signal Rejected/Hold. Waiting for next candle.")
                    
            time.sleep(2)
            
        time.sleep(0.5)

except KeyboardInterrupt:
    logging.info("Terminating Live Execution Manually...")

except Exception as e:
    error_traceback = traceback.format_exc()
    logging.error(f"CRITICAL BOT CRASH:\n{error_traceback}")
    
    sos_msg = (
        f"💥 **CRITICAL BOT CRASH** 💥\n\n"
        f"The Python execution script has encountered a fatal error and shut down.\n\n"
        f"**Error Details:**\n`{str(e)}`\n\n"
        f"⚠️ *Please check trading_bot.log on your server immediately!*"
    )
    send_telegram_alert(sos_msg)

finally:
    mt5.shutdown()
    logging.info("MT5 Connection safely closed.")