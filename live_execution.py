import time
from datetime import datetime, timezone
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

from strategy_config import FEATURE_COLUMNS, MODEL, STRATEGY, load_gatekeeper_threshold

# ==========================================
# 0. SYSTEM LOGGING SETUP
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log", encoding='utf-8'), # Saves to text file
        logging.StreamHandler()                 # Also prints to the console
    ]
)

logging.getLogger("urllib3").setLevel(logging.WARNING)

# ==========================================
# 1. LIVE TRADING CONFIGURATION (Master Menu)
# ==========================================
SYMBOL = "XAUUSD.m"  # Update this to match your exact broker symbol
TIMEFRAME = mt5.TIMEFRAME_M5
LOOKBACK = MODEL.lookback
REQUIRED_CANDLES = 400 

# --- Strategy Controls ---
STRATEGY_TYPE = "FIXED"            # Options: "FIXED" or "DYNAMIC"
ALLOW_MULTIPLE_TRADES = STRATEGY.allow_multiple_trades

# --- Lot Size Settings ---
FIXED_LOT_SIZE = 0.01              # Active if STRATEGY_TYPE = "FIXED"
DYN_STEP_EQUITY = 100.0            # Active if STRATEGY_TYPE = "DYNAMIC" (Every $X Equity...)
DYN_STEP_LOT = 0.01                # Active if STRATEGY_TYPE = "DYNAMIC" (...Trade Y Lots)
MAX_LOT_SAFETY = 50.0              # Broker Maximum Lot Cap

# --- Risk Management ---
TP_MULT = STRATEGY.take_profit_atr
SL_MULT = STRATEGY.stop_loss_atr
MAGIC_NUMBER = 2026

# --- AI Settings ---
GATEKEEPER_THRESHOLD = load_gatekeeper_threshold()
IS_BOT_ACTIVE = True               # <--- NEW: Master switch for Pause/Resume
WEEKEND_PROTECTION = True          # <--- NEW: Enable Friday Flat protocol
FRIDAY_LIQ_HOUR = 23               # <--- NEW: MT5 Server Hour to liquidate (23 = 11 PM)
FRIDAY_LIQ_MINUTE = 50             # <--- NEW: MT5 Server Minute to liquidate

# --- Trade Management ---
USE_BREAK_EVEN = STRATEGY.use_break_even
BE_TRIGGER = STRATEGY.break_even_trigger
BE_BUFFER = STRATEGY.break_even_buffer

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

telegram_offset = None  # Remembers the last message read so we don't process it twice

def check_telegram_commands():
    # 1. Bring ALL Master Menu variables into the global scope
    global GATEKEEPER_THRESHOLD, telegram_offset, IS_BOT_ACTIVE
    global STRATEGY_TYPE, ALLOW_MULTIPLE_TRADES
    global FIXED_LOT_SIZE, DYN_STEP_EQUITY, DYN_STEP_LOT, MAX_LOT_SAFETY
    global TP_MULT, SL_MULT # <--- NEW: Added TP and SL to global scope
    global USE_BREAK_EVEN, BE_TRIGGER, BE_BUFFER # <--- NEW: Break-Even Globals
    
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    params = {"timeout": 1, "allowed_updates": ["message"]}
    if telegram_offset:
        params["offset"] = telegram_offset
        
    try:
        response = requests.get(url, params=params, timeout=2)
        data = response.json()
        
        if not data.get("ok"):
            return
            
        for result in data.get("result", []):
            telegram_offset = result["update_id"] + 1
            message = result.get("message", {})
            chat_id = str(message.get("chat", {}).get("id", ""))
            text = message.get("text", "").strip()
            
            if chat_id != str(TELEGRAM_CHAT_ID):
                continue
                
            # --- COMMAND ROUTER ---
            text_lower = text.lower()

            if text_lower.startswith("/kill"):
                send_telegram_alert("💀 **EMERGENCY KILL SWITCH** 💀\nSevering MT5 connection and shutting down Python script...")
                logging.info("Remote /kill command executed. Exiting script.")
                mt5.shutdown()
                os._exit(0) # Immediately kills the CMD process

            elif text_lower.startswith("/stop"):
                IS_BOT_ACTIVE = False
                send_telegram_alert("⏸️ **BOT PAUSED** ⏸️\nThe AI will ignore the market until you type `/start`.")
                logging.info("C2 Update: Bot state set to PAUSED.")

            elif text_lower.startswith("/start"):
                IS_BOT_ACTIVE = True
                send_telegram_alert("▶️ **BOT RESUMED** ▶️\nThe AI is actively analyzing the market again.")
                logging.info("C2 Update: Bot state set to ACTIVE.")
                
            elif text_lower.startswith("/status"):
                eq = mt5.account_info().equity
                state_icon = "🟢 ACTIVE" if IS_BOT_ACTIVE else "🟡 PAUSED"
                
                status_msg = (
                    f"📊 **BOT STATUS & SETTINGS** 📊\n\n"
                    f"⚙️ **Engine State:** `{state_icon}`\n"
                    f"🏦 **Equity:** `${eq:,.2f}`\n"
                    f"🧠 **Threshold:** `{GATEKEEPER_THRESHOLD * 100:.2f}%`\n"
                    f"🎯 **Target Multipliers:** `TP {TP_MULT}x | SL {SL_MULT}x`\n"
                    f"🛡️ **Break-Even:** `{'ON' if USE_BREAK_EVEN else 'OFF'}` | `{BE_TRIGGER*100:.0f}%` | `+{BE_BUFFER}`\n" # <--- NEW
                    f"🔄 **Strategy:** `{STRATEGY_TYPE}`\n"
                    f"🔀 **Multi-Trade:** `{'ON' if ALLOW_MULTIPLE_TRADES else 'OFF'}`\n"
                    f"📌 **Fixed Lot:** `{FIXED_LOT_SIZE}`\n"
                    f"📈 **Dyn Equity Step:** `${DYN_STEP_EQUITY}`\n"
                    f"📈 **Dyn Lot Step:** `{DYN_STEP_LOT}`\n"
                    f"🛡️ **Max Lot:** `{MAX_LOT_SAFETY}`"
                )
                send_telegram_alert(status_msg)
                
            elif text_lower.startswith("/threshold"):
                try:
                    new_val = float(text.split()[1])
                    if 0.0 <= new_val <= 1.0:
                        GATEKEEPER_THRESHOLD = new_val
                        send_telegram_alert(f"✅ Threshold updated to: `{new_val * 100:.2f}%`")
                        logging.info(f"C2 Update: Threshold -> {new_val}")
                    else:
                        send_telegram_alert("⚠️ Value must be between 0.0 and 1.0")
                except: send_telegram_alert("⚠️ Format: `/threshold 0.55`")

            elif text_lower.startswith("/sl"):
                try:
                    new_val = float(text.split()[1])
                    SL_MULT = new_val
                    send_telegram_alert(f"✅ Stop Loss Multiplier updated to: `{SL_MULT}x ATR`")
                    logging.info(f"C2 Update: SL_MULT -> {SL_MULT}")
                except: send_telegram_alert("⚠️ Format: `/sl 2.5`")

            elif text_lower.startswith("/tp"):
                try:
                    new_val = float(text.split()[1])
                    TP_MULT = new_val
                    send_telegram_alert(f"✅ Take Profit Multiplier updated to: `{TP_MULT}x ATR`")
                    logging.info(f"C2 Update: TP_MULT -> {TP_MULT}")
                except: send_telegram_alert("⚠️ Format: `/tp 3.5`")

            elif text_lower.startswith("/strategy"):
                try:
                    new_val = text.split()[1].upper()
                    if new_val in ["FIXED", "DYNAMIC"]:
                        STRATEGY_TYPE = new_val
                        send_telegram_alert(f"✅ Strategy Type updated to: `{STRATEGY_TYPE}`")
                        logging.info(f"C2 Update: Strategy -> {STRATEGY_TYPE}")
                    else:
                        send_telegram_alert("⚠️ Strategy must be FIXED or DYNAMIC")
                except: send_telegram_alert("⚠️ Format: `/strategy DYNAMIC`")

            elif text_lower.startswith("/multitrade"):
                try:
                    new_val = text.split()[1].upper()
                    if new_val in ["ON", "TRUE", "1"]:
                        ALLOW_MULTIPLE_TRADES = True
                    elif new_val in ["OFF", "FALSE", "0"]:
                        ALLOW_MULTIPLE_TRADES = False
                    send_telegram_alert(f"✅ Multi-Trade is now: `{'ON' if ALLOW_MULTIPLE_TRADES else 'OFF'}`")
                    logging.info(f"C2 Update: Multi-Trade -> {ALLOW_MULTIPLE_TRADES}")
                except: send_telegram_alert("⚠️ Format: `/multitrade ON` or `/multitrade OFF`")

            elif text_lower.startswith("/fixedlot"):
                try:
                    new_val = float(text.split()[1])
                    FIXED_LOT_SIZE = new_val
                    send_telegram_alert(f"✅ Fixed Lot Size updated to: `{FIXED_LOT_SIZE}`")
                    logging.info(f"C2 Update: Fixed Lot -> {FIXED_LOT_SIZE}")
                except: send_telegram_alert("⚠️ Format: `/fixedlot 0.05`")

            elif text_lower.startswith("/dynequity"):
                try:
                    new_val = float(text.split()[1])
                    DYN_STEP_EQUITY = new_val
                    send_telegram_alert(f"✅ Dynamic Equity Step updated to: `${DYN_STEP_EQUITY}`")
                    logging.info(f"C2 Update: Dyn Equity -> {DYN_STEP_EQUITY}")
                except: send_telegram_alert("⚠️ Format: `/dynequity 200.0`")

            elif text_lower.startswith("/dynlot"):
                try:
                    new_val = float(text.split()[1])
                    DYN_STEP_LOT = new_val
                    send_telegram_alert(f"✅ Dynamic Lot Step updated to: `{DYN_STEP_LOT}`")
                    logging.info(f"C2 Update: Dyn Lot Step -> {DYN_STEP_LOT}")
                except: send_telegram_alert("⚠️ Format: `/dynlot 0.02`")

            elif text_lower.startswith("/maxlot"):
                try:
                    new_val = float(text.split()[1])
                    MAX_LOT_SAFETY = new_val
                    send_telegram_alert(f"✅ Max Lot Safety Cap updated to: `{MAX_LOT_SAFETY}`")
                    logging.info(f"C2 Update: Max Lot -> {MAX_LOT_SAFETY}")
                except: send_telegram_alert("⚠️ Format: `/maxlot 10.0`")

            elif text_lower.startswith("/breakeven"):
                try:
                    new_val = text.split()[1].upper()
                    if new_val in ["ON", "TRUE", "1"]:
                        USE_BREAK_EVEN = True
                    elif new_val in ["OFF", "FALSE", "0"]:
                        USE_BREAK_EVEN = False
                    send_telegram_alert(f"✅ Break-Even is now: `{'ON' if USE_BREAK_EVEN else 'OFF'}`")
                    logging.info(f"C2 Update: Break-Even -> {USE_BREAK_EVEN}")
                except: send_telegram_alert("⚠️ Format: `/breakeven ON` or `/breakeven OFF`")

            elif text_lower.startswith("/betrigger"):
                try:
                    new_val = float(text.split()[1])
                    BE_TRIGGER = new_val
                    send_telegram_alert(f"✅ Break-Even Trigger updated to: `{BE_TRIGGER * 100:.0f}%` of Take Profit target")
                    logging.info(f"C2 Update: BE_TRIGGER -> {BE_TRIGGER}")
                except: send_telegram_alert("⚠️ Format: `/betrigger 0.60` (for 60%)")

            elif text_lower.startswith("/bebuffer"):
                try:
                    new_val = float(text.split()[1])
                    BE_BUFFER = new_val
                    send_telegram_alert(f"✅ Break-Even Buffer updated to: `+{BE_BUFFER}`")
                    logging.info(f"C2 Update: BE_BUFFER -> {BE_BUFFER}")
                except: send_telegram_alert("⚠️ Format: `/bebuffer 0.05`")
                    
    except Exception as e:
        pass

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
    # Position 0 is the newly forming candle. Training uses a completed signal
    # candle and enters on the following open, so live inference must start at 1.
    rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 1, REQUIRED_CANDLES)
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
    
    feat_cols = list(FEATURE_COLUMNS)
    X_scaled = scaler.transform(df[feat_cols])
    
    live_sequence = X_scaled[-LOOKBACK:]
    tensor = np.expand_dims(live_sequence, axis=0).astype(np.float32)
    return tensor, live_atr

def get_server_time_str():
    """Fetches the latest MT5 server time and formats it to a readable string."""
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick:
        return datetime.fromtimestamp(tick.time, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    return "UNKNOWN TIME"


def get_strategy_positions():
    """Return only positions opened by this strategy, never manual/other-bot trades."""
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None:
        return ()
    return tuple(pos for pos in positions if pos.magic == MAGIC_NUMBER)

# ==========================================
# 3.5 REAL-TIME TRADE MONITOR
# ==========================================
def check_closed_trades():
    global active_trade_tickets
    
    positions = get_strategy_positions()
        
    current_tickets = {p.ticket for p in positions}
    closed_tickets = active_trade_tickets - current_tickets
    
    for ticket in closed_tickets:
        deals = mt5.history_deals_get(position=ticket)
        if deals and len(deals) > 1:
            out_deal = deals[-1]
            profit = out_deal.profit
            price = out_deal.price
            reason_code = out_deal.reason
            comment = out_deal.comment.lower()
            
            # --- NEW: Determine the exact reason for closure ---
            if reason_code == mt5.DEAL_REASON_TP or "tp" in comment:
                close_reason = "🎯 Take Profit (TP)"
            elif reason_code == mt5.DEAL_REASON_SL or "sl" in comment:
                close_reason = "🛑 Stop Loss (SL)"
            elif "friday" in comment or "liquidation" in comment:
                close_reason = "🛡️ Friday Flat (Weekend Close)"
            elif reason_code == mt5.DEAL_REASON_SO:
                close_reason = "💀 Margin Call (Stop Out)"
            elif reason_code == mt5.DEAL_REASON_CLIENT:
                close_reason = "🧑‍💻 Manual Close (User)"
            else:
                close_reason = f"⚙️ Broker/Auto (Code: {reason_code})"
                
            # --- NEW: Get exact server time the deal closed ---
            close_time = datetime.fromtimestamp(out_deal.time, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            
            icon = "🟢" if profit > 0 else "🔴"
            result_text = "PROFIT" if profit > 0 else "LOSS"
            
            close_msg = (
                f"{icon} **TRADE CLOSED ({result_text})** {icon}\n\n"
                f"🕒 **Server Time:** `{close_time}`\n"
                f"💎 **Asset:** {SYMBOL}\n"
                f"🎫 **Ticket:** `#{ticket}`\n"
                f"📝 **Reason:** {close_reason}\n"
                f"💲 **Close Price:** `{price:.2f}`\n"
                f"💵 **Net PnL:** `${profit:.2f}`\n"
                f"🏦 **New Equity:** `${mt5.account_info().equity:,.2f}`"
            )
            
            # --- UPDATED: Injected Server Time into the log ---
            logging.info(f"[{close_time}] Trade Closed: #{ticket} | Reason: {close_reason} | PnL: ${profit:.2f}")
            send_telegram_alert(close_msg)
            
    active_trade_tickets = current_tickets

def apply_breakeven():
    """Dynamically moves SL to Entry only after reaching 60% of the Take Profit distance."""
    if not USE_BREAK_EVEN:
        return
        
    positions = get_strategy_positions()
    if len(positions) == 0:
        return
        
    for pos in positions:
        entry = pos.price_open
        current = pos.price_current
        ticket = pos.ticket
        sl = pos.sl
        tp = pos.tp
        
        # Safety check to avoid division errors
        if tp == 0.0:
            continue 
            
        # --- NEW: DYNAMIC TRIGGER MATH ---
        # Calculate the exact dollar distance between your Entry and Take Profit
        total_target_distance = abs(tp - entry)
        
        # The trigger is now dynamically set to 60% of the way to your Take Profit
        dynamic_trigger = total_target_distance * BE_TRIGGER 
        
        # --- BUY TRADE LOGIC ---
        if pos.type == mt5.ORDER_TYPE_BUY:
            if (current - entry) >= dynamic_trigger:
                new_sl = entry + BE_BUFFER
                
                if sl < new_sl: 
                    req = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": SYMBOL,
                        "position": ticket,
                        "sl": new_sl,
                        "tp": tp,
                        "magic": MAGIC_NUMBER
                    }
                    res = mt5.order_send(req)
                    if res.retcode == mt5.TRADE_RETCODE_DONE:
                        srv_time = get_server_time_str()
                        logging.info(f"[{srv_time}] 🛡️ Dynamic BE Applied to BUY #{ticket}: SL moved to {new_sl:.2f}")
                        send_telegram_alert(f"🛡️ **DYNAMIC BREAK-EVEN** 🛡️\n\n🕒 **Time:** `{srv_time}`\n📈 **Type:** BUY\n🎫 **Ticket:** `#{ticket}`\n📏 **Milestone:** 60% to Target\n🔒 **New SL:** `{new_sl:.2f}` (+${BE_BUFFER})")
                        
        # --- SELL TRADE LOGIC ---
        elif pos.type == mt5.ORDER_TYPE_SELL:
            if (entry - current) >= dynamic_trigger:
                new_sl = entry - BE_BUFFER
                
                if sl > new_sl or sl == 0.0:
                    req = {
                        "action": mt5.TRADE_ACTION_SLTP,
                        "symbol": SYMBOL,
                        "position": ticket,
                        "sl": new_sl,
                        "tp": tp,
                        "magic": MAGIC_NUMBER
                    }
                    res = mt5.order_send(req)
                    if res.retcode == mt5.TRADE_RETCODE_DONE:
                        srv_time = get_server_time_str()
                        logging.info(f"[{srv_time}] 🛡️ Dynamic BE Applied to SELL #{ticket}: SL moved to {new_sl:.2f}")
                        send_telegram_alert(f"🛡️ **DYNAMIC BREAK-EVEN** 🛡️\n\n🕒 **Time:** `{srv_time}`\n📉 **Type:** SELL\n🎫 **Ticket:** `#{ticket}`\n📏 **Milestone:** 60% to Target\n🔒 **New SL:** `{new_sl:.2f}` (+${BE_BUFFER})")

# ==========================================
# 4. EXECUTION & BROKER ROUTING
# ==========================================
def execute_trade(action, atr):
    # 1. Calculate Prices First
    tick = mt5.symbol_info_tick(SYMBOL)
    direction = "🟢 BUY" if action == 1 else "🔴 SELL"
    
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

    # 2. Execution Lock & Reversal Logic
    positions = get_strategy_positions()
    if positions is not None and len(positions) > 0:
        if not ALLOW_MULTIPLE_TRADES:
            for pos in positions:
                # MT5 Types: 0 is BUY, 1 is SELL
                is_opposite = (action == 1 and pos.type == mt5.ORDER_TYPE_SELL) or \
                              (action == -1 and pos.type == mt5.ORDER_TYPE_BUY)
                
                if is_opposite:
                    logging.info(f"🔄 REVERSAL DETECTED: Attempting to close opposite trade #{pos.ticket}...")
                    
                    # Prepare the counter-order to close the existing position
                    close_tick = mt5.symbol_info_tick(SYMBOL)
                    close_price = close_tick.ask if pos.type == mt5.ORDER_TYPE_SELL else close_tick.bid
                    close_type = mt5.ORDER_TYPE_BUY if pos.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL
                    
                    close_req = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": SYMBOL,
                        "volume": pos.volume,
                        "type": close_type,
                        "position": pos.ticket, 
                        "price": close_price,
                        "deviation": 20,
                        "magic": MAGIC_NUMBER,
                        "comment": "AI Reversal Close",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_FOK,
                    }
                    
                    close_result = mt5.order_send(close_req)
                    
                    if close_result.retcode != mt5.TRADE_RETCODE_DONE:
                        logging.error(f"Failed to close trade #{pos.ticket} for reversal. Error Code: {close_result.retcode}")
                        send_telegram_alert(f"⚠️ **REVERSAL FAILED** ⚠️\nCould not close Ticket `#{pos.ticket}`. Broker Error: `{close_result.retcode}`")
                        return # Abort opening the new trade to protect margin
                    else:
                        logging.info(f"Successfully closed trade #{pos.ticket}. Proceeding with new {direction}.")
                        # Let the script continue downward to open the new trade!
                        
                else:
                    # Same direction trade already exists. Let it run.
                    logging.info(
                        f"Skipped Signal (Lock ON) -> "
                        f"Dir: {direction} | Asset: {SYMBOL} | Entry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}"
                    )
                    return

    # 3. Lot Size Calculation
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
    
    # 4. Order Packaging
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
        "type_filling": mt5.ORDER_FILLING_FOK,
    }
    
    # 5. Broker Routing
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.warning(f"Order Failed (Code: {result.retcode}) -> Dir: {direction} | Entry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
        
        fail_msg = (
            f"⚠️ **TRADE REJECTED BY BROKER** ⚠️\n\n"
            f"❌ **Error Code:** `{result.retcode}`\n"
            f"📈 **Attempted:** {direction}\n"
            f"💎 **Asset:** {SYMBOL}\n"
            f"📊 **Volume:** {lot_size} Lots\n"
            f"💲 **Price:** `{price:.2f}`\n"
            f"🛑 **Stop Loss:** `{sl:.2f}`\n"
            f"🎯 **Take Profit:** `{tp:.2f}`\n\n"
            f"🤖 *Please check MT5 Terminal/Logs.*"
        )
        send_telegram_alert(fail_msg)
    else:
        # --- NEW: Grab server time for the receipt ---
        srv_time = get_server_time_str()
        
        alert_msg = (
            f"🚨 **AI TRADE EXECUTED** 🚨\n\n"
            f"🕒 **Server Time:** `{srv_time}`\n"
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
        
        # --- UPDATED: Injected Server Time into the log ---
        logging.info(f"[{srv_time}] $$$ {direction} EXECUTED! Ticket: {result.order} | Dir: {direction} | Asset: {SYMBOL} | Volume: {lot_size} | Entry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
        send_telegram_alert(alert_msg)

def liquidate_weekend_positions():
    positions = get_strategy_positions()
    if len(positions) == 0:
        return False

    logging.info("🛡️ WEEKEND PROTECTION TRIGGERED: Liquidating all open positions...")
    closed_any = False

    for pos in positions:
        tick = mt5.symbol_info_tick(SYMBOL)
        # Calculate the counter-order to close it
        price = tick.ask if pos.type == mt5.ORDER_TYPE_SELL else tick.bid
        close_type = mt5.ORDER_TYPE_BUY if pos.type == mt5.ORDER_TYPE_SELL else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": price,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "Friday Liquidation",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }

        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"Successfully liquidated Ticket #{pos.ticket} for the weekend.")
            closed_any = True
        else:
            logging.error(f"Failed to liquidate Ticket #{pos.ticket}. Error: {result.retcode}")

    if closed_any:
        send_telegram_alert("🛡️ **FRIDAY LIQUIDATION COMPLETE** 🛡️\nAll positions forcefully closed to prevent weekend gap risk.")
        
    return True


def apply_time_stop():
    """Close strategy-owned positions after the label/backtest horizon expires."""
    positions = get_strategy_positions()
    if len(positions) == 0:
        return
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return

    maximum_age_seconds = MODEL.max_horizon * 5 * 60
    for pos in positions:
        if tick.time - pos.time < maximum_age_seconds:
            continue
        close_type = (
            mt5.ORDER_TYPE_BUY
            if pos.type == mt5.ORDER_TYPE_SELL
            else mt5.ORDER_TYPE_SELL
        )
        close_price = tick.ask if pos.type == mt5.ORDER_TYPE_SELL else tick.bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": pos.volume,
            "type": close_type,
            "position": pos.ticket,
            "price": close_price,
            "deviation": 20,
            "magic": MAGIC_NUMBER,
            "comment": "AI 24-Bar Time Stop",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(
                f"Time stop closed strategy position #{pos.ticket} after "
                f"{MODEL.max_horizon} bars."
            )
        else:
            logging.error(
                f"Time stop failed for strategy position #{pos.ticket}: "
                f"broker code {result.retcode}"
            )

# ==========================================
# 5. THE ACTIVE POLLING LOOP 
# ==========================================
send_telegram_alert("✅ **AI Live Execution Script Started** ✅\n\nThe bot is now actively monitoring the market and ready to execute trades based on AI signals.")
logging.info("[*] Starting Live AI Execution Loop (With Real-Time Monitoring)...")

try:
    # --- NEW: FLUSH OLD TELEGRAM COMMANDS ---
    try:
        init_url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
        init_resp = requests.get(init_url, timeout=3).json()
        if init_resp.get("ok") and init_resp.get("result"):
            telegram_offset = init_resp["result"][-1]["update_id"] + 1
            logging.info("Flushed historical Telegram messages.")
    except Exception:
        pass
    # ----------------------------------------
    
    initial_positions = get_strategy_positions()
    if initial_positions:
        active_trade_tickets = {p.ticket for p in initial_positions}

    last_processed_candle = None  # <--- MEMORY FOR THE STALE CHECK
    last_telegram_check = 0  # <--- NEW: Timer for Telegram API
    friday_liquidation_done = False   # <--- NEW: Prevents the bot from liquidating 100 times in a row

    while True:
        current_time = time.time()
        
        # --- CHECK TELEGRAM COMMANDS EVERY 10 SECONDS ---
        if current_time - last_telegram_check >= 10:
            check_telegram_commands()
            last_telegram_check = current_time

        check_closed_trades()

        # --- NEW: ACTIVE BREAK-EVEN MONITOR ---
        apply_breakeven()
        apply_time_stop()
        # --------------------------------------

        # --- NEW: WEEKEND GAP PROTECTION CHECK ---
        if WEEKEND_PROTECTION:
            latest_tick = mt5.symbol_info_tick(SYMBOL)
            if latest_tick:
                # Convert the broker's UNIX timestamp to a readable datetime
                srv_time = datetime.fromtimestamp(latest_tick.time, timezone.utc)
                
                # weekday() 4 is Friday. 0 is Monday.
                if srv_time.weekday() == 4 and srv_time.hour >= FRIDAY_LIQ_HOUR and srv_time.minute >= FRIDAY_LIQ_MINUTE:
                    if not friday_liquidation_done:
                        liquidate_weekend_positions()
                        friday_liquidation_done = True
                        
                        # Automatically pause the bot so it doesn't open new trades 5 minutes later
                        IS_BOT_ACTIVE = False
                        send_telegram_alert("⏸️ **BOT HIBERNATING FOR WEEKEND** ⏸️\nThe AI has been paused. Use `/start` on Monday to resume trading.")
                        logging.info("Bot put into weekend hibernation mode.")
                        
                # Reset the safety lock on Monday so it works again next week
                elif srv_time.weekday() == 0:
                    if friday_liquidation_done:
                        friday_liquidation_done = False
                        IS_BOT_ACTIVE = True  # <--- NEW: Auto-wakes the bot!
                        send_telegram_alert("▶️ **MONDAY MARKET OPEN** ▶️\nWeekend hibernation complete. The AI is actively trading again.")
        # -----------------------------------------
        
        seconds_past_candle = current_time % 300
        
        if 2 <= seconds_past_candle < 6: 
            
            # --- NEW: Check if the bot is paused before running AI ---
            if not IS_BOT_ACTIVE:
                logging.info("    -> Bot is currently PAUSED via Telegram. Skipping AI inference.")
                time.sleep(2)
                continue # Skips the rest of this loop and waits for the next candle
            
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

            # --- AI INFERENCE (This only runs if IS_BOT_ACTIVE == True) ---
            input_tensor, current_atr = get_live_tensor()
            if input_tensor is not None:
                logits_a = session_a.run(None, {'input': input_tensor})[0]
                sig_a = np.argmax(logits_a, axis=1)[0]
                
                logits_b = session_b.run(None, {'input': input_tensor})[0]
                exp_b = np.exp(logits_b - np.max(logits_b, axis=1, keepdims=True))
                prob_b = (exp_b / np.sum(exp_b, axis=1, keepdims=True))[0][1]
                
                # --- NEW: Translate the numeric signal to a readable word ---
                action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
                sig_name = action_map.get(sig_a, "UNKNOWN")

                # --- NEW: Grab time for the signal log ---
                srv_time = get_server_time_str()

                # --- UPDATED: Injected Server Time ---
                logging.info(f"    -> [{srv_time}] Model A: {sig_a} ({sig_name}) | Model B Gatekeeper: {prob_b*100:.2f}%")
                
                if prob_b >= GATEKEEPER_THRESHOLD and sig_a in [1, 2]:
                    trade_action = 1 if sig_a == 1 else -1
                    logging.info(f"    -> [{srv_time}] AI THRESHOLD MET. Initiating Trade Routing...")
                    execute_trade(trade_action, current_atr)
                else:
                    logging.info(f"    -> [{srv_time}] Signal Rejected/Hold. Waiting for next candle.")
                    
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
