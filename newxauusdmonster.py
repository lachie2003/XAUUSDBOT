import os
import time
import datetime as dt
from dataclasses import dataclass
from typing import Dict
import threading
import sys
import csv
import traceback
import json

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pytz
import requests
from flask import Flask, render_template_string

from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor, black
from reportlab.pdfgen import canvas

from license_manager import verify_license
from auto_update import check_for_update, apply_update

# ============================================================
# SIMPLE LOGGING HELPERS
# ============================================================

LOG_BASE_DIR = "logs"


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        # last resort: ignore logging dir errors
        pass


def _get_daily_log_path(subfolder: str, prefix: str) -> str:
    """
    Build a daily CSV file path like:
    logs/trades/2025-11-16_trades.csv
    """
    today = dt.datetime.now().strftime("%Y-%m-%d")
    folder = os.path.join(LOG_BASE_DIR, subfolder)
    _ensure_dir(folder)
    filename = f"{today}_{prefix}.csv"
    return os.path.join(folder, filename)


def log_trade(row: dict) -> None:
    """
    row example:
    {
        "time": "2025-11-16 14:30:00",
        "symbol": "XAUUSD",
        "direction": "BUY",
        "entry": 1925.50,
        "sl": 1919.00,
        "tp": 1935.00,
        "risk_pct": 1.0,
        "result": "open"
    }
    """
    try:
        path = _get_daily_log_path("trades", "trades")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        # don't crash the bot because of logging
        pass


def log_setup(row: dict) -> None:
    """
    row example:
    {
        "time": "2025-11-16 14:25:00",
        "symbol": "XAUUSD",
        "score": 8,
        "direction": "SELL",
        "reason": "4H/1H bias short, liquidity sweep, FVG, A+"
    }
    """
    try:
        path = _get_daily_log_path("setups", "setups")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception:
        pass


def log_error(message: str) -> None:
    """
    Saves errors into logs/errors/...
    """
    try:
        path = _get_daily_log_path("errors", "errors")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["time", "message"])
            now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([now, message])
    except Exception:
        pass


def log_debug(message: str) -> None:
    """
    Light-weight debug logging to logs/debug/...
    """
    try:
        path = _get_daily_log_path("debug", "debug")
        write_header = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writeheader(["time", "message"])
            now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([now, message])
    except Exception:
        pass


# ============================================================
# GLOBAL EXCEPTION HANDLER
# ============================================================

def _format_exception(exc_type, exc_value, exc_tb) -> str:
    return "".join(traceback.format_exception(exc_type, exc_value, exc_tb))


def global_exception_handler(exc_type, exc_value, exc_tb):
    msg = _format_exception(exc_type, exc_value, exc_tb)
    log_error(msg)
    print("UNHANDLED EXCEPTION:")
    print(msg)


sys.excepthook = global_exception_handler

# ------------------------------------------------------------
# Bot Version / License / Auto-Update
# ------------------------------------------------------------

BOT_VERSION = "1.0.1"
LICENSE_KEY = "TJR-GODMODE-001"

print("[UPDATE] Checking for new version...")
need_update, msg = check_for_update()
print("[UPDATE]", msg)
if need_update:
    ok, msg2 = apply_update()
    print("[UPDATE]", msg2)
    print("[UPDATE] Update applied. Exiting so launcher can restart the bot...")
    print("[UPDATE] If you started this manually, just run it again.")
    sys.exit(0)

print("[LICENSE] Verifying license...")
licensed, lic_msg = verify_license(LICENSE_KEY)
print("[LICENSE]", lic_msg)

# ============================================================
# LOAD CONFIG & SECRETS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_json(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[CFG] Missing {path}")
        return default if default is not None else {}
    except Exception as e:
        print(f"[CFG] Error loading {path}:", e)
        return default if default is not None else {}


CONFIG = load_json(os.path.join(BASE_DIR, "config.json"), {})
SECRETS = load_json(os.path.join(BASE_DIR, "secrets.json"), {})

# ============================================================
# TELEGRAM ALERT SYSTEM
# ============================================================

TELEGRAM_TOKEN = SECRETS.get("telegram", {}).get("token", "")
CHAT_ID = SECRETS.get("telegram", {}).get("chat_id", "")


def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }
    try:
        r = requests.post(url, data=payload, timeout=4)
        if r.status_code != 200:
            time.sleep(1)
            requests.post(url, data=payload, timeout=4)
    except Exception as e:
        print("[TEL] Failed:", e)


def send_telegram_document(filepath: str, caption: str = ""):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        return
    if not os.path.exists(filepath):
        print("[TEL] File not found for document:", filepath)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendDocument"
    try:
        with open(filepath, "rb") as f:
            files = {"document": f}
            data = {
                "chat_id": CHAT_ID,
                "caption": caption,
                "parse_mode": "HTML",
            }
            r = requests.post(url, data=data, files=files, timeout=10)
            if r.status_code != 200:
                print("[TEL] sendDocument failed:", r.text)
    except Exception as e:
        print("[TEL] Failed to send document:", e)


# ============================================================
# USER SETTINGS FROM CONFIG / SECRETS
# ============================================================

ACCOUNT = SECRETS.get("mt5", {}).get("account")
PASSWORD = SECRETS.get("mt5", {}).get("password")
SERVER = SECRETS.get("mt5", {}).get("server")
MT5_PATH = SECRETS.get("mt5", {}).get(
    "path",
    r"C:\Program Files\MetaTrader 5\terminal64.exe"
)

trading_cfg = CONFIG.get("trading", {})
DRY_RUN = bool(trading_cfg.get("dry_run", False))

if not licensed:
    DRY_RUN = True
    print("[LICENSE] Forcing DRY_RUN=True because bot is NOT licensed.")
else:
    print(f"[LICENSE] DRY_RUN remains {DRY_RUN} (licensed).")

RISK_PER_TRADE = float(trading_cfg.get("risk_per_trade", 0.01))
MIN_SCORE = float(trading_cfg.get("min_score", 7))
REFRESH_SECONDS = int(trading_cfg.get("refresh_seconds", 60))
MAX_POS_PER_SYMBOL = int(trading_cfg.get("max_positions", 2))
MAGIC = int(trading_cfg.get("magic", 777001))

symbol_cfg = CONFIG.get("symbol", {})
SYMBOL = symbol_cfg.get("name", "XAUUSD")
SPREAD_CAP = float(symbol_cfg.get("spread_cap", 60.0))

# ============================================================
# SESSIONS & NEWS
# ============================================================

NY = pytz.timezone("America/New_York")
UTC = pytz.UTC

sess_cfg = CONFIG.get("sessions", {}).get("ny_killzones", {})
am_times = sess_cfg.get("am", ["08:00", "10:00"])
pm_times = sess_cfg.get("pm", ["13:00", "15:00"])

NY_AM = (
    dt.time.fromisoformat(am_times[0]),
    dt.time.fromisoformat(am_times[1]),
)
NY_PM = (
    dt.time.fromisoformat(pm_times[0]),
    dt.time.fromisoformat(pm_times[1]),
)

news_cfg = CONFIG.get("news", {})
NEWS_BLOCK_MINUTES = int(news_cfg.get("block_minutes", 15))
RED_NEWS = news_cfg.get("events", [])

# ============================================================
# DEBUG
# ============================================================

DEBUG = True


def dbg(*a):
    if not DEBUG:
        return
    msg = " ".join(str(x) for x in a)
    print("[DBG]", msg)
    try:
        log_debug(msg)
    except Exception:
        pass


# ============================================================
# HUMAN READABLE LOGGING DIRS
# ============================================================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)

CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

def log_event(filename: str, text: str):
    ts = dt.datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(LOG_DIR, filename)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{ts} | {text}\n")
    except Exception as e:
        print("[LOG] Failed:", e)

# ============================================================
# STRUCTURES & DASHBOARD STATE
# ============================================================

@dataclass
class Setup:
    direction: str
    entry: float
    sl: float
    tp: float
    score: float
    rationale: Dict[str, float]


BOT_STATE = {
    "connected": False,
    "last_error": "",
    "last_update": "",
    "in_killzone": False,
    "in_news_block": False,
    "open_positions": 0,
    "last_setup": None,
    "last_trade": None,
    "equity": 0.0,
    "version": BOT_VERSION,
    "licensed": bool(licensed),
    "license_msg": lic_msg,
}


def update_state(**kwargs):
    now = dt.datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    BOT_STATE["last_update"] = now
    for k, v in kwargs.items():
        BOT_STATE[k] = v

# ============================================================
# MT5 CONNECTION
# ============================================================

def mt5_connect():
    mt5.shutdown()
    if MT5_PATH and os.path.exists(MT5_PATH):
        ok = mt5.initialize(path=MT5_PATH)
    else:
        ok = mt5.initialize()

    if not ok:
        err = f"MT5 init failed: {mt5.last_error()}"
        print("[MT5]", err)
        update_state(connected=False, last_error=err)
        log_event("errors.log", err)
        return False

    for _ in range(20):
        ti, ai = mt5.terminal_info(), mt5.account_info()
        if ti and ai and ti.connected:
            msg = f"Connected to {ai.server} | Balance: {ai.balance}"
            print("[MT5]", msg)
            update_state(connected=True, equity=ai.equity, last_error="")
            log_event("events.log", msg)
            send_telegram("🤖 <b>TJR Bot connected to MT5</b>")
            return True
        time.sleep(0.25)

    err = "Login or connection failed."
    print("[MT5]", err)
    update_state(connected=False, last_error=err)
    log_event("errors.log", err)
    return False

# ============================================================
# TIME FILTERS
# ============================================================

def in_killzone(now_utc):
    ny_time = now_utc.astimezone(NY).time()
    return (NY_AM[0] <= ny_time <= NY_AM[1]) or (NY_PM[0] <= ny_time <= NY_PM[1])


def in_news_block(now_utc):
    for name, timestr in RED_NEWS:
        news_time = dt.datetime.strptime(timestr, "%Y-%m-%d %H:%M").replace(tzinfo=UTC)
        diff = abs((now_utc - news_time).total_seconds()) / 60
        if diff <= NEWS_BLOCK_MINUTES:
            dbg(f"NEWS BLOCK: {name} (within {diff:.1f} mins)")
            return True
    return False

# ============================================================
# DATA FETCH
# ============================================================

def fetch(symbol, tf, bars):
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if rates is None:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df

# ============================================================
# ICT LOGIC
# ============================================================

def swings(df, left=2, right=2):
    h = df["high"].values
    l = df["low"].values
    n = len(df)
    sh = np.zeros(n, dtype=bool)
    sl = np.zeros(n, dtype=bool)
    for i in range(left, n - right):
        if all(h[i] > h[i-k] for k in range(1, left+1)) and all(h[i] > h[i+k] for k in range(1, right+1)):
            sh[i] = True
        if all(l[i] < l[i-k] for k in range(1, left+1)) and all(l[i] < l[i+k] for k in range(1, right+1)):
            sl[i] = True
    return sh, sl


def bias_h4_h1():
    h4 = fetch(SYMBOL, mt5.TIMEFRAME_H4, 800)
    h1 = fetch(SYMBOL, mt5.TIMEFRAME_H1, 800)
    if h4.empty or h1.empty:
        return "neutral"

    def dir(df):
        sh, sl = swings(df, 3, 3)
        pts = []
        for i in range(len(df)):
            if sh[i]:
                pts.append(("H", df["high"].iloc[i]))
            if sl[i]:
                pts.append(("L", df["low"].iloc[i]))
        if len(pts) < 3:
            return 0
        a, b, c = pts[-3:]
        if a[0] == "L" and b[0] == "H" and c[0] == "L":
            return -1
        if a[0] == "H" and b[0] == "L" and c[0] == "H":
            return 1
        ema50 = df["close"].ewm(span=50).mean()
        return 1 if df["close"].iloc[-1] > ema50.iloc[-1] else -1

    d4 = dir(h4)
    d1 = dir(h1)
    if d4 == 1 and d1 == 1:
        return "bull"
    if d4 == -1 and d1 == -1:
        return "bear"
    return "neutral"


def detect_sweep(df):
    if len(df) < 40:
        return False, "", 0.0
    window = df.tail(35)
    prev = window.iloc[:-3]
    last = window.iloc[-3]

    hi = prev["high"].max()
    lo = prev["low"].min()

    if last.high > hi and last.close < last.high:
        return True, "sell", hi
    if last.low < lo and last.close > last.low:
        return True, "buy", lo
    return False, "", 0.0


def detect_displacement(df):
    if len(df) < 20:
        return False, False, False

    atr5 = (df["high"] - df["low"]).rolling(5).mean()
    last = df.iloc[-2]
    disp = (last.high - last.low) > 1.2 * atr5.iloc[-2]

    sw_h, sw_l = swings(df.iloc[:-2], 2, 2)
    highs = df.iloc[:-2]["high"][sw_h].max() if sw_h.any() else df.iloc[:-2]["high"].max()
    lows = df.iloc[:-2]["low"][sw_l].min() if sw_l.any() else df.iloc[:-2]["low"].min()

    bos = last.close > highs or last.close < lows

    c2 = df.iloc[-3]
    c0 = df.iloc[-1]
    ifvg = (c2.high < c0.low) or (c2.low > c0.high)

    return disp, bos, ifvg


def refined_poi(df, direction):
    for i in range(len(df) - 3, 3, -1):
        c = df.iloc[i]
        rng = c.high - c.low
        if rng == 0:
            continue
        body = abs(c.open - c.close) / rng
        if direction == "buy" and c.close < c.open and body > 0.5:
            return c.low, c.high
        if direction == "sell" and c.close > c.open and body > 0.5:
            return c.low, c.high
    return 0.0, 0.0


def confirm_at_poi(df, direction, low, high):
    last = df.iloc[-2]
    touched = (
        (low <= last.low <= high)
        or (low <= last.high <= high)
        or (last.low < low and last.high > high)
    )
    if not touched:
        return False
    mid = (low + high) / 2
    return last.close > mid if direction == "buy" else last.close < mid


def room_to_target(direction, entry, sl):
    h1 = fetch(SYMBOL, mt5.TIMEFRAME_H1, 600)
    if h1.empty:
        return False, 0.0

    sw_h, sw_l = swings(h1, 2, 2)
    highs = h1["high"][sw_h].max() if sw_h.any() else h1["high"].max()
    lows = h1["low"][sw_l].min() if sw_l.any() else h1["low"].min()

    info = mt5.symbol_info(SYMBOL)
    point = info.point
    dist = (highs - entry) / point if direction == "buy" else (entry - lows) / point
    rr = abs(dist) / abs((entry - sl) / point)
    return rr >= 1.2, rr

# ============================================================
# SETUP FINDER
# ============================================================

def find_setup():
    info = mt5.symbol_info(SYMBOL)
    tick = mt5.symbol_info_tick(SYMBOL)
    if not info or not tick:
        log_event("errors.log", "No symbol info/tick in find_setup")
        return None

    spread = (tick.ask - tick.bid) / info.point
    if spread > SPREAD_CAP:
        dbg("Spread too high:", spread)
        send_telegram("⚠️ <b>Spread too high</b> — skipping XAUUSD")
        log_event("errors.log", f"Spread too high: {spread:.1f} pts (cap {SPREAD_CAP})")
        return None

    m5 = fetch(SYMBOL, mt5.TIMEFRAME_M5, 500)
    if m5.empty:
        return None

    bias = bias_h4_h1()
    if bias == "neutral":
        dbg("No HTF bias")
        return None

    swept, swept_side, lvl = detect_sweep(m5)
    disp, bos, ifvg = detect_displacement(m5)
    if not swept or not disp or not bos or not ifvg:
        dbg("Weak impulse")
        return None

    direction = None
    if swept_side == "sell" and bias == "bull":
        direction = "buy"
    if swept_side == "buy" and bias == "bear":
        direction = "sell"
    if direction is None:
        dbg("Direction mismatch")
        return None

    low, high = refined_poi(m5, direction)
    if low == 0 and high == 0:
        dbg("No POI")
        return None

    if not confirm_at_poi(m5, direction, low, high):
        dbg("No confirm at POI")
        return None

    entry = tick.bid if direction == "buy" else tick.ask
    point = info.point
    sl = low - 50 * point if direction == "buy" else high + 50 * point
    rr_ok, rr_val = room_to_target(direction, entry, sl)
    r = abs(entry - sl)
    tp = entry + 2 * r if direction == "buy" else entry - 2 * r

    rationale = {
        "htf": 2,
        "sweep": 2,
        "bos": 3,
        "poi": 1.5,
        "confirm": 1.5,
        "room": 1 if rr_ok else 0,
    }
    score = sum(rationale.values())
    if score < MIN_SCORE:
        dbg("Score low")
        return None

    update_state(last_setup={
        "direction": direction,
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "score": round(score, 2),
    })

    log_setup({
        "time": dt.datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": SYMBOL,
        "score": round(score, 2),
        "direction": direction,
        "reason": (
            f"bias={bias}, sweep_side={swept_side}, "
            f"displacement={disp}, bos={bos}, ifvg={ifvg}, rr_ok={rr_ok}"
        ),
    })

    send_telegram(
        f"📊 <b>Setup Detected</b>\n"
        f"Direction: {direction.upper()}\n"
        f"Entry: {entry}\n"
        f"Score: {score:.1f}"
    )

    return Setup(direction, entry, sl, tp, score, rationale)

# ============================================================
# TRADE CHART
# ============================================================

def create_trade_chart(setup: Setup) -> str:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        log_event("errors.log", f"matplotlib not available for chart: {e}")
        return ""

    df = fetch(SYMBOL, mt5.TIMEFRAME_M5, 200)
    if df.empty:
        log_event("errors.log", "No M5 data for trade chart")
        return ""

    df = df.tail(120)

    try:
        plt.figure(figsize=(10, 5))
        plt.plot(df["time"], df["close"], linewidth=1.2)

        plt.axhline(setup.entry, linestyle="--", linewidth=0.9)
        plt.axhline(setup.sl, linestyle="--", linewidth=0.9)
        plt.axhline(setup.tp, linestyle="--", linewidth=0.9)

        plt.title(
            f"{SYMBOL} {setup.direction.upper()} "
            f"Entry {setup.entry:.2f} SL {setup.sl:.2f} TP {setup.tp:.2f} "
            f"Score {setup.score:.1f}"
        )
        plt.xlabel("Time (M5)")
        plt.ylabel("Price")
        plt.xticks(rotation=45)
        plt.tight_layout()

        fname = f"{SYMBOL}_{dt.datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.png"
        fpath = os.path.join(CHART_DIR, fname)
        plt.savefig(fpath, dpi=120)
        plt.close()

        log_event("events.log", f"Trade chart saved: {fpath}")
        return fpath
    except Exception as e:
        log_event("errors.log", f"Failed to create trade chart: {e}")
        return ""

# ============================================================
# RISK & ORDER EXECUTION
# ============================================================

def calc_lots(entry, sl):
    acc = mt5.account_info()
    if acc is None:
        log_event("errors.log", "account_info() returned None in calc_lots")
        return 0.0
    risk_money = acc.equity * RISK_PER_TRADE

    info = mt5.symbol_info(SYMBOL)
    if info is None:
        log_event("errors.log", "symbol_info() returned None in calc_lots")
        return 0.0

    is_buy = sl < entry
    one_lot_loss = mt5.order_calc_profit(
        mt5.ORDER_TYPE_BUY if is_buy else mt5.ORDER_TYPE_SELL,
        SYMBOL, 1.0, entry, sl
    )

    if one_lot_loss is None or one_lot_loss == 0:
        log_event("errors.log", f"order_calc_profit invalid: {one_lot_loss}")
        return 0.0

    raw = risk_money / abs(one_lot_loss)
    step = info.volume_step
    return round(raw / step) * step


def open_positions():
    pos = mt5.positions_get(symbol=SYMBOL)
    if pos is None:
        return 0
    return sum(1 for p in pos if p.magic == MAGIC)


def send_order(setup: Setup):
    info = mt5.symbol_info(SYMBOL)
    tick = mt5.symbol_info_tick(SYMBOL)
    if not info or not tick:
        log_event("errors.log", "No symbol info/tick in send_order")
        return

    price = tick.ask if setup.direction == "buy" else tick.bid
    lots = calc_lots(price, setup.sl)

    if lots <= 0:
        dbg("Zero lots")
        log_error("calc_lots returned 0")
        return

    chart_path = create_trade_chart(setup)

    if DRY_RUN:
        print("[DRY] ORDER:", setup.direction, lots)
        log_event(
            "trades.log",
            f"DRY RUN | {setup.direction.upper()} | Lots: {lots:.2f} | "
            f"Entry: {price:.2f} | SL: {setup.sl:.2f} | TP: {setup.tp:.2f}"
        )

        send_telegram(
            f"🧪 <b>DRY RUN TRADE</b>\n"
            f"{setup.direction.upper()} {lots} lots\n"
            f"Entry: {price}\nSL: {setup.sl}\nTP: {setup.tp}"
        )

        if chart_path:
            send_telegram_document(
                chart_path,
                caption="📈 DRY RUN – trade setup chart"
            )
        return

    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lots,
        "type": mt5.ORDER_TYPE_BUY if setup.direction == "buy" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": setup.sl,
        "tp": setup.tp,
        "magic": MAGIC,
        "comment": "TJR",
        "deviation": 50,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)

    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
        log_trade({
            "time": dt.datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": SYMBOL,
            "direction": setup.direction,
            "entry": price,
            "sl": setup.sl,
            "tp": setup.tp,
            "risk_pct": RISK_PER_TRADE * 100,
            "result": "open",
        })
    else:
        log_error(f"Order failed: retcode={res.retcode} message={res.comment}")
        print("[LIVE] ORDER FAILED:", res.retcode)

    log_event(
        "trades.log",
        f"TRADE ATTEMPT | {setup.direction.upper()} | Lots: {lots:.2f} | "
        f"Entry: {price:.2f} | SL: {setup.sl:.2f} | TP: {setup.tp:.2f} | "
        f"retcode={res.retcode}"
    )

    update_state(last_trade={
        "direction": setup.direction,
        "entry": round(price, 2),
        "sl": round(setup.sl, 2),
        "tp": round(setup.tp, 2),
        "lots": lots,
        "retcode": res.retcode,
    })

    send_telegram(
        f"🚀 <b>TRADE PLACED</b>\n"
        f"{setup.direction.upper()} {lots} lots\n"
        f"Entry: {price}\nSL: {setup.sl}\nTP: {setup.tp}"
    )

    if chart_path:
        send_telegram_document(
            chart_path,
            caption="📈 Live trade — setup chart"
        )

# ============================================================
# POSITION MANAGEMENT
# ============================================================

def manage_positions():
    pos = mt5.positions_get(symbol=SYMBOL)
    if pos is None:
        update_state(open_positions=0)
        return

    info = mt5.symbol_info(SYMBOL)
    tick = mt5.symbol_info_tick(SYMBOL)
    if not info or not tick:
        log_event("errors.log", "No symbol info/tick in manage_positions")
        return

    point = info.point

    count = 0
    for p in pos:
        if p.magic != MAGIC:
            continue

        count += 1
        entry = p.price_open
        sl = p.sl
        price = tick.bid if p.type == mt5.POSITION_TYPE_BUY else tick.ask

        r_unit = abs(entry - sl)
        if r_unit <= 0:
            continue

        r_mult = (price - entry) / r_unit if p.type == 0 else (entry - price) / r_unit

        if r_mult >= 1.0 and abs(sl - entry) > 0.1 * point:
            mt5.order_send({
                "action": mt5.TRADE_ACTION_SLTP,
                "position": p.ticket,
                "sl": entry,
                "tp": p.tp
            })
            print("[LIVE] SL BE")
            log_event(
                "trades.log",
                f"SL->BE | ticket={p.ticket} | entry={entry:.2f}"
            )
            send_telegram("🔒 SL moved to Break Even")

        if r_mult >= 2.0:
            mt5.order_send({
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": p.volume,
                "type": mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY,
                "position": p.ticket,
                "price": price,
                "magic": MAGIC,
                "comment": "TJR_2R",
                "deviation": 40,
                "type_filling": mt5.ORDER_FILLING_IOC
            })
            print("[LIVE] TP 2R")
            log_event(
                "trades.log",
                f"TP 2R | ticket={p.ticket} | close_price={price:.2f} | vol={p.volume}"
            )
            send_telegram("💰 <b>FULL TAKE PROFIT — 2R hit!</b>")

    update_state(open_positions=count)

# ============================================================
# WEEKLY REPORT GENERATION
# ============================================================

LAST_REPORT_WEEK = None

def get_week_bounds(now_utc: dt.datetime):
    today = now_utc.date()
    monday = today - dt.timedelta(days=today.weekday())
    monday_start = dt.datetime(monday.year, monday.month, monday.day)
    friday = monday + dt.timedelta(days=4)
    friday_end = dt.datetime(friday.year, friday.month, friday.day, 23, 59, 59)

    label = f"{monday.strftime('%b %d')}–{friday.strftime('%d, %Y')}"
    week_number = now_utc.isocalendar()[1]
    return monday_start, friday_end, label, week_number


def generate_weekly_report(now_utc: dt.datetime):
    monday_start, friday_end, label, week_no = get_week_bounds(now_utc)
    pdf_name = f"week_{week_no}.pdf"
    pdf_path = os.path.join(REPORT_DIR, pdf_name)

    if os.path.exists(pdf_path):
        log_event("events.log", f"Weekly report already exists: {pdf_name}")
        return

    try:
        deals = mt5.history_deals_get(monday_start, friday_end)
    except Exception as e:
        log_event("errors.log", f"history_deals_get error: {e}")
        return

    if deals is None or len(deals) == 0:
        log_event("events.log", f"No deals found for weekly report ({label})")
        return

    filtered = []
    for d in deals:
        try:
            if getattr(d, "symbol", "") != SYMBOL:
                continue
            if getattr(d, "magic", 0) != MAGIC:
                continue
            if float(getattr(d, "profit", 0.0)) == 0.0:
                continue
            filtered.append(d)
        except Exception:
            continue

    if not filtered:
        log_event("events.log", f"No closed trades for weekly report ({label})")
        return

    filtered = sorted(filtered, key=lambda x: x.time)
    total_trades = len(filtered)
    profits = [float(d.profit) for d in filtered]
    total_pnl = sum(profits)
    wins = sum(1 for p in profits if p > 0)
    losses = sum(1 for p in profits if p < 0)
    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
    best_trade = max(profits)
    worst_trade = min(profits)

    eq = []
    running = 0.0
    for p in profits:
        running += p
        eq.append(running)

    peak = eq[0]
    max_dd = 0.0
    for v in eq:
        if v > peak:
            peak = v
        dd = peak - v
        if dd > max_dd:
            max_dd = dd

    avg_pnl = total_pnl / total_trades if total_trades > 0 else 0.0

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFillColor(black)
    c.rect(0, 0, width, height, fill=1, stroke=0)

    gold = HexColor("#FFD700")
    grey = HexColor("#CCCCCC")

    c.setFont("Helvetica-Bold", 20)
    c.setFillColor(gold)
    c.drawString(40, height - 50, "TJR XAUUSD — Weekly Report")
    c.setFont("Helvetica", 12)
    c.setFillColor(grey)
    c.drawString(40, height - 70, f"Week: {label}")
    c.drawString(40, height - 85, f"Generated: {now_utc.strftime('%Y-%m-%d %H:%M UTC')}")

    y = height - 120
    line_h = 16
    c.setFont("Helvetica", 12)
    stats_lines = [
        f"Total Trades: {total_trades}",
        f"Wins: {wins}",
        f"Losses: {losses}",
        f"Win Rate: {win_rate:.1f}%",
        f"Total PnL: ${total_pnl:.2f} USD",
        f"Best Trade: ${best_trade:.2f} USD",
        f"Worst Trade: ${worst_trade:.2f} USD",
        f"Max Drawdown: ${max_dd:.2f} USD",
        f"Average PnL per Trade: ${avg_pnl:.2f} USD",
    ]
    for line in stats_lines:
        c.drawString(40, y, line)
        y -= line_h

    box_x = 40
    box_y = 140
    box_w = width - 80
    box_h = 200

    c.setStrokeColor(gold)
    c.rect(box_x, box_y, box_w, box_h, stroke=1, fill=0)
    c.setFont("Helvetica", 12)
    c.setFillColor(grey)
    c.drawString(box_x, box_y + box_h + 15, "Equity Curve (PnL over week, USD)")

    if len(eq) >= 2:
        min_eq = min(eq + [0.0])
        max_eq = max(eq + [0.0])
        span = max_eq - min_eq
        if span == 0:
            span = 1.0
        c.setStrokeColor(gold)
        c.setLineWidth(1.2)

        def norm_x(i):
            if len(eq) == 1:
                return box_x + box_w / 2
            return box_x + (i / (len(eq) - 1)) * box_w

        def norm_y(val):
            return box_y + ((val - min_eq) / span) * box_h

        prev_x = norm_x(0)
        prev_y = norm_y(eq[0])
        for i in range(1, len(eq)):
            x = norm_x(i)
            y2 = norm_y(eq[i])
            c.line(prev_x, prev_y, x, y2)
            prev_x, prev_y = x, y2

    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(grey)
    c.drawRightString(width - 20, 20, "TJR Monster Bot — XAUUSD NY Session")

    c.showPage()
    c.save()

    log_event("events.log", f"Weekly report generated: {pdf_name}")

    summary = (
        f"📊 <b>WEEKLY REPORT</b> ({label})\n\n"
        f"Total Trades: {total_trades}\n"
        f"Wins: {wins} | Losses: {losses}\n"
        f"Win Rate: {win_rate:.1f}%\n"
        f"Total PnL: ${total_pnl:.2f} USD\n"
        f"Best Trade: ${best_trade:.2f} USD\n"
        f"Worst Trade: ${worst_trade:.2f} USD\n"
        f"Max Drawdown: ${max_dd:.2f} USD\n"
        f"Avg PnL/Trade: ${avg_pnl:.2f} USD"
    )
    send_telegram(summary)
    send_telegram_document(pdf_path, caption=f"📎 Weekly report PDF ({label})")

# ============================================================
# DASHBOARD (FLASK APP)
# ============================================================

app = Flask(__name__)

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>TJR XAUUSD Bot Dashboard</title>
  <style>
    body { font-family: Arial, sans-serif; background: #111; color: #eee; padding: 20px; }
    h1 { color: #0fdf8f; }
    .card { background: #1b1b1b; padding: 15px; border-radius: 10px; margin-bottom: 15px; }
    .label { color: #999; font-size: 0.9rem; }
    .value { font-size: 1.1rem; }
    .status-online { color: #0fdf8f; }
    .status-offline { color: #ff4d4d; }
    .status-warn { color: #ffd24d; }
    pre { background: #000; padding: 10px; border-radius: 8px; }
  </style>
</head>
<body>
  <h1>TJR XAUUSD Bot Dashboard</h1>
  <div class="card">
    <div class="label">Connection</div>
    <div class="value">
      {% if state.connected %}
        <span class="status-online">● CONNECTED</span>
      {% else %}
        <span class="status-offline">● DISCONNECTED</span>
      {% endif %}
    </div>
    <div class="label">Bot Version</div>
    <div class="value">v{{ state.version }}</div>
    <div class="label">Last Update</div>
    <div class="value">{{ state.last_update }}</div>
    <div class="label">Equity</div>
    <div class="value">${{ "%.2f"|format(state.equity) }}</div>
    {% if state.last_error %}
      <div class="label">Last Error</div>
      <div class="value status-warn">{{ state.last_error }}</div>
    {% endif %}
  </div>


  <div class="card">
    <div class="label">Session / News</div>
    <div class="value">
      {% if state.in_killzone %}
        <span class="status-online">In NY Killzone</span>
      {% else %}
        <span class="status-offline">Outside Killzone</span>
      {% endif %}
      |
      {% if state.in_news_block %}
        <span class="status-warn">News Block ACTIVE</span>
      {% else %}
        News Clear
      {% endif %}
    </div>
    <div class="label">Open Positions</div>
    <div class="value">{{ state.open_positions }}</div>
  </div>

  <div class="card">
    <div class="label">Last Setup</div>
    {% if state.last_setup %}
      <pre>{{ state.last_setup | tojson(indent=2) }}</pre>
    {% else %}
      <div class="value">No setup yet.</div>
    {% endif %}
  </div>

  <div class="card">
    <div class="label">Last Trade</div>
    {% if state.last_trade %}
      <pre>{{ state.last_trade | tojson(indent=2) }}</pre>
    {% else %}
      <div class="value">No trade yet.</div>
    {% endif %}
  </div>
</body>
</html>
"""

@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD_HTML, state=BOT_STATE)


def run_dashboard():
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

# ============================================================
# MAIN LOOP
# ============================================================

def main():
    global LAST_REPORT_WEEK

    if not mt5_connect():
        return

    send_telegram("🟢 <b>TJR Bot started</b>\nXAUUSD • NY Session • ICT Logic Active")
    log_event("events.log", "Bot started")
    mt5.symbol_select(SYMBOL, True)

    while True:
        now = dt.datetime.utcnow().replace(tzinfo=None)
        now_utc = dt.datetime.now(UTC)

        in_kill = in_killzone(now_utc)
        in_news = in_news_block(now_utc)

        update_state(in_killzone=in_kill, in_news_block=in_news)

        weekday = now.weekday()
        hour = now.hour
        if weekday == 4 and hour >= 21:
            _, _, _, week_no = get_week_bounds(now_utc)
            if LAST_REPORT_WEEK != week_no:
                try:
                    generate_weekly_report(now_utc)
                    LAST_REPORT_WEEK = week_no
                except Exception as e:
                    log_event("errors.log", f"Weekly report error: {e}")

        if not in_kill:
            dbg("Outside killzone")
            time.sleep(REFRESH_SECONDS)
            continue

        if in_news:
            dbg("News block")
            time.sleep(REFRESH_SECONDS)
            continue

        manage_positions()

        if open_positions() >= MAX_POS_PER_SYMBOL:
            dbg("Max positions reached")
            time.sleep(REFRESH_SECONDS)
            continue

        setup = find_setup()
        if setup:
            send_order(setup)

        acc = mt5.account_info()
        if acc:
            update_state(equity=acc.equity)

        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    threading.Thread(target=run_dashboard, daemon=True).start()
    main()
