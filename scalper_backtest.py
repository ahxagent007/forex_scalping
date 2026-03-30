"""
scalper_backtest.py — BB Scalp Backtest v8
==========================================
Strategy: Bollinger Band Squeeze-Back Scalp

SIGNAL LOGIC (all gates must pass):
  Gate 0: Not a news blackout date (BOJ / Fed / NFP days skipped)
  Gate 1: H1 ADX >= 25 — trending market, not ranging
  Gate 2: H1 EMA20 vs EMA50 — defines direction (long-only or short-only)
  Gate 3: M5 BB bandwidth >= 0.002 — bands wide enough to trade
  Gate 4: M5 ATR >= 3 pips — sufficient volatility
  Gate 5: Candle body >= 55% of range
  Gate 6: Previous bar touched outer BB
  Gate 7: Current bar closes back inside BB by >= 20% of half-width
  Gate 8: RSI < 55 for longs, > 45 for shorts
  + Session filter (London/NY only) + max 2 trades per day

KEY FIXES VS v6:
  - News blackout dates added (BOJ Dec 19 caused the -$682 loss)
  - Pip value is now configurable per symbol (USDJPY vs EURUSD)
  - Wider SL (2.0x ATR) + wider TP (3.0x ATR) = R:R 1:1.5
  - BB bandwidth filter prevents trading in low-vol squeeze periods
  - BB squeeze depth check ensures meaningful reversal candle

Usage:
    python scalper_backtest.py --days 180 --symbol USDJPY
    python scalper_backtest.py --days 90  --symbol USDJPY
    python scalper_backtest.py --days 180 --symbol EURUSD
    python scalper_backtest.py --days 60  --demo
"""

import argparse
import csv
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import BacktestConfig, get_data, get_date_range

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="BB Scalp Backtest v8")
    p.add_argument("--days",   type=int, default=180,
                   help="Number of past days to test (default: 180)")
    p.add_argument("--demo",   action="store_true",
                   help="Use synthetic data — no MT5 needed")
    p.add_argument("--symbol", type=str, default=None,
                   help="Override symbol (e.g. --symbol USDJPY)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    id:             int
    symbol:         str
    direction:      str
    open_time:      object
    close_time:     object
    entry_price:    float
    sl_price:       float
    tp_price:       float
    close_price:    float
    outcome:        str       # "WIN" or "LOSS"
    pnl_pips:       float
    pnl_dollars:    float
    balance_after:  float
    equity_after:   float
    lot_size:       float
    atr_at_entry:   float
    rsi_at_entry:   float
    bb_upper:       float
    bb_lower:       float
    bb_mid:         float
    bb_bandwidth:   float
    adx_at_entry:   float
    h1_trend:       str       # "UP" or "DOWN"
    session:        str
    bars_held:      int


@dataclass
class BacktestResult:
    trades:       list   = field(default_factory=list)
    equity_curve: list   = field(default_factory=list)
    config:       object = None


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def calc_ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def calc_rsi(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=n-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=n-1, adjust=False).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def calc_atr(hi, lo, cl, n=14):
    pc = cl.shift(1)
    tr = pd.concat([hi-lo, (hi-pc).abs(), (lo-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(com=n-1, adjust=False).mean()

def calc_bollinger(close, period, std_mult):
    """Returns (upper, middle, lower, bandwidth)."""
    mid = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    bw    = (upper - lower) / mid.replace(0, np.nan)
    return upper, mid, lower, bw

def calc_adx(high, low, close, period=14):
    """Average Directional Index — measures trend strength regardless of direction."""
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    dm_pos = pd.Series(
        np.where((high - prev_high) > (prev_low - low), np.maximum(high - prev_high, 0), 0),
        index=high.index
    )
    dm_neg = pd.Series(
        np.where((prev_low - low) > (high - prev_high), np.maximum(prev_low - low, 0), 0),
        index=high.index
    )
    atr_s  = tr.ewm(com=period-1, adjust=False).mean()
    di_pos = 100 * dm_pos.ewm(com=period-1, adjust=False).mean() / atr_s.replace(0, np.nan)
    di_neg = 100 * dm_neg.ewm(com=period-1, adjust=False).mean() / atr_s.replace(0, np.nan)
    dx     = 100 * (di_pos - di_neg).abs() / (di_pos + di_neg).replace(0, np.nan)
    return dx.ewm(com=period-1, adjust=False).mean()


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATOR PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_indicators(df_m5: pd.DataFrame, df_h1: pd.DataFrame,
                       cfg: BacktestConfig) -> pd.DataFrame:
    # ── M5 indicators ─────────────────────────────────────────────────────────
    df = df_m5.copy()
    df.index.name = "time"

    df["bb_upper"], df["bb_mid"], df["bb_lower"], df["bb_bw"] = calc_bollinger(
        df["close"], cfg.BB_PERIOD, cfg.BB_STD
    )
    # Previous bar values for BB touch detection
    df["bb_upper_prev"] = df["bb_upper"].shift(1)
    df["bb_lower_prev"] = df["bb_lower"].shift(1)
    df["low_prev"]      = df["low"].shift(1)
    df["high_prev"]     = df["high"].shift(1)

    df["rsi"] = calc_rsi(df["close"], cfg.RSI_PERIOD)
    df["atr"] = calc_atr(df["high"], df["low"], df["close"], cfg.ATR_PERIOD)

    body           = (df["close"] - df["open"]).abs()
    rng            = (df["high"]  - df["low"]).replace(0, np.nan)
    df["body_pct"] = body / rng
    df["hour"]     = df.index.hour
    df["date"]     = df.index.date
    df["date_str"] = df.index.strftime("%Y-%m-%d")

    # ── H1 indicators ─────────────────────────────────────────────────────────
    h1 = df_h1.copy()
    h1.index.name = "time"
    h1["h1_ema_fast"] = calc_ema(h1["close"], cfg.H1_EMA_FAST)
    h1["h1_ema_slow"] = calc_ema(h1["close"], cfg.H1_EMA_SLOW)
    h1["h1_adx"]      = calc_adx(h1["high"], h1["low"], h1["close"], cfg.ADX_PERIOD)

    # +1 = uptrend (EMA20 > EMA50), -1 = downtrend, 0 = flat/tangled
    h1["h1_trend"] = np.where(
        h1["h1_ema_fast"] > h1["h1_ema_slow"] * 1.0001,  1,
        np.where(h1["h1_ema_fast"] < h1["h1_ema_slow"] * 0.9999, -1, 0)
    )

    # Forward-fill H1 values into every M5 bar
    h1_ri = h1[["h1_trend", "h1_adx"]].reindex(df.index, method="ffill").fillna(0)
    df["h1_trend"] = h1_ri["h1_trend"]
    df["h1_adx"]   = h1_ri["h1_adx"]

    warmup = max(cfg.BB_PERIOD, cfg.RSI_PERIOD, cfg.ATR_PERIOD,
                 cfg.H1_EMA_SLOW, cfg.ADX_PERIOD) + 5
    return df.iloc[warmup:].copy()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def in_session(hour, cfg):
    if cfg.LONDON_START <= hour < cfg.LONDON_END: return True, "London"
    if cfg.NY_START     <= hour < cfg.NY_END:     return True, "NewYork"
    return False, ""

def calc_lot(equity, sl_dist, cfg):
    """
    Position size so RISK_PERCENT% of equity is risked.
    Uses cfg.PIP_VALUE_PER_LOT which is symbol-specific.
    """
    sl_pips = sl_dist / (cfg.POINT * 10)
    if sl_pips <= 0:
        return 0.01
    raw = (equity * cfg.RISK_PERCENT / 100) / (sl_pips * cfg.PIP_VALUE_PER_LOT)
    return round(max(0.01, min(raw, 100.0)), 2)


# ══════════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(df: pd.DataFrame, cfg: BacktestConfig) -> BacktestResult:
    result  = BacktestResult(config=cfg)
    balance = cfg.INITIAL_BALANCE
    equity  = cfg.INITIAL_BALANCE
    spread  = cfg.SPREAD_PIPS   * cfg.POINT * 10
    slip    = cfg.SLIPPAGE_PIPS * cfg.POINT * 10

    in_trade   = False
    trade_id   = 0
    open_trade = {}
    daily_tc   = {}

    # Pre-build set of blackout dates for O(1) lookup
    blackout_set = set(cfg.NEWS_BLACKOUT_DATES or [])

    df = df.copy()
    df.index.name = "time"
    eq_curve = [{"time": df.index[0], "equity": equity, "balance": balance}]

    bars   = df.reset_index()
    cols   = {c: i for i, c in enumerate(bars.columns)}
    values = bars.values
    n      = len(values)

    def g(row, col):
        return row[cols[col]]

    for i in range(1, n - 1):
        row      = values[i]
        next_row = values[i + 1]

        # ── Exit check on open trade ──────────────────────────────────────────
        if in_trade:
            n_o = g(next_row, "open"); n_h = g(next_row, "high")
            n_l = g(next_row, "low");  n_t = g(next_row, "time")
            d, tp, sl, entry, lot = (
                open_trade["direction"], open_trade["tp"],
                open_trade["sl"], open_trade["entry"], open_trade["lot"]
            )
            open_trade["bars_held"] += 1
            cp = oc = None

            if d == "BUY":
                if   n_o <= sl:               cp, oc = sl, "LOSS"
                elif n_o >= tp:               cp, oc = tp, "WIN"
                elif n_l <= sl and n_h >= tp: cp, oc = sl, "LOSS"
                elif n_l <= sl:               cp, oc = sl, "LOSS"
                elif n_h >= tp:               cp, oc = tp, "WIN"
            else:
                if   n_o >= sl:               cp, oc = sl, "LOSS"
                elif n_o <= tp:               cp, oc = tp, "WIN"
                elif n_h >= sl and n_l <= tp: cp, oc = sl, "LOSS"
                elif n_h >= sl:               cp, oc = sl, "LOSS"
                elif n_l <= tp:               cp, oc = tp, "WIN"

            if cp is not None:
                pips    = ((cp - entry) if d == "BUY" else (entry - cp)) / (cfg.POINT * 10)
                net_pnl = pips * cfg.PIP_VALUE_PER_LOT * lot - cfg.COMMISSION_PER_LOT * lot
                balance += net_pnl
                equity   = balance
                result.trades.append(Trade(
                    id=open_trade["id"], symbol=cfg.SYMBOL, direction=d,
                    open_time=open_trade["open_time"], close_time=n_t,
                    entry_price=round(entry, 5), sl_price=round(sl, 5),
                    tp_price=round(tp, 5), close_price=round(cp, 5),
                    outcome=oc, pnl_pips=round(pips, 1),
                    pnl_dollars=round(net_pnl, 2), balance_after=round(balance, 2),
                    equity_after=round(equity, 2), lot_size=lot,
                    atr_at_entry=open_trade["atr"], rsi_at_entry=open_trade["rsi"],
                    bb_upper=open_trade["bb_upper"], bb_lower=open_trade["bb_lower"],
                    bb_mid=open_trade["bb_mid"], bb_bandwidth=open_trade["bb_bw"],
                    adx_at_entry=open_trade["adx"],
                    h1_trend=open_trade["h1_trend_str"],
                    session=open_trade["session"], bars_held=open_trade["bars_held"],
                ))
                eq_curve.append({"time": n_t, "equity": equity, "balance": balance})
                in_trade = False

        if in_trade:
            continue

        # ── Gate 0: News blackout date ────────────────────────────────────────
        date_str = str(g(row, "date_str"))
        if date_str in blackout_set:
            continue

        # ── Session filter ────────────────────────────────────────────────────
        hour = int(g(row, "hour"))
        active, sess = in_session(hour, cfg)
        if not active:
            continue

        # ── Daily trade cap ───────────────────────────────────────────────────
        bar_date = g(row, "date")
        if daily_tc.get(bar_date, 0) >= cfg.MAX_TRADES_PER_DAY:
            continue

        # ── Read indicator values ─────────────────────────────────────────────
        close     = g(row, "close");  open_    = g(row, "open")
        low_prev  = g(row, "low_prev"); high_prev = g(row, "high_prev")
        bb_upper  = g(row, "bb_upper"); bb_lower  = g(row, "bb_lower")
        bb_mid    = g(row, "bb_mid");   bb_bw     = g(row, "bb_bw")
        bb_u_prev = g(row, "bb_upper_prev")
        bb_l_prev = g(row, "bb_lower_prev")
        rsi_v     = g(row, "rsi");    atr_v    = g(row, "atr")
        body_pct  = g(row, "body_pct")
        body_pct  = body_pct if not np.isnan(body_pct) else 0
        h1_trend  = float(g(row, "h1_trend"))
        h1_adx    = float(g(row, "h1_adx"))

        # ── Gate 1: ADX ≥ 25 — trending market ───────────────────────────────
        if h1_adx < cfg.ADX_MIN:
            continue

        # ── Gate 2: H1 trend must be defined ─────────────────────────────────
        if h1_trend == 0:
            continue

        # ── Gate 3: BB bandwidth ──────────────────────────────────────────────
        if np.isnan(bb_bw) or bb_bw < cfg.BB_MIN_BANDWIDTH:
            continue

        # ── Gate 4: ATR minimum ───────────────────────────────────────────────
        if atr_v < cfg.ATR_MIN:
            continue

        # ── Gate 5: Candle body ───────────────────────────────────────────────
        if body_pct < cfg.CANDLE_BODY_PCT:
            continue

        # ── Gates 6–8: BB squeeze-back signal ─────────────────────────────────
        half_width = (bb_upper - bb_lower) / 2.0

        # LONG: prev bar touched/crossed lower BB, current closes back inside meaningfully
        long_touch   = (low_prev  <= bb_l_prev)
        long_squeeze = (close > bb_lower) and \
                       (close - bb_lower >= cfg.BB_SQUEEZE_DEPTH * half_width)
        long_bull    = (close > open_)
        long_rsi_ok  = (rsi_v <= cfg.RSI_LONG_MAX)
        long_signal  = (h1_trend > 0 and long_touch and long_squeeze
                        and long_bull and long_rsi_ok)

        # SHORT: prev bar touched/crossed upper BB, current closes back inside meaningfully
        short_touch   = (high_prev >= bb_u_prev)
        short_squeeze = (close < bb_upper) and \
                        (bb_upper - close >= cfg.BB_SQUEEZE_DEPTH * half_width)
        short_bear    = (close < open_)
        short_rsi_ok  = (rsi_v >= cfg.RSI_SHORT_MIN)
        short_signal  = (h1_trend < 0 and short_touch and short_squeeze
                         and short_bear and short_rsi_ok)

        if not long_signal and not short_signal:
            continue

        # ── Execute trade ─────────────────────────────────────────────────────
        signal = "BUY" if long_signal else "SELL"
        entry  = g(next_row, "open")

        if signal == "BUY":
            entry += spread + slip
            sl = entry - atr_v * cfg.ATR_SL_MULT
            tp = entry + atr_v * cfg.ATR_TP_MULT
        else:
            entry -= spread + slip
            sl = entry + atr_v * cfg.ATR_SL_MULT
            tp = entry - atr_v * cfg.ATR_TP_MULT

        trade_id += 1
        in_trade  = True
        daily_tc[bar_date] = daily_tc.get(bar_date, 0) + 1

        open_trade = {
            "id":           trade_id,
            "direction":    signal,
            "entry":        entry,
            "sl":           sl,
            "tp":           tp,
            "lot":          calc_lot(equity, abs(entry - sl), cfg),
            "atr":          round(atr_v, 6),
            "rsi":          round(rsi_v, 2),
            "bb_upper":     round(bb_upper, 5),
            "bb_lower":     round(bb_lower, 5),
            "bb_mid":       round(bb_mid, 5),
            "bb_bw":        round(bb_bw, 6),
            "adx":          round(h1_adx, 2),
            "h1_trend_str": "UP" if h1_trend > 0 else "DOWN",
            "session":      sess,
            "open_time":    g(next_row, "time"),
            "bars_held":    0,
        }

    result.equity_curve = eq_curve
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(result: BacktestResult) -> dict:
    trades = result.trades
    cfg    = result.config
    if not trades:
        return {}

    wins   = [t for t in trades if t.outcome == "WIN"]
    losses = [t for t in trades if t.outcome == "LOSS"]
    gw     = sum(t.pnl_dollars for t in wins)        if wins   else 0
    gl     = abs(sum(t.pnl_dollars for t in losses)) if losses else 1

    eq   = [e["equity"] for e in result.equity_curve]
    peak = np.maximum.accumulate(eq)
    dd   = (np.array(eq) - peak) / peak * 100

    cw = cl = mcw = mcl = 0
    for o in [t.outcome for t in trades]:
        if o == "WIN": cw += 1; cl = 0
        else:          cl += 1; cw = 0
        mcw = max(mcw, cw); mcl = max(mcl, cl)

    monthly = {}
    for t in trades:
        k = pd.Timestamp(t.close_time).strftime("%Y-%m")
        monthly[k] = monthly.get(k, 0) + t.pnl_dollars

    daily = {}
    for t in trades:
        k = pd.Timestamp(t.close_time).strftime("%Y-%m-%d")
        daily[k] = daily.get(k, 0) + t.pnl_dollars
    dv     = list(daily.values())
    sharpe = (np.mean(dv) / np.std(dv) * np.sqrt(252)) if len(dv) > 1 and np.std(dv) > 0 else 0

    mp, rb = {}, cfg.INITIAL_BALANCE
    for k in sorted(monthly):
        mp[k] = round(monthly[k] / rb * 100, 2); rb += monthly[k]

    london = [t for t in trades if t.session == "London"]
    ny     = [t for t in trades if t.session == "NewYork"]
    up     = [t for t in trades if t.h1_trend == "UP"]
    down   = [t for t in trades if t.h1_trend == "DOWN"]
    net    = sum(t.pnl_dollars for t in trades)
    aw     = np.mean([t.pnl_dollars for t in wins])        if wins   else 0
    al     = np.mean([abs(t.pnl_dollars) for t in losses]) if losses else 0

    return {
        "symbol":              cfg.SYMBOL,
        "start_date":          cfg.START_DATE.strftime("%Y-%m-%d"),
        "end_date":            cfg.END_DATE.strftime("%Y-%m-%d"),
        "initial_balance":     cfg.INITIAL_BALANCE,
        "final_balance":       round(cfg.INITIAL_BALANCE + net, 2),
        "net_profit":          round(net, 2),
        "net_profit_pct":      round(net / cfg.INITIAL_BALANCE * 100, 1),
        "avg_monthly_return":  round(np.mean(list(mp.values())), 2) if mp else 0,
        "sharpe_ratio":        round(sharpe, 2),
        "total_trades":        len(trades),
        "total_wins":          len(wins),
        "total_losses":        len(losses),
        "win_rate":            round(len(wins) / len(trades) * 100, 1),
        "profit_factor":       round(gw / gl, 2),
        "avg_win":             round(aw, 2),
        "avg_loss":            round(al, 2),
        "avg_rr":              round(aw / al, 2) if al > 0 else 0,
        "max_drawdown_pct":    round(abs(dd.min()), 2),
        "max_consec_wins":     mcw,
        "max_consec_losses":   mcl,
        "london_trades":       len(london),
        "london_win_rate":     round(len([t for t in london if t.outcome=="WIN"])/len(london)*100,1) if london else 0,
        "ny_trades":           len(ny),
        "ny_win_rate":         round(len([t for t in ny if t.outcome=="WIN"])/len(ny)*100,1) if ny else 0,
        "uptrend_trades":      len(up),
        "uptrend_win_rate":    round(len([t for t in up if t.outcome=="WIN"])/len(up)*100,1) if up else 0,
        "downtrend_trades":    len(down),
        "downtrend_win_rate":  round(len([t for t in down if t.outcome=="WIN"])/len(down)*100,1) if down else 0,
        "monthly":             {k: round(v, 2) for k, v in sorted(monthly.items())},
        "monthly_pcts":        mp,
        "daily_pnl":           {k: round(v, 2) for k, v in sorted(daily.items())},
        "equity_curve":        [{"time": str(e["time"])[:19], "equity": round(e["equity"], 2)}
                                for e in result.equity_curve],
        "trades": [
            {
                "id":             t.id,
                "direction":      t.direction,
                "session":        t.session,
                "h1_trend":       t.h1_trend,
                "open_time":      str(pd.Timestamp(t.open_time))[:19],
                "close_time":     str(pd.Timestamp(t.close_time))[:19],
                "entry_price":    t.entry_price,
                "sl_price":       t.sl_price,
                "tp_price":       t.tp_price,
                "close_price":    t.close_price,
                "outcome":        t.outcome,
                "pnl_pips":       t.pnl_pips,
                "pnl_dollars":    t.pnl_dollars,
                "lot_size":       t.lot_size,
                "balance_after":  t.balance_after,
                "bars_held":      t.bars_held,
                "bb_upper":       t.bb_upper,
                "bb_lower":       t.bb_lower,
                "bb_mid":         t.bb_mid,
                "bb_bandwidth":   t.bb_bandwidth,
                "adx_at_entry":   t.adx_at_entry,
            }
            for t in trades
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CONSOLE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(stats: dict, cfg: BacktestConfig):
    W = 56
    def row(l, v): print(f"  {l:<30} {str(v):>22}")

    print("\n" + "=" * W)
    print(f"  BB SCALP REPORT v8  |  {stats['symbol']}")
    print(f"  {stats['start_date']}  to  {stats['end_date']}")
    print("=" * W)

    print("\n  STRATEGY CONFIG")
    row("BB period / std",        f"{cfg.BB_PERIOD} / {cfg.BB_STD}")
    row("BB min bandwidth",       cfg.BB_MIN_BANDWIDTH)
    row("BB squeeze depth",       f"{cfg.BB_SQUEEZE_DEPTH*100:.0f}% of half-width")
    row("H1 trend filter",        f"EMA{cfg.H1_EMA_FAST} vs EMA{cfg.H1_EMA_SLOW}")
    row("ADX filter (H1)",        f"ADX{cfg.ADX_PERIOD} >= {cfg.ADX_MIN}")
    row("RSI long max",           cfg.RSI_LONG_MAX)
    row("RSI short min",          cfg.RSI_SHORT_MIN)
    row("SL / TP",                f"{cfg.ATR_SL_MULT}x / {cfg.ATR_TP_MULT}x ATR")
    row("R:R",                    f"1 : {cfg.ATR_TP_MULT/cfg.ATR_SL_MULT:.2f}")
    row("Max trades/day",         cfg.MAX_TRADES_PER_DAY)
    row("News blackout dates",    len(cfg.NEWS_BLACKOUT_DATES or []))
    row("Pip value per lot",      f"${cfg.PIP_VALUE_PER_LOT}")

    print("\n  PERFORMANCE")
    row("Initial balance",        f"${stats['initial_balance']:,.2f}")
    row("Final balance",          f"${stats['final_balance']:,.2f}")
    row("Net profit",             f"${stats['net_profit']:,.2f}  ({stats['net_profit_pct']}%)")
    row("Avg monthly return",     f"+{stats['avg_monthly_return']}%")
    row("Sharpe ratio",           stats['sharpe_ratio'])

    print("\n  TRADE STATISTICS")
    row("Total trades",           stats['total_trades'])
    row("Win rate",               f"{stats['win_rate']}%")
    row("Wins / Losses",          f"{stats['total_wins']} / {stats['total_losses']}")
    row("Profit factor",          stats['profit_factor'])
    row("Avg win",                f"${stats['avg_win']}")
    row("Avg loss",               f"${stats['avg_loss']}")
    row("Avg R:R",                f"1 : {stats['avg_rr']}")

    print("\n  RISK")
    row("Max drawdown",           f"{stats['max_drawdown_pct']}%")
    row("Max consec. wins",       stats['max_consec_wins'])
    row("Max consec. losses",     stats['max_consec_losses'])

    print("\n  SESSION BREAKDOWN")
    row("London",                 f"{stats['london_trades']} trades  |  WR {stats['london_win_rate']}%")
    row("New York",               f"{stats['ny_trades']} trades  |  WR {stats['ny_win_rate']}%")

    print("\n  H1 TREND BREAKDOWN")
    row("Uptrend  (long)",        f"{stats['uptrend_trades']} trades  |  WR {stats['uptrend_win_rate']}%")
    row("Downtrend (short)",      f"{stats['downtrend_trades']} trades  |  WR {stats['downtrend_win_rate']}%")

    print("\n  MONTHLY P&L")
    for k, v in stats["monthly"].items():
        bar  = "#" * min(35, int(abs(v) / 15))
        sign = "+" if v >= 0 else "-"
        print(f"  {k}   {sign}${abs(v):>7.0f}  {bar}")

    print("=" * W + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT: JSON / CSV / CHART
# ══════════════════════════════════════════════════════════════════════════════

def save_json(stats: dict, cfg: BacktestConfig):
    with open(cfg.RESULTS_JSON, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Results saved    →  {cfg.RESULTS_JSON}")


def export_csv(result: BacktestResult, cfg: BacktestConfig):
    fields = [
        "id", "symbol", "direction", "session", "h1_trend",
        "open_time", "close_time",
        "entry_price", "sl_price", "tp_price", "close_price",
        "outcome", "pnl_pips", "pnl_dollars", "lot_size",
        "balance_after", "equity_after", "bars_held",
        "atr_at_entry", "rsi_at_entry",
        "bb_upper", "bb_lower", "bb_mid", "bb_bandwidth", "adx_at_entry",
    ]
    with open(cfg.TRADES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in result.trades:
            w.writerow({k: getattr(t, k) for k in fields})
    print(f"  Trades exported  →  {cfg.TRADES_CSV}  ({len(result.trades)} rows)")


def plot_results(result: BacktestResult, stats: dict, cfg: BacktestConfig):
    trades  = result.trades
    eq_data = result.equity_curve

    C = {
        "g": "#1D9E75", "r": "#D85A30", "b": "#378ADD",
        "gr": "#888780", "bg": "#FAFAF8", "tx": "#2C2C2A",
        "mu": "#5F5E5A", "li": "#E8E6DF",
    }
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "axes.facecolor": C["bg"],
        "figure.facecolor": "#FFFFFF", "axes.edgecolor": C["li"],
        "axes.labelcolor": C["mu"], "xtick.color": C["mu"], "ytick.color": C["mu"],
        "grid.color": C["li"], "grid.linewidth": 0.5, "axes.grid": True,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"BB Scalp v8  ·  {cfg.SYMBOL}  ·  "
        f"BB({cfg.BB_PERIOD},{cfg.BB_STD}) + H1 EMA{cfg.H1_EMA_FAST}/{cfg.H1_EMA_SLOW} + "
        f"ADX{cfg.ADX_PERIOD}≥{cfg.ADX_MIN}  ·  "
        f"{cfg.START_DATE.date()} to {cfg.END_DATE.date()}",
        fontsize=11, fontweight="bold", color=C["tx"], y=0.98
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.48, wspace=0.30,
                           left=0.07, right=0.96, top=0.93, bottom=0.07)

    times  = [e["time"] for e in eq_data]
    equity = [e["equity"] for e in eq_data]

    # Panel 1 — Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(times, equity, color=C["b"], linewidth=1.2, zorder=3)
    ax1.fill_between(times, cfg.INITIAL_BALANCE, equity,
                     where=[e >= cfg.INITIAL_BALANCE for e in equity], color=C["g"], alpha=0.12)
    ax1.fill_between(times, cfg.INITIAL_BALANCE, equity,
                     where=[e < cfg.INITIAL_BALANCE for e in equity], color=C["r"], alpha=0.15)
    ax1.axhline(cfg.INITIAL_BALANCE, color=C["gr"], linewidth=0.8, linestyle="--", alpha=0.6)
    for t in trades:
        ax1.scatter(t.close_time, t.equity_after,
                    color=C["g"] if t.outcome == "WIN" else C["r"], s=8, zorder=4, alpha=0.4)
    final = equity[-1]
    pct   = (final - cfg.INITIAL_BALANCE) / cfg.INITIAL_BALANCE * 100
    ax1.annotate(f"${final:,.0f}  ({'+' if pct >= 0 else ''}{pct:.1f}%)",
                 xy=(times[-1], final), xytext=(-140, 12),
                 textcoords="offset points", fontsize=9, fontweight="bold",
                 color=C["g"] if final >= cfg.INITIAL_BALANCE else C["r"])
    ax1.set_ylabel("Equity ($)")
    ax1.set_title("Equity curve", fontsize=10, color=C["mu"], pad=6)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Panel 2 — Drawdown
    ax2  = fig.add_subplot(gs[1, :])
    eq_a = np.array(equity)
    peak = np.maximum.accumulate(eq_a)
    dd   = (eq_a - peak) / peak * 100
    ax2.fill_between(times, dd, 0, color=C["r"], alpha=0.35, zorder=2)
    ax2.plot(times, dd, color=C["r"], linewidth=0.7, zorder=3)
    ax2.axhline(0, color=C["gr"], linewidth=0.5)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Drawdown from equity peak", fontsize=10, color=C["mu"], pad=6)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1f}%"))

    # Panel 3 — Monthly heatmap
    ax3     = fig.add_subplot(gs[2, 0])
    monthly = stats.get("monthly", {})
    if monthly:
        keys  = sorted(monthly.keys())
        years = sorted(set(k[:4] for k in keys))
        mlbls = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
        hdata = np.full((len(years), 12), np.nan)
        rb    = cfg.INITIAL_BALANCE
        for k in keys:
            y = years.index(k[:4]); m = int(k[5:7]) - 1
            hdata[y, m] = monthly[k] / rb * 100
            rb += monthly[k]
        vmax = max(abs(np.nanmin(hdata)), abs(np.nanmax(hdata)), 1)
        im   = ax3.imshow(hdata, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
        ax3.set_xticks(range(12)); ax3.set_xticklabels(mlbls, fontsize=8)
        ax3.set_yticks(range(len(years))); ax3.set_yticklabels(years, fontsize=9)
        ax3.set_title("Monthly returns (%)", fontsize=10, color=C["mu"], pad=6)
        ax3.grid(False)
        for y in range(len(years)):
            for m in range(12):
                v = hdata[y, m]
                if not np.isnan(v):
                    ax3.text(m, y, f"{v:.1f}", ha="center", va="center", fontsize=7,
                             color="white" if abs(v) > vmax * 0.55 else C["tx"])
        plt.colorbar(im, ax=ax3, fraction=0.04, pad=0.04, label="%")

    # Panel 4 — P&L distribution
    ax4      = fig.add_subplot(gs[2, 1])
    wp       = [t.pnl_dollars for t in trades if t.outcome == "WIN"]
    lp       = [abs(t.pnl_dollars) for t in trades if t.outcome == "LOSS"]
    all_vals = wp + lp
    if all_vals:
        bins = np.linspace(min(all_vals), max(all_vals), 28)
        if wp: ax4.hist(wp, bins=bins, color=C["g"], alpha=0.65, label=f"Wins ({len(wp)})")
        if lp: ax4.hist(lp, bins=bins, color=C["r"], alpha=0.65, label=f"Losses ({len(lp)})")
        if wp: ax4.axvline(np.mean(wp), color=C["g"], linewidth=1.2, linestyle="--",
                           label=f"Avg win ${np.mean(wp):.0f}")
        if lp: ax4.axvline(np.mean(lp), color=C["r"], linewidth=1.2, linestyle="--",
                           label=f"Avg loss ${np.mean(lp):.0f}")
    ax4.set_xlabel("P&L ($)")
    ax4.set_ylabel("Number of trades")
    ax4.set_title("P&L distribution", fontsize=10, color=C["mu"], pad=6)
    ax4.legend(fontsize=8, frameon=False)

    footer = (
        f"Trades: {stats['total_trades']}   WR: {stats['win_rate']}%   "
        f"PF: {stats['profit_factor']}   Max DD: {stats['max_drawdown_pct']}%   "
        f"Sharpe: {stats['sharpe_ratio']}   Avg monthly: +{stats['avg_monthly_return']}%   "
        f"News blackout: {len(cfg.NEWS_BLACKOUT_DATES or [])} dates"
    )
    fig.text(0.5, 0.012, footer, ha="center", fontsize=9, color=C["mu"],
             bbox=dict(boxstyle="round,pad=0.4", facecolor=C["li"], alpha=0.5, edgecolor="none"))

    plt.savefig(cfg.CHART_PNG, dpi=150, bbox_inches="tight")
    print(f"  Chart saved      →  {cfg.CHART_PNG}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    start_date, end_date = get_date_range(args.days)

    # Override symbol pip settings based on --symbol flag
    symbol = args.symbol.upper() if args.symbol else "USDJPY"
    is_jpy = "JPY" in symbol

    cfg = BacktestConfig(
        SYMBOL             = symbol,
        START_DATE         = start_date,
        END_DATE           = end_date,
        INITIAL_BALANCE    = 10_000.0,
        RISK_PERCENT       = 1.0,

        # BB settings
        BB_PERIOD          = 20,
        BB_STD             = 2.0,
        BB_MIN_BANDWIDTH   = 0.002,
        BB_SQUEEZE_DEPTH   = 0.20,

        # Risk
        ATR_SL_MULT        = 2.0,
        ATR_TP_MULT        = 3.0,

        # ADX + RSI
        ADX_MIN            = 25.0,
        RSI_LONG_MAX       = 55.0,
        RSI_SHORT_MIN      = 45.0,

        # Costs
        SPREAD_PIPS        = 0.8,
        COMMISSION_PER_LOT = 7.0,
        SLIPPAGE_PIPS      = 0.2,

        # Pip sizing — auto-selected by symbol type
        POINT             = 0.001    if is_jpy else 0.00001,
        PIP_VALUE_PER_LOT = 6.8     if is_jpy else 10.0,

        # Max 2 trades per day
        MAX_TRADES_PER_DAY = 2,
    )

    print("=" * 56)
    print(f"  BB SCALP BACKTEST  v8")
    print(f"  Signal   : BB({cfg.BB_PERIOD},{cfg.BB_STD}) squeeze-back")
    print(f"  Filter 1 : H1 EMA{cfg.H1_EMA_FAST}/{cfg.H1_EMA_SLOW} trend direction")
    print(f"  Filter 2 : H1 ADX{cfg.ADX_PERIOD} >= {cfg.ADX_MIN} (trending market only)")
    print(f"  SL / TP  : {cfg.ATR_SL_MULT}x / {cfg.ATR_TP_MULT}x ATR  (R:R 1:{cfg.ATR_TP_MULT/cfg.ATR_SL_MULT:.1f})")
    print(f"  Pip val  : ${cfg.PIP_VALUE_PER_LOT}/pip/lot  (POINT={cfg.POINT})")
    print(f"  Blackout : {len(cfg.NEWS_BLACKOUT_DATES)} dates (BOJ + Fed + NFP)")
    print(f"  Period   : last {args.days} days  [{start_date.date()} → {end_date.date()}]")
    print(f"  Symbol   : {cfg.SYMBOL}")
    print(f"  Mode     : {'DEMO (synthetic)' if args.demo else 'LIVE (MT5)'}")
    print("=" * 56)

    df_m5, df_h1 = get_data(cfg, use_demo=args.demo)

    print("\n  Calculating BB + ADX + H1 trend indicators...")
    df = prepare_indicators(df_m5, df_h1, cfg)

    # Summary of market conditions
    up_pct   = (df["h1_trend"] > 0).mean() * 100
    down_pct = (df["h1_trend"] < 0).mean() * 100
    adx_ok   = (df["h1_adx"]   >= cfg.ADX_MIN).mean() * 100
    bw_ok    = (df["bb_bw"]    >= cfg.BB_MIN_BANDWIDTH).mean() * 100
    blackout_bars = df["date_str"].isin(set(cfg.NEWS_BLACKOUT_DATES or [])).mean() * 100

    print(f"  H1 trend  : {up_pct:.1f}% up  |  {down_pct:.1f}% down")
    print(f"  ADX >= {cfg.ADX_MIN}: {adx_ok:.1f}% of bars qualify")
    print(f"  BB bw >= {cfg.BB_MIN_BANDWIDTH}: {bw_ok:.1f}% of bars qualify")
    print(f"  Blackout  : {blackout_bars:.1f}% of bars excluded (news dates)")

    print("  Running backtest — processing bar by bar...")
    result = run_backtest(df, cfg)

    if not result.trades:
        print("\n  No trades generated. Try in config.py:")
        print("  - Lower ADX_MIN to 20")
        print("  - Lower BB_MIN_BANDWIDTH to 0.001")
        print("  - Lower BB_SQUEEZE_DEPTH to 0.10")
        return

    print(f"  Complete — {len(result.trades)} trades processed.\n")

    stats = compute_stats(result)
    print_report(stats, cfg)
    save_json(stats, cfg)
    export_csv(result, cfg)
    print("  Generating charts...")
    plot_results(result, stats, cfg)
    print("\n  Done. Open dashboard.html and load backtest_results.json.")


if __name__ == "__main__":
    main()