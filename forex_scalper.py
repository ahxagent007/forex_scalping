"""
scalper_backtest.py — Forex Scalping Backtest Engine
=====================================================
Imports all config and data fetching from config.py.
Runs the bar-by-bar backtest and saves:
  - backtest_results.json  (loaded by dashboard.html)
  - backtest_trades.csv    (full trade log)
  - backtest_report.png    (equity curve chart)

Requirements:
    pip install MetaTrader5 pandas numpy matplotlib

Usage:
    python scalper_backtest.py
"""

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

from config import BacktestConfig, get_data

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    id:                int
    symbol:            str
    direction:         str
    open_time:         object
    close_time:        object
    entry_price:       float
    sl_price:          float
    tp_price:          float
    close_price:       float
    outcome:           str
    pnl_pips:          float
    pnl_dollars:       float
    balance_after:     float
    equity_after:      float
    lot_size:          float
    atr_at_entry:      float
    rsi_at_entry:      float
    ema_fast_at_entry: float
    ema_slow_at_entry: float
    session:           str
    bars_held:         int


@dataclass
class BacktestResult:
    trades:       list   = field(default_factory=list)
    equity_curve: list   = field(default_factory=list)
    config:       object = None


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def calc_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def calc_atr(high, low, close, period=14):
    pc = close.shift(1)
    tr = pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


# ══════════════════════════════════════════════════════════════════════════════
#  INDICATOR PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_indicators(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    df = df.copy()
    df.index.name = "time"
    df["ema_fast"] = calc_ema(df["close"], cfg.EMA_FAST)
    df["ema_slow"] = calc_ema(df["close"], cfg.EMA_SLOW)
    df["rsi"]      = calc_rsi(df["close"], cfg.RSI_PERIOD)
    df["atr"]      = calc_atr(df["high"], df["low"], df["close"], cfg.ATR_PERIOD)
    body           = (df["close"] - df["open"]).abs()
    rng            = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_pct"] = body / rng
    df["is_bull"]  = (df["close"] > df["open"]) & (df["body_pct"] >= cfg.CANDLE_BODY_PCT)
    df["is_bear"]  = (df["close"] < df["open"]) & (df["body_pct"] >= cfg.CANDLE_BODY_PCT)
    df["hour"]     = df.index.hour
    warmup = max(cfg.EMA_SLOW, cfg.RSI_PERIOD, cfg.ATR_PERIOD) + 5
    return df.iloc[warmup:].copy()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def in_session(hour, cfg):
    if cfg.LONDON_START <= hour < cfg.LONDON_END:
        return True, "London"
    if cfg.NY_START <= hour < cfg.NY_END:
        return True, "NewYork"
    return False, ""

def calc_lot(equity, sl_distance, cfg):
    risk_amount = equity * (cfg.RISK_PERCENT / 100)
    sl_pips     = sl_distance / (cfg.POINT * 10)
    if sl_pips <= 0:
        return 0.01
    raw = risk_amount / (sl_pips * 10.0)
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

    df = df.copy()
    df.index.name = "time"   # fix for KeyError: 'time'

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

        if in_trade:
            n_open = g(next_row, "open")
            n_high = g(next_row, "high")
            n_low  = g(next_row, "low")
            n_time = g(next_row, "time")
            d, tp, sl, entry, lot = (
                open_trade["direction"], open_trade["tp"], open_trade["sl"],
                open_trade["entry"],     open_trade["lot"]
            )
            open_trade["bars_held"] += 1
            close_price = outcome = None

            if d == "BUY":
                if   n_open <= sl:               close_price, outcome = sl, "LOSS"
                elif n_open >= tp:               close_price, outcome = tp, "WIN"
                elif n_low <= sl and n_high >= tp: close_price, outcome = sl, "LOSS"
                elif n_low  <= sl:               close_price, outcome = sl, "LOSS"
                elif n_high >= tp:               close_price, outcome = tp, "WIN"
            else:
                if   n_open >= sl:               close_price, outcome = sl, "LOSS"
                elif n_open <= tp:               close_price, outcome = tp, "WIN"
                elif n_high >= sl and n_low <= tp: close_price, outcome = sl, "LOSS"
                elif n_high >= sl:               close_price, outcome = sl, "LOSS"
                elif n_low  <= tp:               close_price, outcome = tp, "WIN"

            if close_price is not None:
                pips    = ((close_price - entry) if d == "BUY" else (entry - close_price)) / (cfg.POINT * 10)
                net_pnl = pips * 10.0 * lot - cfg.COMMISSION_PER_LOT * lot
                balance += net_pnl
                equity   = balance
                result.trades.append(Trade(
                    id=open_trade["id"], symbol=cfg.SYMBOL, direction=d,
                    open_time=open_trade["open_time"], close_time=n_time,
                    entry_price=round(entry,5), sl_price=round(sl,5),
                    tp_price=round(tp,5), close_price=round(close_price,5),
                    outcome=outcome, pnl_pips=round(pips,1),
                    pnl_dollars=round(net_pnl,2), balance_after=round(balance,2),
                    equity_after=round(equity,2), lot_size=lot,
                    atr_at_entry=open_trade["atr"], rsi_at_entry=open_trade["rsi"],
                    ema_fast_at_entry=open_trade["ema_fast"],
                    ema_slow_at_entry=open_trade["ema_slow"],
                    session=open_trade["session"], bars_held=open_trade["bars_held"],
                ))
                eq_curve.append({"time": n_time, "equity": equity, "balance": balance})
                in_trade = False

        if in_trade:
            continue

        hour, (active, sess) = int(g(row, "hour")), in_session(int(g(row, "hour")), cfg)
        if not active:
            continue

        ema_f, ema_s = g(row, "ema_fast"), g(row, "ema_slow")
        rsi_v, atr_v = g(row, "rsi"),      g(row, "atr")
        is_b,  is_br = bool(g(row, "is_bull")), bool(g(row, "is_bear"))

        signal = None
        if ema_f > ema_s and cfg.RSI_LONG_MIN  <= rsi_v <= cfg.RSI_LONG_MAX  and is_b:  signal = "BUY"
        elif ema_f < ema_s and cfg.RSI_SHORT_MIN <= rsi_v <= cfg.RSI_SHORT_MAX and is_br: signal = "SELL"
        if signal is None:
            continue

        entry = g(next_row, "open")
        if signal == "BUY":
            entry += spread + slip; sl = entry - atr_v * cfg.ATR_SL_MULT; tp = entry + atr_v * cfg.ATR_TP_MULT
        else:
            entry -= spread + slip; sl = entry + atr_v * cfg.ATR_SL_MULT; tp = entry - atr_v * cfg.ATR_TP_MULT

        trade_id += 1
        in_trade  = True
        open_trade = {
            "id": trade_id, "direction": signal, "entry": entry,
            "sl": sl, "tp": tp, "lot": calc_lot(equity, abs(entry - sl), cfg),
            "atr": round(atr_v,6), "rsi": round(rsi_v,2),
            "ema_fast": round(ema_f,5), "ema_slow": round(ema_s,5),
            "session": sess, "open_time": g(next_row, "time"), "bars_held": 0,
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
    gross_win  = sum(t.pnl_dollars for t in wins)        if wins   else 0
    gross_loss = abs(sum(t.pnl_dollars for t in losses)) if losses else 1

    eq   = [e["equity"] for e in result.equity_curve]
    peak = np.maximum.accumulate(eq)
    dd   = (np.array(eq) - peak) / peak * 100

    consec_w = consec_l = cw = cl = 0
    for o in [t.outcome for t in trades]:
        if o == "WIN": cw += 1; cl = 0
        else:          cl += 1; cw = 0
        consec_w = max(consec_w, cw); consec_l = max(consec_l, cl)

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

    monthly_pcts, rb = {}, cfg.INITIAL_BALANCE
    for k in sorted(monthly):
        monthly_pcts[k] = round(monthly[k] / rb * 100, 2)
        rb += monthly[k]

    london = [t for t in trades if t.session == "London"]
    ny     = [t for t in trades if t.session == "NewYork"]
    net    = sum(t.pnl_dollars for t in trades)

    avg_win_val  = np.mean([t.pnl_dollars for t in wins])        if wins   else 0
    avg_loss_val = np.mean([abs(t.pnl_dollars) for t in losses]) if losses else 0

    return {
        "symbol":             cfg.SYMBOL,
        "start_date":         cfg.START_DATE.strftime("%Y-%m-%d"),
        "end_date":           cfg.END_DATE.strftime("%Y-%m-%d"),
        "initial_balance":    cfg.INITIAL_BALANCE,
        "final_balance":      round(cfg.INITIAL_BALANCE + net, 2),
        "net_profit":         round(net, 2),
        "net_profit_pct":     round(net / cfg.INITIAL_BALANCE * 100, 1),
        "avg_monthly_return": round(np.mean(list(monthly_pcts.values())), 2) if monthly_pcts else 0,
        "sharpe_ratio":       round(sharpe, 2),
        "total_trades":       len(trades),
        "total_wins":         len(wins),
        "total_losses":       len(losses),
        "win_rate":           round(len(wins) / len(trades) * 100, 1),
        "profit_factor":      round(gross_win / gross_loss, 2),
        "avg_win":            round(avg_win_val, 2),
        "avg_loss":           round(avg_loss_val, 2),
        "avg_rr":             round(avg_win_val / avg_loss_val, 2) if avg_loss_val > 0 else 0,
        "max_drawdown_pct":   round(abs(dd.min()), 2),
        "max_consec_wins":    consec_w,
        "max_consec_losses":  consec_l,
        "london_trades":      len(london),
        "london_win_rate":    round(len([t for t in london if t.outcome=="WIN"])/len(london)*100,1) if london else 0,
        "ny_trades":          len(ny),
        "ny_win_rate":        round(len([t for t in ny if t.outcome=="WIN"])/len(ny)*100,1) if ny else 0,
        "monthly":            {k: round(v, 2) for k, v in sorted(monthly.items())},
        "monthly_pcts":       monthly_pcts,
        "daily_pnl":          {k: round(v, 2) for k, v in sorted(daily.items())},
        "equity_curve": [
            {"time": str(e["time"])[:19], "equity": round(e["equity"], 2)}
            for e in result.equity_curve
        ],
        "trades": [
            {
                "id":            t.id,
                "direction":     t.direction,
                "session":       t.session,
                "open_time":     str(pd.Timestamp(t.open_time))[:19],
                "close_time":    str(pd.Timestamp(t.close_time))[:19],
                "entry_price":   t.entry_price,
                "sl_price":      t.sl_price,
                "tp_price":      t.tp_price,
                "close_price":   t.close_price,
                "outcome":       t.outcome,
                "pnl_pips":      t.pnl_pips,
                "pnl_dollars":   t.pnl_dollars,
                "lot_size":      t.lot_size,
                "balance_after": t.balance_after,
                "bars_held":     t.bars_held,
            }
            for t in trades
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CONSOLE REPORT
# ══════════════════════════════════════════════════════════════════════════════

def print_report(stats: dict):
    W = 56
    def row(lbl, val): print(f"  {lbl:<30} {str(val):>22}")
    print("\n" + "=" * W)
    print(f"  BACKTEST REPORT  |  {stats['symbol']}  M5")
    print(f"  {stats['start_date']}  to  {stats['end_date']}")
    print("=" * W)
    print("\n  PERFORMANCE")
    row("Initial balance",     f"${stats['initial_balance']:,.2f}")
    row("Final balance",       f"${stats['final_balance']:,.2f}")
    row("Net profit",          f"${stats['net_profit']:,.2f}  ({stats['net_profit_pct']}%)")
    row("Avg monthly return",  f"+{stats['avg_monthly_return']}%")
    row("Sharpe ratio",        stats['sharpe_ratio'])
    print("\n  TRADE STATISTICS")
    row("Total trades",        stats['total_trades'])
    row("Win rate",            f"{stats['win_rate']}%")
    row("Wins / Losses",       f"{stats['total_wins']} / {stats['total_losses']}")
    row("Profit factor",       stats['profit_factor'])
    row("Average win",         f"${stats['avg_win']}")
    row("Average loss",        f"${stats['avg_loss']}")
    row("Avg risk : reward",   f"1 : {stats['avg_rr']}")
    print("\n  RISK")
    row("Max drawdown",        f"{stats['max_drawdown_pct']}%")
    row("Max consec. wins",    stats['max_consec_wins'])
    row("Max consec. losses",  stats['max_consec_losses'])
    print("\n  SESSION BREAKDOWN")
    row("London",              f"{stats['london_trades']} trades  |  WR {stats['london_win_rate']}%")
    row("New York",            f"{stats['ny_trades']} trades  |  WR {stats['ny_win_rate']}%")
    print("\n  MONTHLY P&L")
    for k, v in stats["monthly"].items():
        bar = "#" * min(35, int(abs(v) / 40))
        print(f"  {k}   {'+'if v>=0 else '-'}${abs(v):>7.0f}  {bar}")
    print("=" * W + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

def save_json(stats: dict, cfg: BacktestConfig):
    with open(cfg.RESULTS_JSON, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Results saved    →  {cfg.RESULTS_JSON}")

def export_csv(result: BacktestResult, cfg: BacktestConfig):
    fields = [
        "id","symbol","direction","session","open_time","close_time",
        "entry_price","sl_price","tp_price","close_price","outcome",
        "pnl_pips","pnl_dollars","lot_size","balance_after","equity_after",
        "bars_held","atr_at_entry","rsi_at_entry","ema_fast_at_entry","ema_slow_at_entry",
    ]
    with open(cfg.TRADES_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in result.trades:
            w.writerow({k: getattr(t, k) for k in fields})
    print(f"  Trades exported  →  {cfg.TRADES_CSV}  ({len(result.trades)} rows)")

def plot_results(result: BacktestResult, stats: dict, cfg: BacktestConfig):
    trades = result.trades; eq_data = result.equity_curve
    C = {"g":"#1D9E75","r":"#D85A30","b":"#378ADD","gr":"#888780",
         "bg":"#FAFAF8","tx":"#2C2C2A","mu":"#5F5E5A","li":"#E8E6DF"}
    plt.rcParams.update({
        "font.family":"DejaVu Sans","axes.facecolor":C["bg"],
        "figure.facecolor":"#FFFFFF","axes.edgecolor":C["li"],
        "axes.labelcolor":C["mu"],"xtick.color":C["mu"],"ytick.color":C["mu"],
        "grid.color":C["li"],"grid.linewidth":0.5,"axes.grid":True,
        "axes.spines.top":False,"axes.spines.right":False,
    })
    fig = plt.figure(figsize=(16,14))
    fig.suptitle(f"Scalping Backtest · {cfg.SYMBOL} · {cfg.START_DATE.date()} to {cfg.END_DATE.date()}",
                 fontsize=13, fontweight="bold", color=C["tx"], y=0.98)
    gs = gridspec.GridSpec(3,2,figure=fig,hspace=0.48,wspace=0.30,left=0.07,right=0.96,top=0.93,bottom=0.07)

    times  = [e["time"] for e in eq_data]
    equity = [e["equity"] for e in eq_data]

    ax1 = fig.add_subplot(gs[0,:])
    ax1.plot(times,equity,color=C["b"],linewidth=1.2,zorder=3)
    ax1.fill_between(times,cfg.INITIAL_BALANCE,equity,where=[e>=cfg.INITIAL_BALANCE for e in equity],color=C["g"],alpha=0.12)
    ax1.fill_between(times,cfg.INITIAL_BALANCE,equity,where=[e<cfg.INITIAL_BALANCE for e in equity],color=C["r"],alpha=0.15)
    ax1.axhline(cfg.INITIAL_BALANCE,color=C["gr"],linewidth=0.8,linestyle="--",alpha=0.6)
    for t in trades:
        ax1.scatter(t.close_time,t.equity_after,color=C["g"] if t.outcome=="WIN" else C["r"],s=8,zorder=4,alpha=0.4)
    final=equity[-1]; pct=(final-cfg.INITIAL_BALANCE)/cfg.INITIAL_BALANCE*100
    ax1.annotate(f"${final:,.0f}  ({'+' if pct>=0 else ''}{pct:.1f}%)",
                 xy=(times[-1],final),xytext=(-140,12),textcoords="offset points",
                 fontsize=9,fontweight="bold",color=C["g"] if final>=cfg.INITIAL_BALANCE else C["r"])
    ax1.set_ylabel("Equity ($)"); ax1.set_title("Equity curve",fontsize=10,color=C["mu"],pad=6)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"${x:,.0f}"))

    ax2=fig.add_subplot(gs[1,:]); eq_a=np.array(equity); peak=np.maximum.accumulate(eq_a); dd=(eq_a-peak)/peak*100
    ax2.fill_between(times,dd,0,color=C["r"],alpha=0.35,zorder=2)
    ax2.plot(times,dd,color=C["r"],linewidth=0.7,zorder=3)
    ax2.axhline(0,color=C["gr"],linewidth=0.5); ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Drawdown from equity peak",fontsize=10,color=C["mu"],pad=6)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"{x:.1f}%"))

    ax3=fig.add_subplot(gs[2,0]); monthly=stats.get("monthly",{})
    if monthly:
        keys=sorted(monthly.keys()); years=sorted(set(k[:4] for k in keys))
        mlbls=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        hdata=np.full((len(years),12),np.nan); rb=cfg.INITIAL_BALANCE
        for k in keys:
            y=years.index(k[:4]); m=int(k[5:7])-1
            hdata[y,m]=monthly[k]/rb*100; rb+=monthly[k]
        vmax=max(abs(np.nanmin(hdata)),abs(np.nanmax(hdata)),1)
        im=ax3.imshow(hdata,cmap="RdYlGn",aspect="auto",vmin=-vmax,vmax=vmax)
        ax3.set_xticks(range(12)); ax3.set_xticklabels(mlbls,fontsize=8)
        ax3.set_yticks(range(len(years))); ax3.set_yticklabels(years,fontsize=9)
        ax3.set_title("Monthly returns (%)",fontsize=10,color=C["mu"],pad=6); ax3.grid(False)
        for y in range(len(years)):
            for m in range(12):
                v=hdata[y,m]
                if not np.isnan(v):
                    ax3.text(m,y,f"{v:.1f}",ha="center",va="center",fontsize=7,
                             color="white" if abs(v)>vmax*0.55 else C["tx"])
        plt.colorbar(im,ax=ax3,fraction=0.04,pad=0.04,label="%")

    ax4=fig.add_subplot(gs[2,1])
    wp=[t.pnl_dollars for t in trades if t.outcome=="WIN"]
    lp=[abs(t.pnl_dollars) for t in trades if t.outcome=="LOSS"]
    av=wp+lp
    if av:
        bins=np.linspace(min(av),max(av),28)
        if wp: ax4.hist(wp,bins=bins,color=C["g"],alpha=0.65,label=f"Wins ({len(wp)})")
        if lp: ax4.hist(lp,bins=bins,color=C["r"],alpha=0.65,label=f"Losses ({len(lp)})")
        if wp: ax4.axvline(np.mean(wp),color=C["g"],linewidth=1.2,linestyle="--",label=f"Avg win ${np.mean(wp):.0f}")
        if lp: ax4.axvline(np.mean(lp),color=C["r"],linewidth=1.2,linestyle="--",label=f"Avg loss ${np.mean(lp):.0f}")
    ax4.set_xlabel("P&L ($)"); ax4.set_ylabel("Number of trades")
    ax4.set_title("P&L distribution",fontsize=10,color=C["mu"],pad=6); ax4.legend(fontsize=8,frameon=False)

    footer=(f"Trades: {stats['total_trades']}   Win rate: {stats['win_rate']}%   "
            f"Profit factor: {stats['profit_factor']}   Max DD: {stats['max_drawdown_pct']}%   "
            f"Sharpe: {stats['sharpe_ratio']}   Avg monthly: +{stats['avg_monthly_return']}%")
    fig.text(0.5,0.012,footer,ha="center",fontsize=9,color=C["mu"],
             bbox=dict(boxstyle="round,pad=0.4",facecolor=C["li"],alpha=0.5,edgecolor="none"))
    plt.savefig(cfg.CHART_PNG,dpi=150,bbox_inches="tight")
    print(f"  Chart saved      →  {cfg.CHART_PNG}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    cfg = BacktestConfig(
        ACCOUNT_LOGIN    = 0,           # your MT5 login
        ACCOUNT_PASSWORD = "",          # your MT5 password
        ACCOUNT_SERVER   = "",          # e.g. "ICMarkets-Demo"
        SYMBOL           = "EURUSD",
        START_DATE       = datetime(2023, 1, 1,  tzinfo=timezone.utc),
        END_DATE         = datetime(2024, 12, 31, tzinfo=timezone.utc),
        INITIAL_BALANCE  = 10_000.0,
        RISK_PERCENT     = 1.0,
        ATR_SL_MULT      = 1.5,
        ATR_TP_MULT      = 2.5,
        SPREAD_PIPS      = 0.8,
        COMMISSION_PER_LOT = 7.0,
        SLIPPAGE_PIPS    = 0.2,
    )

    USE_DEMO = True   # set False to pull real data from MT5

    print("=" * 56)
    print("  FOREX SCALPING BACKTEST ENGINE")
    print("=" * 56)

    df_raw = get_data(cfg, use_demo=USE_DEMO)

    print("\n  Calculating indicators...")
    df = prepare_indicators(df_raw, cfg)

    print("  Running backtest — processing bar by bar...")
    result = run_backtest(df, cfg)

    if not result.trades:
        print("\n  No trades generated. Check date range / session hours.")
        return

    print(f"  Complete — {len(result.trades)} trades processed.\n")

    stats = compute_stats(result)
    print_report(stats)
    save_json(stats, cfg)
    export_csv(result, cfg)
    print("  Generating charts...")
    plot_results(result, stats, cfg)
    print("\n  Done. Open dashboard.html in your browser to view results.")


if __name__ == "__main__":
    main()