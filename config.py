"""
config.py — Scalping Backtest Configuration & MT5 Data Fetcher
===============================================================
Credentials are hardcoded directly below.

PARAMETER HISTORY:
  v1 (original):  1087 trades / 180 days — way too many, 31% win rate
  v2 (tight):         5 trades / 180 days — way too few, filters too strict
  v3 (balanced):  target 80–150 trades / 180 days (~1-2 quality trades/day)

KEY CHANGES FROM v2:
  - RSI bands widened back slightly: 52-68 long / 32-48 short
  - Candle body back to 0.60 (0.70 was cutting too many valid signals)
  - EMA separation reduced: 5 pips (was 8 pips — too strict)
  - ATR minimum reduced: 3 pips (was 5 pips — excluded most hours)
  - Max trades/day increased: 4 (was 3)
"""

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
#  HARDCODED CREDENTIALS
# ══════════════════════════════════════════════════════════════════════════════

MT5_LOGIN    = 463194350
MT5_PASSWORD = "abcdABCD123!@#"
MT5_SERVER   = "Exness-MT5Trial17"


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestConfig:

    # MT5 credentials
    ACCOUNT_LOGIN:    int = MT5_LOGIN
    ACCOUNT_PASSWORD: str = MT5_PASSWORD
    ACCOUNT_SERVER:   str = MT5_SERVER

    # ── Symbol ────────────────────────────────────────────────────────────────
    SYMBOL: str = "XAUUSD"

    # ── Date range (set automatically via --days) ─────────────────────────────
    START_DATE: datetime = None
    END_DATE:   datetime = None

    # ── Account ───────────────────────────────────────────────────────────────
    INITIAL_BALANCE: float = 10_000.0

    # ── EMA trend filter ──────────────────────────────────────────────────────
    EMA_FAST: int = 9
    EMA_SLOW: int = 21

    # ── RSI settings ──────────────────────────────────────────────────────────
    RSI_PERIOD: int = 14

    # Balanced: wide enough to generate signals, tight enough to avoid noise
    # Long:  RSI 52–68  (above 50 = bullish bias, below 70 = not overbought)
    # Short: RSI 32–48  (below 50 = bearish bias, above 30 = not oversold)
    RSI_LONG_MIN:  float = 52.0
    RSI_LONG_MAX:  float = 68.0
    RSI_SHORT_MIN: float = 32.0
    RSI_SHORT_MAX: float = 48.0

    # ── ATR risk management ───────────────────────────────────────────────────
    ATR_PERIOD:  int   = 14
    ATR_SL_MULT: float = 1.5
    ATR_TP_MULT: float = 2.5

    # ── Candle body filter ────────────────────────────────────────────────────
    # 0.60 = 60% of candle range must be body — filters dojis, allows signals
    CANDLE_BODY_PCT: float = 0.60

    # ── EMA separation filter ─────────────────────────────────────────────────
    # Minimum pip distance between EMA9 and EMA21
    # 5 pips = avoids tight ranging but still allows moderate trends
    EMA_MIN_SEPARATION: float = 0.0005   # 5 pips

    # ── ATR minimum filter ────────────────────────────────────────────────────
    # Minimum ATR to confirm sufficient volatility
    # 3 pips = low bar, just excludes truly dead markets (overnight Asia)
    ATR_MIN_VALUE: float = 0.0003   # 3 pips

    # ── Max trades per day ────────────────────────────────────────────────────
    # 4 trades/day = allows up to 2 London + 2 NY, prevents runaway overtrading
    MAX_TRADES_PER_DAY: int = 4

    # ── Session filter (UTC hours) ────────────────────────────────────────────
    LONDON_START: int = 7
    LONDON_END:   int = 12
    NY_START:     int = 13
    NY_END:       int = 17

    # ── Broker cost simulation ────────────────────────────────────────────────
    RISK_PERCENT:       float = 1.0
    SPREAD_PIPS:        float = 0.8
    COMMISSION_PER_LOT: float = 7.0
    SLIPPAGE_PIPS:      float = 0.2

    # ── Pip size ──────────────────────────────────────────────────────────────
    POINT: float = 0.00001

    # ── Output paths ──────────────────────────────────────────────────────────
    RESULTS_JSON: str = "backtest_results.json"
    TRADES_CSV:   str = "backtest_trades.csv"
    CHART_PNG:    str = "backtest_report.png"

    def __post_init__(self):
        if self.START_DATE is None or self.END_DATE is None:
            self.END_DATE   = datetime.now(tz=timezone.utc)
            self.START_DATE = self.END_DATE - timedelta(days=90)


def get_date_range(days: int):
    """Returns (start_date, end_date) UTC datetimes for last N days."""
    end   = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)
    return start, end


# ══════════════════════════════════════════════════════════════════════════════
#  SYMBOL AUTO-DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

def resolve_symbol(base: str, mt5) -> Optional[str]:
    all_symbols = mt5.symbols_get()
    if all_symbols is None:
        return None

    all_names  = [s.name for s in all_symbols]
    base_upper = base.upper()

    if base in all_names:
        print(f"  [MT5] Symbol: {base}")
        return base

    for name in all_names:
        if name.upper() == base_upper:
            print(f"  [MT5] Symbol: {name}  (case match)")
            return name

    candidates = sorted([n for n in all_names if n.upper().startswith(base_upper)], key=len)
    if candidates:
        print(f"  [MT5] Symbol: {candidates[0]}  (best match from: {candidates[:5]})")
        return candidates[0]

    candidates = sorted([n for n in all_names if base_upper in n.upper()], key=len)
    if candidates:
        print(f"  [MT5] Symbol: {candidates[0]}  (contains match)")
        return candidates[0]

    print(f"  [MT5] ERROR: Could not find '{base}' on this broker.")
    eur = [n for n in all_names if "EUR" in n.upper()][:20]
    print(f"  [MT5] EUR symbols available: {eur}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  MT5 DATA FETCHER
# ══════════════════════════════════════════════════════════════════════════════

def fetch_mt5_data(cfg: BacktestConfig) -> Optional[pd.DataFrame]:
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("  [MT5] MetaTrader5 not installed — run: pip install MetaTrader5")
        return None

    print("  [MT5] Initialising terminal...")
    if not mt5.initialize():
        print(f"  [MT5] Initialize failed: {mt5.last_error()}")
        print("        Make sure the MT5 desktop app is open.")
        return None

    print(f"  [MT5] Logging in as account {cfg.ACCOUNT_LOGIN} on {cfg.ACCOUNT_SERVER}...")
    if not mt5.login(cfg.ACCOUNT_LOGIN, password=cfg.ACCOUNT_PASSWORD, server=cfg.ACCOUNT_SERVER):
        err = mt5.last_error()
        mt5.shutdown()
        print(f"  [MT5] Login failed: {err}")
        return None

    info = mt5.account_info()
    print(f"  [MT5] Connected — {info.name} | Balance: {info.balance:.2f} {info.currency}")

    symbol = resolve_symbol(cfg.SYMBOL, mt5)
    if symbol is None:
        mt5.shutdown()
        return None

    mt5.symbol_select(symbol, True)

    days       = (cfg.END_DATE - cfg.START_DATE).days
    n_bars_req = int(days * 24 * 12 * 1.05)

    print(f"  [MT5] Fetching last {n_bars_req:,} M5 bars for {symbol} "
          f"(~{days} days [{cfg.START_DATE.date()} → {cfg.END_DATE.date()}])...")

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n_bars_req)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print(f"  [MT5] No data returned.")
        return None

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")
    df.index.name = "time"

    keep = [c for c in ["open", "high", "low", "close", "tick_volume"] if c in df.columns]
    df   = df[keep]

    df = df[df.index >= pd.Timestamp(cfg.START_DATE)]
    df = df[df.index <= pd.Timestamp(cfg.END_DATE)]

    if len(df) == 0:
        print(f"  [MT5] No data in requested range.")
        return None

    print(f"  [MT5] Downloaded {len(df):,} M5 bars  "
          f"[{df.index[0].date()} → {df.index[-1].date()}]")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATA GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(cfg: BacktestConfig) -> pd.DataFrame:
    print("  [Demo] Generating synthetic EURUSD M5 data...")
    np.random.seed(42)

    n_days = (cfg.END_DATE - cfg.START_DATE).days
    n_bars = n_days * 24 * 12
    times  = pd.date_range(start=cfg.START_DATE, periods=n_bars, freq="5min", tz="UTC")

    price, vol = 1.0800, 0.0003
    prices = []
    for _ in range(n_bars):
        vol   = max(0.00005, min(vol * (0.95 + 0.1 * abs(np.random.randn())), 0.002))
        ret   = -0.001 * (price - 1.0800) + np.random.randn() * vol
        price = max(1.03, min(1.15, price + ret))
        prices.append(price)

    prices = np.array(prices)
    noise  = np.random.uniform(0.00005, 0.0004, n_bars)

    df = pd.DataFrame({
        "open":        prices,
        "high":        prices + noise,
        "low":         prices - noise,
        "close":       prices + np.random.randn(n_bars) * 0.00005,
        "tick_volume": np.random.randint(100, 2000, n_bars),
    }, index=times)

    df["high"]    = df[["open", "close", "high"]].max(axis=1)
    df["low"]     = df[["open", "close", "low"]].min(axis=1)
    df.index.name = "time"

    print(f"  [Demo] Generated {len(df):,} bars.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED DATA ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def get_data(cfg: BacktestConfig, use_demo: bool = False) -> pd.DataFrame:
    if use_demo:
        return generate_synthetic_data(cfg)

    df = fetch_mt5_data(cfg)
    if df is None:
        print("  Falling back to synthetic data...\n")
        df = generate_synthetic_data(cfg)

    return df