"""
config.py — Scalping Backtest Configuration & MT5 Data Fetcher v8
=================================================================
Strategy: Bollinger Band Squeeze-Back Scalp with News Blackout

WHAT'S NEW IN v8:
  - NEWS_BLACKOUT_DATES: skip trading on BOJ / Fed meeting days
    (December 2025 loss was caused entirely by BOJ surprise volatility)
  - USDJPY pip value corrected: POINT=0.001, PIP_VALUE_PER_LOT=6.8
  - All other parameters carried over from v7

HOW TO USE:
  1. Keep MT5 desktop open and logged in
  2. Run: python scalper_backtest.py --days 180 --symbol USDJPY
  3. For EURUSD: python scalper_backtest.py --days 180 --symbol EURUSD
     (set POINT=0.00001, PIP_VALUE_PER_LOT=10.0 for EURUSD)
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
#  CREDENTIALS
# ══════════════════════════════════════════════════════════════════════════════

MT5_LOGIN    = 463194350
MT5_PASSWORD = "abcdABCD123!@#"
MT5_SERVER   = "Exness-MT5Trial17"


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestConfig:

    # ── MT5 credentials ───────────────────────────────────────────────────────
    ACCOUNT_LOGIN:    int = MT5_LOGIN
    ACCOUNT_PASSWORD: str = MT5_PASSWORD
    ACCOUNT_SERVER:   str = MT5_SERVER

    # ── Symbol ────────────────────────────────────────────────────────────────
    SYMBOL: str = "USDJPY"

    # ── Date range (set by --days argument) ───────────────────────────────────
    START_DATE: datetime = None
    END_DATE:   datetime = None

    # ── Account ───────────────────────────────────────────────────────────────
    INITIAL_BALANCE: float = 10_000.0

    # ── Bollinger Bands (M5) ──────────────────────────────────────────────────
    BB_PERIOD: int   = 20
    BB_STD:    float = 2.0

    # Minimum BB bandwidth = (upper-lower)/mid
    # Filters out low-volatility squeeze periods that generate false signals
    BB_MIN_BANDWIDTH: float = 0.002

    # How far inside the band close must be (fraction of half-bandwidth)
    # 0.20 = close must be at least 20% of half-width inside the band
    BB_SQUEEZE_DEPTH: float = 0.20

    # ── H1 trend filter ───────────────────────────────────────────────────────
    # H1 EMA20 > EMA50 = uptrend  → only BUY signals
    # H1 EMA20 < EMA50 = downtrend → only SELL signals
    H1_EMA_FAST: int = 20
    H1_EMA_SLOW: int = 50

    # ── ADX regime filter (H1) ────────────────────────────────────────────────
    # ADX > 25 = clearly trending market → trade
    # ADX < 25 = ranging/choppy → skip all signals
    ADX_PERIOD: int   = 14
    ADX_MIN:    float = 25.0

    # ── RSI confirmation (M5) — asymmetric ───────────────────────────────────
    RSI_PERIOD:    int   = 14
    RSI_LONG_MAX:  float = 55.0   # long: RSI must be below 55 (not overbought)
    RSI_SHORT_MIN: float = 45.0   # short: RSI must be above 45 (not oversold)

    # ── ATR risk management (M5) ──────────────────────────────────────────────
    ATR_PERIOD:  int   = 14
    ATR_SL_MULT: float = 2.0    # wider SL to avoid wick clipping
    ATR_TP_MULT: float = 3.0    # R:R = 1:1.5
    ATR_MIN:     float = 0.0003  # 3 pips minimum volatility (in price terms)

    # ── Candle body filter ────────────────────────────────────────────────────
    CANDLE_BODY_PCT: float = 0.55

    # ── Trade management ──────────────────────────────────────────────────────
    MAX_TRADES_PER_DAY: int = 2

    # ── News blackout dates ───────────────────────────────────────────────────
    # Trading is suspended on these dates to avoid central bank event spikes.
    # Add or remove dates as needed for your test period.
    # Format: "YYYY-MM-DD"
    NEWS_BLACKOUT_DATES: List[str] = None

    # ── Sessions (UTC hours) ──────────────────────────────────────────────────
    LONDON_START: int = 7
    LONDON_END:   int = 12
    NY_START:     int = 13
    NY_END:       int = 17

    # ── Pip sizing ────────────────────────────────────────────────────────────
    # USDJPY: POINT = 0.001,   PIP_VALUE_PER_LOT = ~6.8 USD (at ~148 rate)
    # EURUSD: POINT = 0.00001, PIP_VALUE_PER_LOT = 10.0 USD
    # GBPUSD: POINT = 0.00001, PIP_VALUE_PER_LOT = 10.0 USD
    POINT:              float = 0.001    # USDJPY default
    PIP_VALUE_PER_LOT:  float = 6.8     # USD per pip per standard lot (USDJPY)

    # ── Broker costs ──────────────────────────────────────────────────────────
    RISK_PERCENT:       float = 1.0
    SPREAD_PIPS:        float = 0.8
    COMMISSION_PER_LOT: float = 7.0
    SLIPPAGE_PIPS:      float = 0.2

    # ── Output paths ──────────────────────────────────────────────────────────
    RESULTS_JSON: str = "backtest_results.json"
    TRADES_CSV:   str = "backtest_trades.csv"
    CHART_PNG:    str = "backtest_report.png"

    def __post_init__(self):
        # Default date range: last 90 days
        if self.START_DATE is None or self.END_DATE is None:
            self.END_DATE   = datetime.now(tz=timezone.utc)
            self.START_DATE = self.END_DATE - timedelta(days=90)

        # Default news blackout dates: BOJ + Fed meeting dates for 2025-2026
        if self.NEWS_BLACKOUT_DATES is None:
            self.NEWS_BLACKOUT_DATES = [
                # Fed FOMC meetings
                "2025-09-17", "2025-09-18",
                "2025-10-28", "2025-10-29",
                "2025-11-06", "2025-11-07",
                "2025-12-09", "2025-12-10",
                "2026-01-28", "2026-01-29",
                "2026-03-18", "2026-03-19",
                # BOJ policy meetings
                "2025-10-22", "2025-10-23",
                "2025-12-18", "2025-12-19",   # caused December -$682 loss
                "2026-01-23", "2026-01-24",
                "2026-03-18", "2026-03-19",
                # US NFP (first Friday each month — high vol)
                "2025-10-03",
                "2025-11-07",
                "2025-12-05",
                "2026-01-09",
                "2026-02-06",
                "2026-03-06",
            ]


def get_date_range(days: int):
    """Returns (start_date, end_date) as UTC datetimes for the last N days."""
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
        print(f"  [MT5] Symbol: {base}"); return base
    for name in all_names:
        if name.upper() == base_upper:
            print(f"  [MT5] Symbol: {name}"); return name
    candidates = sorted([n for n in all_names if n.upper().startswith(base_upper)], key=len)
    if candidates:
        print(f"  [MT5] Symbol: {candidates[0]}  (matched from {candidates[:3]})"); return candidates[0]
    candidates = sorted([n for n in all_names if base_upper in n.upper()], key=len)
    if candidates:
        print(f"  [MT5] Symbol: {candidates[0]}  (contains match)"); return candidates[0]
    print(f"  [MT5] ERROR: Cannot find '{base}'")
    print(f"  [MT5] Available symbols containing base currency: "
          f"{[n for n in all_names if base_upper[:3] in n.upper()][:15]}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  MT5 DATA FETCHER
# ══════════════════════════════════════════════════════════════════════════════

def fetch_mt5_data(cfg: BacktestConfig) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Fetches M5 and H1 bars. Uses copy_rates_from_pos — works on trial accounts."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("  [MT5] Run: pip install MetaTrader5"); return None

    print("  [MT5] Initialising terminal...")
    if not mt5.initialize():
        print(f"  [MT5] Init failed: {mt5.last_error()}"); return None

    print(f"  [MT5] Logging in as {cfg.ACCOUNT_LOGIN} on {cfg.ACCOUNT_SERVER}...")
    if not mt5.login(cfg.ACCOUNT_LOGIN, password=cfg.ACCOUNT_PASSWORD, server=cfg.ACCOUNT_SERVER):
        print(f"  [MT5] Login failed: {mt5.last_error()}"); mt5.shutdown(); return None

    info = mt5.account_info()
    print(f"  [MT5] Connected — {info.name} | Balance: {info.balance:.2f} {info.currency}")

    symbol = resolve_symbol(cfg.SYMBOL, mt5)
    if symbol is None:
        mt5.shutdown(); return None
    mt5.symbol_select(symbol, True)

    days  = (cfg.END_DATE - cfg.START_DATE).days
    n_m5  = int(days * 24 * 12 * 1.05)
    n_h1  = int(days * 24 * 1.1) + 200   # extra for ADX warmup

    # Quick probe to confirm data is available
    probe = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 1)
    if probe is None or len(probe) == 0:
        print("  [MT5] M5 data not available on this account.")
        mt5.shutdown(); return None

    print(f"  [MT5] Fetching M5  (~{days} days, up to {n_m5:,} bars)...")
    rates_m5 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n_m5)

    print(f"  [MT5] Fetching H1  (~{days} days + ADX/EMA warmup)...")
    rates_h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, n_h1)

    mt5.shutdown()

    if rates_m5 is None or len(rates_m5) == 0:
        print("  [MT5] No M5 data returned."); return None
    if rates_h1 is None or len(rates_h1) == 0:
        print("  [MT5] No H1 data returned."); return None

    def to_df(rates):
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time"); df.index.name = "time"
        keep = [c for c in ["open","high","low","close","tick_volume"] if c in df.columns]
        return df[keep]

    df_m5 = to_df(rates_m5)
    df_h1 = to_df(rates_h1)

    # Trim M5 to requested date range
    df_m5 = df_m5[(df_m5.index >= pd.Timestamp(cfg.START_DATE)) &
                   (df_m5.index <= pd.Timestamp(cfg.END_DATE))]

    if len(df_m5) == 0:
        print(f"  [MT5] No M5 data in range. Try --days 90 or --days 120.")
        return None

    print(f"  [MT5] M5: {len(df_m5):,} bars  [{df_m5.index[0].date()} → {df_m5.index[-1].date()}]")
    print(f"  [MT5] H1: {len(df_h1):,} bars  [{df_h1.index[0].date()} → {df_h1.index[-1].date()}]")
    return df_m5, df_h1


# ══════════════════════════════════════════════════════════════════════════════
#  SYNTHETIC DATA GENERATOR  (demo / offline testing)
# ══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(cfg: BacktestConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print("  [Demo] Generating synthetic M5 + H1 data...")
    np.random.seed(42)
    n_days = (cfg.END_DATE - cfg.START_DATE).days
    n_m5   = n_days * 24 * 12
    times  = pd.date_range(start=cfg.START_DATE, periods=n_m5, freq="5min", tz="UTC")

    # USDJPY-like price simulation (~148 range)
    base_price = 148.00 if "JPY" in cfg.SYMBOL.upper() else 1.0800
    price, vol = base_price, 0.03 if "JPY" in cfg.SYMBOL.upper() else 0.0003
    prices = []
    for _ in range(n_m5):
        vol   = max(0.005 if "JPY" in cfg.SYMBOL.upper() else 0.00005,
                    min(vol * (0.95 + 0.1 * abs(np.random.randn())), 0.2 if "JPY" in cfg.SYMBOL.upper() else 0.002))
        ret   = -0.001 * (price - base_price) + np.random.randn() * vol
        price = max(base_price * 0.9, min(base_price * 1.1, price + ret))
        prices.append(price)

    prices = np.array(prices)
    noise  = np.random.uniform(0.005 if "JPY" in cfg.SYMBOL.upper() else 0.00005,
                               0.04  if "JPY" in cfg.SYMBOL.upper() else 0.0004, n_m5)
    df_m5  = pd.DataFrame({
        "open":  prices, "high": prices + noise, "low": prices - noise,
        "close": prices + np.random.randn(n_m5) * (0.005 if "JPY" in cfg.SYMBOL.upper() else 0.00005),
        "tick_volume": np.random.randint(100, 2000, n_m5),
    }, index=times)
    df_m5["high"] = df_m5[["open","close","high"]].max(axis=1)
    df_m5["low"]  = df_m5[["open","close","low"]].min(axis=1)
    df_m5.index.name = "time"

    df_h1 = df_m5.resample("1h").agg({
        "open":"first","high":"max","low":"min","close":"last","tick_volume":"sum"
    }).dropna()
    df_h1.index.name = "time"
    print(f"  [Demo] M5: {len(df_m5):,}  |  H1: {len(df_h1):,}")
    return df_m5, df_h1


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED DATA ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def get_data(cfg: BacktestConfig, use_demo: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (df_m5, df_h1). Falls back to synthetic data if MT5 fails."""
    if use_demo:
        return generate_synthetic_data(cfg)
    result = fetch_mt5_data(cfg)
    if result is None:
        print("  Falling back to synthetic data...\n")
        return generate_synthetic_data(cfg)
    return result