"""
OMEGA Forex v3 — Hourly inference script
GitHub Actions: omega-forex-signals.yml (:05 every hour)

Signal flow:
  1. Fetch ForexFactory calendar (news masking)
  2. Fetch USD/JPY first (Standard Candle anchor)
  3. Fetch 500 bars per active pair (Twelve Data)
  4. Strip weekends and gap artifacts
  5. Build 46 features (technical + session + lag + regime + 3 USD/JPY anchor)
  6. Scale with train-fitted scaler
  7. Build seq_len=24 window
  8. Run model (CNN → RegimeGate → Transformer) with temperature T
  9. Apply per-pair confidence threshold from config.json
 10. RedShift + Spread-aware + Drawdown-aware Kelly sizing
 11. Skip signals masked by high-impact news events
 12. Resolve outcomes from 55 min ago → append to forex_signals_log.csv
 13. Post signal table + resolved outcomes to Discord
 14. git commit + pull rebase + push
"""

import os, json, time, pickle, math, subprocess, warnings
import numpy as np
import pandas as pd
import pandas_ta_classic as ta  # NOT pandas_ta
import requests
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from datetime import datetime, timezone

warnings.filterwarnings("ignore", category=UserWarning, message=".*enable_nested_tensor.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", message=".*sklearn.*")

# ── Constants ────────────────────────────────────────────────────────────────
HF_REPO           = "sato2ru/omega-forex-v3"
ENTRY_PRICES_FILE = "forex_entry_prices.json"
SIGNALS_LOG_FILE  = "forex_signals_log.csv"
TWELVE_API_KEY    = os.environ["TWELVE_DATA_API_KEY"]
DISCORD_WEBHOOK   = os.environ["DISCORD_WEBHOOK_FOREX"]
HF_TOKEN          = os.environ.get("HF_TOKEN")

# Session boundaries (UTC)
ASIA_OPEN   = 0;  ASIA_END    = 8
LONDON_OPEN = 8;  NY_OPEN     = 13
OVERLAP_END = 17; NY_END      = 22

# Spread and pip sizes for Kelly sizing
SPREAD_PIPS = {
    "EUR/USD": 0.5, "GBP/USD": 0.8, "USD/JPY": 0.7,
    "USD/CHF": 1.0, "AUD/USD": 0.8, "GBP/JPY": 2.5,
}
PIP_SIZE = {
    "EUR/USD": 0.0001, "GBP/USD": 0.0001, "USD/JPY": 0.01,
    "USD/CHF": 0.0001, "AUD/USD": 0.0001, "GBP/JPY": 0.01,
}

# Currencies affected by news events per pair
CURRENCIES_AFFECTED = {
    "EUR/USD": ["EUR", "USD"], "GBP/USD": ["GBP", "USD"],
    "USD/JPY": ["USD", "JPY"], "USD/CHF": ["USD", "CHF"],
    "AUD/USD": ["AUD", "USD"], "GBP/JPY": ["GBP", "JPY"],
}


# ── Model definition (must match training exactly) ────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class RegimeGate(nn.Module):
    """
    Matches v3 notebook exactly — dropout inside regime_proj,
    d_model//2 intermediate, attribute named regime_proj (not gate).
    """
    def __init__(self, d_model=64, n_regime=5, dropout=0.1):
        super().__init__()
        self.regime_proj = nn.Sequential(
            nn.Linear(n_regime, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),            # added in v3
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, regime):
        gate = self.regime_proj(regime)
        return self.norm(x * gate)


class OmegaForex(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=3,
                 dropout=0.1, seq_len=24, regime_idx=None):
        super().__init__()
        self.regime_idx = regime_idx

        self.conv_block = nn.Sequential(
            nn.Conv1d(n_features, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
            nn.Dropout(dropout),
        )
        self.regime_gate = RegimeGate(d_model=d_model, n_regime=5, dropout=dropout)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 8, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        regime = x[:, :, self.regime_idx]
        z = self.conv_block(x.transpose(1, 2)).transpose(1, 2)
        z = self.regime_gate(z, regime)
        z = self.encoder(self.pos_enc(z))
        return self.classifier(z[:, -1, :])


# ── Load model + config from HuggingFace ─────────────────────────────────────

def load_model():
    print("Loading model from HuggingFace...")
    kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

    cfg_path     = hf_hub_download(HF_REPO, "config.json",  **kwargs)
    weights_path = hf_hub_download(HF_REPO, "model.pt",     **kwargs)
    scalers_path = hf_hub_download(HF_REPO, "scalers.pkl",  **kwargs)

    with open(cfg_path) as f:
        cfg = json.load(f)

    # Derive regime_idx from feature_cols + regime_cols (v3 config no longer stores it directly)
    feature_cols = cfg["feature_cols"]
    regime_cols  = cfg.get("regime_cols", ["atr_rank","adx_norm","vol_regime","trend_regime","regime_bars"])
    regime_idx   = [feature_cols.index(c) for c in regime_cols]

    model = OmegaForex(
        n_features = cfg["n_features"],
        d_model    = cfg["d_model"],
        n_heads    = cfg["n_heads"],
        n_layers   = cfg["n_layers"],
        dropout    = cfg["dropout"],
        seq_len    = cfg["seq_len"],
        regime_idx = regime_idx,
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    T = cfg["test_metrics"]["temperature"]
    print(f"  Architecture : {cfg['architecture']}")
    print(f"  Version      : {cfg['version']}")
    print(f"  Features     : {cfg['n_features']}")
    print(f"  Temperature  : {T}")
    print(f"  Anchor pair  : {cfg.get('anchor_pair', 'USD/JPY')}")

    return model, scalers, cfg


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_page(symbol, api_key, end_datetime=None):
    params = {
        "symbol":     symbol,
        "interval":   "1h",
        "outputsize": 500,
        "apikey":     api_key,
        "format":     "JSON",
        "timezone":   "UTC",
    }
    if end_datetime:
        params["end_date"] = end_datetime
    r = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "values" not in data:
        raise RuntimeError(f"Twelve Data error for {symbol}: {data.get('message', data)}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = 0.0
    return df.dropna(subset=["open", "high", "low", "close"])


def strip_weekends_and_gaps(df, max_gap_hours=4):
    df = df[df["datetime"].dt.dayofweek < 5].copy()
    gaps = df["datetime"].diff().dt.total_seconds() / 3600
    df = df[gaps.isna() | (gaps <= max_gap_hours)].copy()
    return df.reset_index(drop=True)


def fetch_forex(pair):
    df = fetch_page(pair, TWELVE_API_KEY)
    df = strip_weekends_and_gaps(df)
    return df


# ── Feature engineering (must match training exactly) ────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Trend
    d["ema9"]  = ta.ema(d["close"], length=9)
    d["ema21"] = ta.ema(d["close"], length=21)
    d["ema50"] = ta.ema(d["close"], length=50)
    d["ema9_ratio"]  = d["close"] / d["ema9"]  - 1
    d["ema21_ratio"] = d["close"] / d["ema21"] - 1
    d["ema50_ratio"] = d["close"] / d["ema50"] - 1

    adx_out = ta.adx(d["high"], d["low"], d["close"], length=14)
    d["adx"] = adx_out["ADX_14"]
    d["dmp"] = adx_out["DMP_14"]
    d["dmn"] = adx_out["DMN_14"]

    # Momentum
    d["rsi"] = ta.rsi(d["close"], length=14)
    macd_out = ta.macd(d["close"], fast=12, slow=26, signal=9)
    d["macd"]        = macd_out["MACD_12_26_9"]
    d["macd_signal"] = macd_out["MACDs_12_26_9"]
    d["macd_hist"]   = macd_out["MACDh_12_26_9"]
    stoch = ta.stoch(d["high"], d["low"], d["close"], k=14, d=3)
    d["stoch_k"] = stoch["STOCHk_14_3_3"]
    d["stoch_d"] = stoch["STOCHd_14_3_3"]
    d["cci"] = ta.cci(d["high"], d["low"], d["close"], length=14)
    d["roc"] = ta.roc(d["close"], length=10)
    d["mom"] = ta.mom(d["close"], length=10)

    # Volatility
    d["atr"] = ta.atr(d["high"], d["low"], d["close"], length=14)
    bb = ta.bbands(d["close"], length=20, std=2)
    bb_cols = bb.columns.tolist()
    d["bb_upper"]  = bb[bb_cols[2]]
    d["bb_middle"] = bb[bb_cols[1]]
    d["bb_lower"]  = bb[bb_cols[0]]
    d["bb_width"]  = (d["bb_upper"] - d["bb_lower"]) / d["bb_middle"]
    d["bb_pct"]    = (d["close"] - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"] + 1e-10)

    # Price structure
    d["body"]     = (d["close"] - d["open"]).abs() / (d["high"] - d["low"] + 1e-10)
    d["wick_up"]  = (d["high"] - d[["open","close"]].max(axis=1)) / (d["high"] - d["low"] + 1e-10)
    d["wick_dn"]  = (d[["open","close"]].min(axis=1) - d["low"])  / (d["high"] - d["low"] + 1e-10)
    d["hl_range"] = (d["high"] - d["low"]) / d["close"]

    # Session features
    hour = d["datetime"].dt.hour
    dow  = d["datetime"].dt.dayofweek
    d["hour_sin"]     = np.sin(2 * np.pi * hour / 24)
    d["hour_cos"]     = np.cos(2 * np.pi * hour / 24)
    d["dow_sin"]      = np.sin(2 * np.pi * dow  / 5)
    d["dow_cos"]      = np.cos(2 * np.pi * dow  / 5)
    d["sess_asia"]    = ((hour >= ASIA_OPEN)   & (hour < ASIA_END)).astype(float)
    d["sess_london"]  = ((hour >= LONDON_OPEN) & (hour < NY_OPEN)).astype(float)
    d["sess_ny"]      = ((hour >= NY_OPEN)     & (hour < NY_END)).astype(float)
    d["sess_overlap"] = ((hour >= NY_OPEN)     & (hour < OVERLAP_END)).astype(float)
    d["sess_dead"]    = ((hour >= OVERLAP_END) & (hour < 20)).astype(float)
    d["monday_open"]  = ((dow == 0) & (hour < 3)).astype(float)
    d["friday_close"] = ((dow == 4) & (hour >= 18)).astype(float)

    # Seasonal lag features (Keplerian trick)
    d["lag24_ret"]  = d["close"].pct_change(24)
    d["lag168_ret"] = d["close"].pct_change(168)
    d["lag24_rsi"]  = d["rsi"] - d["rsi"].shift(24)
    d["lag168_rsi"] = d["rsi"] - d["rsi"].shift(168)
    d["lag24_atr"]  = d["atr"] / (d["atr"].shift(24) + 1e-10) - 1

    # Regime features — matches notebook exactly (regime_id = trend*2 + vol, clipped at 48)
    d["atr_rank"]     = d["atr"].rolling(168, min_periods=24).rank(pct=True)
    d["adx_norm"]     = d["adx"] / 100.0
    d["vol_regime"]   = (d["atr_rank"] >= 0.70).astype(float)
    d["trend_regime"] = (d["adx"] > 25).astype(float)
    regime_id         = d["trend_regime"] * 2 + d["vol_regime"]
    d["regime_bars"]  = (regime_id != regime_id.shift(1)).cumsum()
    d["regime_bars"]  = d.groupby("regime_bars").cumcount().clip(upper=48) / 48.0

    return d


# Base feature cols (43) — anchor cols injected separately in main
BASE_FEATURE_COLS = [
    "ema9_ratio","ema21_ratio","ema50_ratio",
    "adx","dmp","dmn","rsi",
    "macd","macd_signal","macd_hist",
    "stoch_k","stoch_d","cci","roc","mom","atr",
    "bb_width","bb_pct","body","wick_up","wick_dn","hl_range",
    "hour_sin","hour_cos","dow_sin","dow_cos",
    "sess_asia","sess_london","sess_ny","sess_overlap","sess_dead",
    "monday_open","friday_close",
    "lag24_ret","lag168_ret","lag24_rsi","lag168_rsi","lag24_atr",
    "atr_rank","adx_norm","vol_regime","trend_regime","regime_bars",
]
ANCHOR_COLS = ["usdjpy_adx_norm", "usdjpy_vol_regime", "usdjpy_trend_regime"]
FEATURE_COLS = BASE_FEATURE_COLS + ANCHOR_COLS   # 46 total


# ── Standard Candle — build USD/JPY anchor lookup ────────────────────────────

def build_anchor(usdjpy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a datetime-indexed anchor table from USD/JPY's own regime features.
    Returns df with columns: datetime, usdjpy_adx_norm, usdjpy_vol_regime, usdjpy_trend_regime
    """
    fd = build_features(usdjpy_df)
    anchor = fd[["datetime","adx_norm","vol_regime","trend_regime"]].copy()
    anchor = anchor.rename(columns={
        "adx_norm":     "usdjpy_adx_norm",
        "vol_regime":   "usdjpy_vol_regime",
        "trend_regime": "usdjpy_trend_regime",
    })
    return anchor.set_index("datetime")


def inject_anchor(fd: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    """Left-join anchor features onto pair df by datetime, ffill any gaps."""
    fd = fd.join(anchor, on="datetime", how="left")
    for col in ANCHOR_COLS:
        fd[col] = fd[col].ffill().bfill()
    return fd.reset_index(drop=True)


# ── News masking (ForexFactory) ───────────────────────────────────────────────

def fetch_forexfactory_calendar() -> list:
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        parsed = []
        for e in r.json():
            try:
                dt = pd.to_datetime(e.get("date") or e.get("datetime"))
                parsed.append({
                    "datetime": dt,
                    "currency": e.get("country", "").upper(),
                    "impact":   e.get("impact", "").lower(),
                    "title":    e.get("title", ""),
                })
            except Exception:
                pass
        high = sum(1 for e in parsed if e["impact"] == "high")
        print(f"  ForexFactory: {len(parsed)} events fetched, {high} HIGH impact")
        return parsed
    except Exception as ex:
        print(f"  ForexFactory fetch failed: {ex} — signals will NOT be masked")
        return []


def is_news_masked(signal_dt, pair: str, events: list, window_hours: float = 1.0) -> tuple:
    currencies = CURRENCIES_AFFECTED.get(pair, [])
    window = pd.Timedelta(hours=window_hours)
    signal_ts = pd.Timestamp(signal_dt)
    for event in events:
        if event["impact"] != "high":
            continue
        if event["currency"] not in currencies:
            continue
        try:
            if abs(event["datetime"] - signal_ts) <= window:
                return True, f"{event['title']} ({event['currency']}, {event['datetime']})"
        except Exception:
            continue
    return False, ""


# ── Kelly sizing pipeline ─────────────────────────────────────────────────────

def swing_high_low(closes: np.ndarray, window: int = 5):
    n = len(closes)
    last_high_idx, last_low_idx = 0, 0
    for i in range(window, n - window):
        if closes[i] == max(closes[i-window:i+window+1]):
            last_high_idx = i
        if closes[i] == min(closes[i-window:i+window+1]):
            last_low_idx = i
    return last_high_idx, last_low_idx


def redshift_kelly(kelly_base: float, closes: np.ndarray,
                   atr: float, direction: str,
                   swing_window: int = 5, max_atr_mult: float = 3.0) -> float:
    if atr <= 0 or len(closes) < swing_window * 2 + 1:
        return kelly_base
    high_idx, low_idx = swing_high_low(closes, swing_window)
    current = closes[-1]
    if direction == "BUY":
        extension = max(0, current - closes[low_idx])
    else:
        extension = max(0, closes[high_idx] - current)
    decay = extension / (max_atr_mult * atr + 1e-10)
    return kelly_base * max(0.0, 1.0 - decay)


def atr_to_pips(atr_price: float, pair: str) -> float:
    return atr_price / PIP_SIZE.get(pair, 0.0001)


def spread_aware_kelly(kelly_base: float, pair: str, expected_move_pips: float) -> float:
    spread = SPREAD_PIPS.get(pair, 1.0)
    if expected_move_pips <= 0:
        return 0.0
    return kelly_base * max(0.0, (expected_move_pips - spread) / expected_move_pips)


def drawdown_aware_kelly(kelly_base: float, recent_outcomes: list,
                         loss_streak_trigger: int = 3) -> float:
    if not recent_outcomes:
        return kelly_base
    streak = 0
    for outcome in reversed(recent_outcomes):
        if outcome == 0:
            streak += 1
        else:
            break
    return kelly_base * 0.5 if streak >= loss_streak_trigger else kelly_base


def get_recent_outcomes(pair: str, n: int = 10) -> list:
    """Read last n outcomes for a pair from the signals log. Returns list of 1/0."""
    if not os.path.exists(SIGNALS_LOG_FILE):
        return []
    try:
        df = pd.read_csv(SIGNALS_LOG_FILE)
        df = df[df["pair"] == pair].tail(n)
        return [1 if o == "WIN" else 0 for o in df["outcome"].tolist()]
    except Exception:
        return []


def compute_kelly(confidence: float, pair: str, closes: np.ndarray,
                  atr_raw: float, direction: str) -> float:
    """Full Kelly pipeline: base → redshift → spread → drawdown, capped at 20%."""
    k_base = min((confidence - 0.5) * 0.20 / 0.10, 0.20)   # 0% at 0.50, 10% at 0.60, cap 20%
    k_base = max(k_base, 0.01)

    k_rs = redshift_kelly(k_base, closes, atr_raw, direction)
    k_sp = spread_aware_kelly(k_rs, pair, atr_to_pips(atr_raw, pair))
    k_dd = drawdown_aware_kelly(k_sp, get_recent_outcomes(pair))
    return round(k_dd, 4)


# ── Inference ─────────────────────────────────────────────────────────────────

def get_signal(pair, model, scaler, cfg, anchor):
    """
    Returns dict with signal details, or None if no signal fires.
    anchor: datetime-indexed DataFrame with usdjpy_* columns
    """
    seq_len      = cfg["seq_len"]
    T            = cfg["test_metrics"]["temperature"]
    pair_thrs    = cfg["test_metrics"].get("pair_thresholds", {})
    thr          = pair_thrs.get(pair, cfg["conf_threshold"])

    # Benched pairs (thr=0.99) — skip early to save API credits
    if thr >= 0.99:
        print(f"  {pair}: benched (thr={thr}), skipping")
        return None

    # 1. Fetch, feature-engineer, inject anchor
    df = fetch_forex(pair)
    fd = build_features(df)
    fd = inject_anchor(fd, anchor)
    fd = fd.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    if len(fd) < seq_len + 5:
        print(f"  {pair}: not enough bars after warmup ({len(fd)})")
        return None

    # 2. Scale
    X_scaled = scaler.transform(fd[FEATURE_COLS].values.astype(np.float32))

    # 3. Build window
    window = X_scaled[-seq_len:]                    # (24, 46)
    x = torch.tensor(window).unsqueeze(0)           # (1, 24, 46)

    # 4. Inference with temperature
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits / T, dim=1)[0]

    prob_up    = probs[1].item()
    prob_down  = probs[0].item()
    confidence = max(prob_up, prob_down)

    # 5. Threshold check
    if confidence < thr:
        return None

    direction   = "BUY" if prob_up > prob_down else "SELL"
    entry_price = float(fd["close"].iloc[-1])
    ts          = fd["datetime"].iloc[-1]
    atr_raw     = float(fd["atr"].iloc[-1])
    closes      = fd["close"].values[-40:]

    # 6. Kelly sizing
    size = compute_kelly(confidence, pair, closes, atr_raw, direction)

    # 7. Regime context
    vol_r   = fd["vol_regime"].iloc[-1]
    trend_r = fd["trend_regime"].iloc[-1]
    if vol_r == 1 and trend_r == 1:   regime_str = "HV-TREND"
    elif vol_r == 1:                   regime_str = "HV-RANGE"
    elif trend_r == 1:                 regime_str = "LV-TREND"
    else:                              regime_str = "LV-RANGE"

    return {
        "pair":        pair,
        "signal":      direction,
        "prob_up":     round(prob_up, 4),
        "prob_down":   round(prob_down, 4),
        "confidence":  round(confidence, 4),
        "entry_price": entry_price,
        "datetime":    ts.strftime("%Y-%m-%d %H:%M"),
        "threshold":   thr,
        "regime":      regime_str,
        "size":        size,
        "atr_raw":     atr_raw,
    }


# ── Entry / outcome tracking ──────────────────────────────────────────────────

def load_entry_prices():
    if not os.path.exists(ENTRY_PRICES_FILE):
        return {}
    with open(ENTRY_PRICES_FILE) as f:
        data = json.load(f)
    if isinstance(data, list):
        print(f"  Migrating {ENTRY_PRICES_FILE} from list (v1) to dict...")
        migrated = {}
        for i, entry in enumerate(data):
            if isinstance(entry, dict) and "pair" in entry:
                ts  = entry.get("datetime", f"unknown_{i}").replace(" ","").replace("-","").replace(":","")
                key = f"{entry['pair'].replace('/','')}_{ ts}"
                migrated[key] = entry
        save_entry_prices(migrated)
        return migrated
    return data


def save_entry_prices(entries):
    with open(ENTRY_PRICES_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def resolve_outcomes(current_prices):
    entries  = load_entry_prices()
    resolved = []
    still_open = {}

    for key, entry in entries.items():
        pair = entry["pair"]
        if pair not in current_prices:
            still_open[key] = entry
            continue

        current  = current_prices[pair]
        entry_px = entry["entry_price"]
        signal   = entry["signal"]

        if signal == "BUY":
            pnl_pct = (current - entry_px) / entry_px
            outcome = "WIN" if current > entry_px else "LOSS"
        else:
            pnl_pct = (entry_px - current) / entry_px
            outcome = "WIN" if current < entry_px else "LOSS"

        size = entry.get("size", 0.05)
        resolved.append({
            "datetime":    entry["datetime"],
            "pair":        pair,
            "signal":      signal,
            "entry_price": entry_px,
            "exit_price":  current,
            "pnl_pct":     round(pnl_pct * 100, 4),
            "pnl_sized":   round(pnl_pct * size * 100, 4),
            "size":        size,
            "outcome":     outcome,
            "confidence":  entry.get("confidence", ""),
            "threshold":   entry.get("threshold", ""),
            "regime":      entry.get("regime", ""),
        })

    save_entry_prices(still_open)
    return resolved


def append_to_csv(rows):
    if not rows:
        return
    df_new = pd.DataFrame(rows)
    if os.path.exists(SIGNALS_LOG_FILE):
        df_existing = pd.read_csv(SIGNALS_LOG_FILE)
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new
    df_out.to_csv(SIGNALS_LOG_FILE, index=False)


# ── Discord posting ───────────────────────────────────────────────────────────

def post_discord(signals, resolved, masked_log):
    now   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"**OMEGA Forex v3** — {now}\n"]

    if signals:
        lines.append("**📡 New Signals**")
        lines.append("```")
        lines.append(f"{'Pair':<10} {'Signal':<6} {'Conf':>6} {'Size':>6} {'Regime':<10} {'Price':>10}")
        lines.append("-" * 54)
        for s in signals:
            lines.append(
                f"{s['pair']:<10} {s['signal']:<6} {s['confidence']:>6.1%} "
                f"{s['size']:>5.1%}  {s['regime']:<10} {s['entry_price']:>10.5f}"
            )
        lines.append("```")
    else:
        lines.append("*No signals this hour (below threshold)*")

    if masked_log:
        lines.append("**🚫 News-masked this hour**")
        lines.append("```")
        for pair, reason in masked_log:
            lines.append(f"  {pair}: {reason[:60]}")
        lines.append("```")

    if resolved:
        wins  = sum(1 for r in resolved if r["outcome"] == "WIN")
        total = len(resolved)
        lines.append(f"\n**✅ Resolved ({wins}/{total} WIN)**")
        lines.append("```")
        lines.append(f"{'Pair':<10} {'Sig':<5} {'Entry':>10} {'Exit':>10} {'PnL%':>7} {'Result'}")
        lines.append("-" * 54)
        for r in resolved:
            sign = "+" if r["pnl_pct"] > 0 else ""
            lines.append(
                f"{r['pair']:<10} {r['signal']:<5} {r['entry_price']:>10.5f} "
                f"{r['exit_price']:>10.5f} {sign}{r['pnl_pct']:>6.3f}% {r['outcome']}"
            )
        lines.append("```")

    payload = {"content": "\n".join(lines)}
    r = requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
    if r.status_code not in (200, 204):
        print(f"Discord post failed: {r.status_code} {r.text}")
    else:
        print("Discord post sent.")


# ── Git push ──────────────────────────────────────────────────────────────────

def git_push(message):
    cmds = [
        ["git", "config", "user.email", "omega-bot@github-actions"],
        ["git", "config", "user.name",  "OMEGA Bot"],
        ["git", "add", "-A"],
    ]
    for cmd in cmds:
        subprocess.run(cmd, check=False)

    diff = subprocess.run(["git", "diff", "--staged", "--quiet"])
    if diff.returncode == 0:
        print("No changes to commit.")
        return

    subprocess.run(["git", "commit", "-m", message], check=False)
    subprocess.run(["git", "pull", "--rebase", "origin", "main"], check=False)
    subprocess.run(["git", "push"], check=False)
    print("Git push done.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"\n=== OMEGA Forex v3 — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ===\n")

    # 1. Load model
    model, scalers, cfg = load_model()
    all_pairs  = cfg["pairs"]
    anchor_pair = cfg.get("anchor_pair", "USD/JPY")
    pair_thrs   = cfg["test_metrics"].get("pair_thresholds", {})
    active_pairs = [p for p in all_pairs if pair_thrs.get(p, cfg["conf_threshold"]) < 0.99]
    benched      = [p for p in all_pairs if pair_thrs.get(p, cfg["conf_threshold"]) >= 0.99]
    print(f"\nActive pairs : {active_pairs}")
    print(f"Benched      : {benched}\n")

    # 2. Fetch news calendar once
    print("Fetching ForexFactory calendar...")
    news_events = fetch_forexfactory_calendar()

    # 3. Fetch USD/JPY first for Standard Candle anchor
    print(f"\nFetching anchor pair {anchor_pair}...")
    try:
        anchor_raw = fetch_forex(anchor_pair)
        anchor     = build_anchor(anchor_raw)
        print(f"  {anchor_pair} anchor ready ({len(anchor)} bars)")
    except Exception as e:
        print(f"  ERROR fetching anchor {anchor_pair}: {e}")
        print("  Falling back to zero anchor (model will still run, edge reduced)")
        # Zero anchor — model sees all-zeros for anchor features (neutral signal)
        anchor = pd.DataFrame(
            {"usdjpy_adx_norm": [0.0], "usdjpy_vol_regime": [0.0], "usdjpy_trend_regime": [0.0]},
            index=[pd.Timestamp.now()]
        )

    # 4. Run inference for each active pair
    signals        = []
    current_prices = {}
    masked_log     = []

    for i, pair in enumerate(all_pairs):
        if pair == anchor_pair:
            # Anchor pair still needs a current price for outcome resolution
            current_prices[anchor_pair] = float(anchor_raw["close"].iloc[-1])

        thr = pair_thrs.get(pair, cfg["conf_threshold"])
        if thr >= 0.99:
            continue   # benched — already noted above

        print(f"\n[{pair}] Fetching and running inference...")
        try:
            # Rate limit — skip sleep for anchor pair (already fetched above)
            if pair != anchor_pair:
                time.sleep(8)

            result = get_signal(pair, model, scalers.get(pair), cfg, anchor)

            if result:
                current_prices[pair] = result["entry_price"]

                # News masking check
                signal_dt = pd.Timestamp(result["datetime"])
                masked, reason = is_news_masked(signal_dt, pair, news_events)
                if masked:
                    print(f"  → MASKED by news: {reason[:80]}")
                    masked_log.append((pair, reason))
                    # Still record current price for outcome resolution
                else:
                    signals.append(result)
                    print(f"  → {result['signal']}  conf={result['confidence']:.3f}  "
                          f"thr={thr}  size={result['size']:.1%}  regime={result['regime']}")
            else:
                # No signal — fetch current price separately for outcome resolution
                df = fetch_forex(pair)
                current_prices[pair] = float(df["close"].iloc[-1])
                time.sleep(8)   # extra sleep since we fetched twice
                print(f"  → HOLD (below threshold {thr})")

        except Exception as e:
            print(f"  ERROR: {e}")

    # 5. Resolve outcomes from ~55 min ago
    print("\nResolving prior entries...")
    resolved = resolve_outcomes(current_prices)
    for r in resolved:
        sign = "+" if r["pnl_pct"] > 0 else ""
        print(f"  {r['pair']} {r['signal']} → {r['outcome']}  {sign}{r['pnl_pct']:.3f}%")

    # 6. Append resolved to CSV
    append_to_csv(resolved)

    # 7. Log new signal entries
    if signals:
        entries = load_entry_prices()
        ts_key  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        for s in signals:
            key = f"{s['pair'].replace('/','') }_{ts_key}"
            entries[key] = s
        save_entry_prices(entries)

    # 8. Post to Discord
    post_discord(signals, resolved, masked_log)

    # 9. Commit and push
    n_sig = len(signals)
    n_res = len(resolved)
    wins  = sum(1 for r in resolved if r["outcome"] == "WIN")
    git_push(f"OMEGA Forex v3: {n_sig} signal(s), resolved {wins}/{n_res}")

    print("\nDone.")


if __name__ == "__main__":
    main()