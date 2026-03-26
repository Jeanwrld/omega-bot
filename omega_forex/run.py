"""
OMEGA Forex v2 — Hourly inference script
GitHub Actions: omega-forex-signals.yml (:05 every hour)

Signal flow:
  1. Fetch 500 bars per active pair (Twelve Data)
  2. Strip weekends and gap artifacts
  3. Build 43 features (technical + session + lag + regime)
  4. Scale with train-fitted scaler
  5. Build seq_len=24 window
  6. Run model (CNN → RegimeGate → Transformer) with temperature T
  7. Apply per-pair confidence threshold from config.json
  8. Half-Kelly sizing, capped at 20%
  9. Resolve outcomes from 55 min ago → append to forex_signals_log.csv
 10. Post signal table + resolved outcomes to Discord
 11. git commit + pull rebase + push
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

# Suppress harmless warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*enable_nested_tensor.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*InconsistentVersionWarning.*")
warnings.filterwarnings("ignore", message=".*sklearn.*")

# ── Constants ────────────────────────────────────────────────────────────────
HF_REPO           = "sato2ru/omega-forex"
ENTRY_PRICES_FILE = "forex_entry_prices.json"
SIGNALS_LOG_FILE  = "forex_signals_log.csv"
TWELVE_API_KEY    = os.environ["TWELVE_DATA_API_KEY"]
DISCORD_WEBHOOK   = os.environ["DISCORD_WEBHOOK_FOREX"]
HF_TOKEN          = os.environ.get("HF_TOKEN")

# Session boundaries (UTC)
ASIA_OPEN   = 0;  ASIA_END    = 8
LONDON_OPEN = 8;  NY_OPEN     = 13
OVERLAP_END = 17; NY_END      = 22


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
    Soft-modulates CNN output by the 5 regime features.
    Gate in [0,1] via Sigmoid — learns which d_model dimensions
    matter per regime without any hand-coded rules.
    """
    def __init__(self, n_regime=5, d_model=64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_regime, 32),
            nn.GELU(),
            nn.Linear(32, d_model),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, regime):
        return self.norm(x * self.gate(regime))


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
        self.regime_gate = RegimeGate(n_regime=5, d_model=d_model)
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
    kwargs = {}
    if HF_TOKEN:
        kwargs["token"] = HF_TOKEN

    cfg_path     = hf_hub_download(HF_REPO, "config.json",  **kwargs)
    weights_path = hf_hub_download(HF_REPO, "model.pt",     **kwargs)
    scalers_path = hf_hub_download(HF_REPO, "scalers.pkl",  **kwargs)

    with open(cfg_path) as f:
        cfg = json.load(f)

    model = OmegaForex(
        n_features = cfg["n_features"],
        d_model    = cfg["d_model"],
        n_heads    = cfg["n_heads"],
        n_layers   = cfg["n_layers"],
        dropout    = cfg["dropout"],
        seq_len    = cfg["seq_len"],
        regime_idx = cfg["regime_idx"],
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    print(f"  Architecture : {cfg['architecture']}")
    print(f"  Version      : {cfg['version']}")
    print(f"  Active pairs : {cfg['active_pairs']}")
    print(f"  Temperature  : {cfg['temperature']}")

    return model, scalers, cfg


# ── Data fetching ─────────────────────────────────────────────────────────────

def fetch_page(symbol, api_key, end_datetime=None):
    params = {
        "symbol":     symbol,
        "interval":   "1h",
        "outputsize": 500,         # fetch 500 — weekend strip reduces count
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
    df["volume"] = 0.0   # forex has no volume
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
    bb_cols = bb.columns.tolist()   # always [BBL, BBM, BBU, BBB, BBP]
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

    # Regime features (H-R Diagram)
    d["atr_rank"]    = d["atr"].rolling(168).rank(pct=True)
    d["adx_norm"]    = d["adx"] / 100.0
    d["vol_regime"]  = (d["atr_rank"] >= 0.70).astype(float)
    d["trend_regime"] = (d["adx"] > 25).astype(float)
    run_id = (d["vol_regime"] != d["vol_regime"].shift()).cumsum()
    d["regime_bars"] = run_id.groupby(run_id).cumcount() / 48.0

    return d


FEATURE_COLS = [
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


# ── Inference ─────────────────────────────────────────────────────────────────

def get_signal(pair, model, scaler, cfg):
    """
    Returns dict with keys: pair, signal, prob_up, prob_down, confidence,
    entry_price, datetime — or None if no signal fires.
    """
    seq_len = cfg["seq_len"]
    T       = cfg["temperature"]
    thr     = cfg["test_metrics"]["pair_thresholds"].get(pair, cfg["conf_threshold"])

    # 1. Fetch and feature-engineer
    df = fetch_forex(pair)
    fd = build_features(df)
    fd = fd.dropna(subset=FEATURE_COLS).reset_index(drop=True)

    if len(fd) < seq_len + 5:
        print(f"  {pair}: not enough bars after feature warmup ({len(fd)})")
        return None

    # 2. Scale
    X_raw = fd[FEATURE_COLS].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw)

    # 3. Build window (last seq_len bars)
    window = X_scaled[-seq_len:]                    # (24, 43)
    x = torch.tensor(window).unsqueeze(0)           # (1, 24, 43)

    # 4. Inference with temperature scaling
    with torch.no_grad():
        logits = model(x)                           # (1, 2)
        probs  = torch.softmax(logits / T, dim=1)[0]

    prob_up   = probs[1].item()
    prob_down = probs[0].item()
    confidence = max(prob_up, prob_down)

    # 5. Threshold check
    if confidence < thr:
        return None

    signal      = "BUY" if prob_up > prob_down else "SELL"
    entry_price = float(fd["close"].iloc[-1])
    ts          = fd["datetime"].iloc[-1]

    # Regime context for logging
    regime_str = ""
    vol_r   = fd["vol_regime"].iloc[-1]
    trend_r = fd["trend_regime"].iloc[-1]
    if vol_r == 1 and trend_r == 1:
        regime_str = "HV-TREND"
    elif vol_r == 1:
        regime_str = "HV-RANGE"
    elif trend_r == 1:
        regime_str = "LV-TREND"
    else:
        regime_str = "LV-RANGE"

    return {
        "pair":        pair,
        "signal":      signal,
        "prob_up":     round(prob_up, 4),
        "prob_down":   round(prob_down, 4),
        "confidence":  round(confidence, 4),
        "entry_price": entry_price,
        "datetime":    ts.strftime("%Y-%m-%d %H:%M"),
        "threshold":   thr,
        "regime":      regime_str,
    }


# ── Kelly sizing ──────────────────────────────────────────────────────────────

def kelly_size(confidence, max_size=0.20):
    """
    Half-Kelly sizing based on model confidence.
    edge = confidence - (1 - confidence) = 2*confidence - 1
    Full Kelly = edge / odds (1:1 binary) = edge
    Half Kelly = edge / 2, capped at max_size
    """
    edge       = 2 * confidence - 1          # e.g. 0.60 → edge = 0.20
    half_kelly = edge / 2
    return round(min(max(half_kelly, 0.01), max_size), 4)


# ── Entry / outcome tracking ──────────────────────────────────────────────────

def load_entry_prices():
    if not os.path.exists(ENTRY_PRICES_FILE):
        return {}
    with open(ENTRY_PRICES_FILE) as f:
        data = json.load(f)
    # v1 run.py saved this as a list — migrate to dict on first read
    if isinstance(data, list):
        print(f"  Migrating {ENTRY_PRICES_FILE} from list (v1) to dict (v2)...")
        migrated = {}
        for i, entry in enumerate(data):
            if isinstance(entry, dict) and "pair" in entry:
                ts = entry.get("datetime", f"unknown_{i}").replace(" ", "").replace("-", "").replace(":", "")
                key = f"{entry['pair'].replace('/', '')}_{ts}"
                migrated[key] = entry
        save_entry_prices(migrated)
        return migrated
    return data


def save_entry_prices(entries):
    with open(ENTRY_PRICES_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def resolve_outcomes(signals, current_prices):
    """
    Check entries logged ~55 min ago against current prices.
    Returns list of resolved trade dicts ready for CSV append.
    """
    entries = load_entry_prices()
    resolved = []
    still_open = {}

    for key, entry in entries.items():
        pair = entry["pair"]
        if pair not in current_prices:
            still_open[key] = entry
            continue

        current = current_prices[pair]
        entry_px = entry["entry_price"]
        signal   = entry["signal"]

        if signal == "BUY":
            pnl_pct = (current - entry_px) / entry_px
            outcome = "WIN" if current > entry_px else "LOSS"
        else:
            pnl_pct = (entry_px - current) / entry_px
            outcome = "WIN" if current < entry_px else "LOSS"

        size = entry.get("size", 0.10)
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

def post_discord(signals, resolved):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"**OMEGA Forex v2** — {now}\n"]

    if signals:
        lines.append("**📡 New Signals**")
        lines.append("```")
        lines.append(f"{'Pair':<10} {'Signal':<6} {'Conf':>6} {'Size':>6} {'Regime':<10} {'Price':>10}")
        lines.append("-" * 54)
        for s in signals:
            size = kelly_size(s["confidence"])
            lines.append(
                f"{s['pair']:<10} {s['signal']:<6} {s['confidence']:>6.1%} "
                f"{size:>5.1%}  {s['regime']:<10} {s['entry_price']:>10.5f}"
            )
        lines.append("```")
    else:
        lines.append("*No signals this hour (below threshold)*")

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
    print(f"\n=== OMEGA Forex v2 — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ===\n")

    # Load model
    model, scalers, cfg = load_model()

    active_pairs = cfg["active_pairs"]
    print(f"\nActive pairs: {active_pairs}")
    print(f"Benched: {list(cfg.get('benched_pairs', {}).keys())}\n")

    # Collect current prices and run inference
    signals       = []
    current_prices = {}

    for i, pair in enumerate(active_pairs):
        print(f"[{pair}] Fetching...")
        try:
            result = get_signal(pair, model, scalers.get(pair), cfg)
            if result:
                size = kelly_size(result["confidence"])
                result["size"] = size
                signals.append(result)
                print(f"  → {result['signal']}  conf={result['confidence']:.3f}  "
                      f"thr={result['threshold']}  regime={result['regime']}  size={size:.1%}")
            else:
                # Still need current price for outcome resolution
                df = fetch_forex(pair)
                current_prices[pair] = float(df["close"].iloc[-1])
                print(f"  → No signal (below threshold {cfg['test_metrics']['pair_thresholds'].get(pair, cfg['conf_threshold'])})")

            if result:
                current_prices[pair] = result["entry_price"]

        except Exception as e:
            print(f"  ERROR: {e}")

        if i < len(active_pairs) - 1:
            time.sleep(8)   # Twelve Data free tier: 8 req/min

    # Resolve outcomes from ~55 min ago
    print("\nResolving prior entries...")
    resolved = resolve_outcomes(signals, current_prices)
    for r in resolved:
        sign = "+" if r["pnl_pct"] > 0 else ""
        print(f"  {r['pair']} {r['signal']} → {r['outcome']}  {sign}{r['pnl_pct']:.3f}%")

    # Append resolved to CSV
    append_to_csv(resolved)

    # Log new entries to JSON
    if signals:
        entries = load_entry_prices()
        ts_key  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        for s in signals:
            key = f"{s['pair'].replace('/', '')}_{ts_key}"
            entries[key] = s
        save_entry_prices(entries)

    # Post to Discord
    post_discord(signals, resolved)

    # Commit and push
    n_sig = len(signals)
    n_res = len(resolved)
    wins  = sum(1 for r in resolved if r["outcome"] == "WIN")
    git_push(f"OMEGA Forex v2: {n_sig} signal(s), resolved {wins}/{n_res}")

    print("\nDone.")


if __name__ == "__main__":
    main()