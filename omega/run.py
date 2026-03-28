#!/usr/bin/env python3
"""
OMEGA v7 — run.py
Crypto-only (BTC/ETH/SOL/XRP/ADA) · 1h bars · 38 features · RegimeGate
3h forward signal · CryptoCompare data · HuggingFace weights

Key differences vs v3:
  • OmegaV7: CNN → RegimeGate → Transformer (no market embedding)
  • 38 features: 33 ratio-based base + 5 regime (no raw price cols)
  • Per-symbol StandardScaler (not per-column)
  • 3h outcome window (not 55 min)
  • HF artifacts: omega_v7_weights.pt / omega_v7_scalers.pkl
  • No stocks — Polygon removed
"""

import os
import json
import math
import pickle
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
HF_REPO         = "sato2ru/omega-v7"
DISCORD_WEBHOOK = os.environ["DISCORD_WEBHOOK_FOREX"]
LOG_FILE        = "omega_v7_signals_log.csv"
ENTRY_LOG_FILE  = "omega_v7_entry_prices.json"

CFG = {
    "symbols":        ["BTC", "ETH", "SOL", "XRP", "ADA"],
    "seq_len":        24,
    "d_model":        64,
    "ff_dim":         128,   # d_model*2 — keeps params near v3's 157K
    "n_heads":        4,
    "n_layers":       3,
    "dropout":        0.25,
    "n_features":     38,
    "n_regime":       5,
    "conf_threshold": 0.60,
    "max_kelly":      0.20,
    "total_capital":  50.0,
    "crypto_alloc":   0.50,
    "target_horizon": 3,     # hours forward (for outcome resolution)
}

TARGET_HORIZON = CFG["target_horizon"]   # 3
TARGET_THRESH  = 0.001                   # 0.1% minimum move

DEVICE = torch.device("cpu")

# ── Feature columns (must match training order exactly) ──────────────────────
BASE_FEATURE_COLS = [
    # price ratios
    "hl_ratio", "co_ratio",
    # ema ratios
    "ema9_ratio", "ema21_ratio", "ema50_ratio",
    # momentum
    "mom1", "mom3", "mom6", "mom12", "mom24",
    # volatility
    "atr14", "atr5", "vol_6", "vol_24", "vol_ratio",
    # rsi
    "rsi14", "rsi7",
    # macd
    "macd_ratio", "macd_signal", "macd_hist",
    # bollinger bands
    "bb_pct", "bb_width",
    # stochastic
    "stoch_k", "stoch_d",
    # v3 core
    "cci14", "roc12", "mom14",
    "log_vol", "vol_ratio_14", "obv_norm",
    "body", "upper_wick", "lower_wick",
]

REGIME_COLS = [
    "atr_rank", "adx_norm", "vol_regime", "trend_regime", "regime_bars",
]

FEATURE_COLS = BASE_FEATURE_COLS + REGIME_COLS  # 38 total
N_FEATURES   = len(FEATURE_COLS)               # 38
N_REGIME     = len(REGIME_COLS)                #  5

assert N_FEATURES == 38, f"Expected 38 features, got {N_FEATURES}"


# ── Model ────────────────────────────────────────────────────────────────────
class RegimeGate(nn.Module):
    """Multiplicative gate conditioned on regime features from the last bar."""

    def __init__(self, d_model: int, n_regime: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_regime, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, regime: torch.Tensor) -> torch.Tensor:
        # x      : (B, T, D)
        # regime : (B, N_REGIME)
        return x * self.gate(regime).unsqueeze(1)


class OmegaV7(nn.Module):
    """
    CNN → RegimeGate → Transformer → Head

    v3 backbone + v6 RegimeGate.  No market embedding (crypto only).
    Input : x  (B, SEQ_LEN, N_FEATURES)
    Output: logits  (B, 2)
    """

    def __init__(
        self,
        n_features: int   = N_FEATURES,
        n_regime:   int   = N_REGIME,
        seq_len:    int   = 24,
        d_model:    int   = 64,
        ff_dim:     int   = 128,
        n_heads:    int   = 4,
        n_layers:   int   = 3,
        dropout:    float = 0.25,
    ):
        super().__init__()
        self.n_regime = n_regime

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, d_model * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model * 2, d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # Regime gate
        self.regime_gate = RegimeGate(d_model, n_regime)

        # Learned positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = ff_dim,
            dropout         = dropout,
            batch_first     = True,
            norm_first      = True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Slice regime features from the last bar
        regime = x[:, -1, -self.n_regime:]                    # (B, N_REGIME)

        # CNN  (B, N_FEATURES, T) → (B, T, D)
        h = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)

        # Regime gate → positional encoding → transformer
        h = self.regime_gate(h, regime)
        h = h + self.pos_emb[:, : h.size(1), :]
        h = self.transformer(h)

        # Mean-pool across time → classify
        return self.head(h.mean(dim=1))


# ── Load model from HuggingFace ──────────────────────────────────────────────
def load_model():
    from huggingface_hub import hf_hub_download

    hf_token = os.environ.get("HF_TOKEN")
    weights_path = hf_hub_download(
        repo_id=HF_REPO, filename="omega_v7_weights.pt", token=hf_token
    )
    scalers_path = hf_hub_download(
        repo_id=HF_REPO, filename="omega_v7_scalers.pkl", token=hf_token
    )

    ckpt = torch.load(weights_path, map_location=DEVICE)
    cfg  = ckpt["config"]

    model = OmegaV7(
        n_features = cfg.get("n_features", N_FEATURES),
        n_regime   = cfg.get("n_regime",   N_REGIME),
        seq_len    = cfg.get("seq_len",     CFG["seq_len"]),
        d_model    = cfg.get("d_model",     CFG["d_model"]),
        ff_dim     = cfg.get("ff_dim",      CFG["ff_dim"]),
        n_heads    = cfg.get("n_heads",     CFG["n_heads"]),
        n_layers   = cfg.get("n_layers",    CFG["n_layers"]),
        dropout    = cfg.get("dropout",     CFG["dropout"]),
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    # Prefer config-embedded column lists; fall back to module-level constants
    feature_cols = cfg.get("feature_cols", FEATURE_COLS)
    regime_cols  = cfg.get("regime_cols",  REGIME_COLS)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"✅ OmegaV7 loaded | "
        f"features={cfg.get('n_features', N_FEATURES)} "
        f"seq_len={cfg.get('seq_len', CFG['seq_len'])} "
        f"params={n_params:,}"
    )
    print(f"   Scaler keys : {list(scalers.keys())}")
    return model, scalers, feature_cols, regime_cols


# ── Data fetch ───────────────────────────────────────────────────────────────
def fetch_crypto(symbol: str, limit: int = 150) -> pd.DataFrame:
    """
    Fetch recent 1h bars from CryptoCompare for a single coin symbol
    (e.g. 'BTC').  Uses USDT pair to match training data.
    """
    r = requests.get(
        "https://min-api.cryptocompare.com/data/v2/histohour",
        params={"fsym": symbol, "tsym": "USDT", "limit": limit},
        timeout=15,
    ).json()

    data = [
        c
        for c in r.get("Data", {}).get("Data", [])
        if c.get("volumefrom", 0) > 0
    ]
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["datetime"] = (
        pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
    )
    df = df.rename(columns={"volumefrom": "volume"})
    df = df[["datetime", "open", "high", "low", "close", "volume"]].astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float}
    )
    return df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)


# ── Feature engineering ──────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mirror of the training-time pipeline (notebook Section 3).
    All features are ratio-based — no raw price columns are passed to the model.

    Note: atr_rank uses rank(pct=True) over the available window, which is a
    reasonable approximation of the training-time global rank.
    """
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # ── Price ratios ─────────────────────────────────────────────────────────
    df["hl_ratio"] = (h - l) / c.shift(1).replace(0, np.nan)
    df["co_ratio"] = (c - df["open"]) / c.shift(1).replace(0, np.nan)

    # ── EMAs ─────────────────────────────────────────────────────────────────
    for span in [9, 21, 50]:
        df[f"ema{span}_ratio"] = c / c.ewm(span=span).mean() - 1

    # ── Momentum ─────────────────────────────────────────────────────────────
    for lag in [1, 3, 6, 12, 24]:
        df[f"mom{lag}"] = c.pct_change(lag)

    # ── Volatility ────────────────────────────────────────────────────────────
    df["atr14"]     = ta.atr(h, l, c, length=14) / c
    df["atr5"]      = ta.atr(h, l, c, length=5)  / c
    df["vol_6"]     = c.pct_change().rolling(6).std()
    df["vol_24"]    = c.pct_change().rolling(24).std()
    df["vol_ratio"] = df["vol_6"] / (df["vol_24"] + 1e-8)

    # ── RSI ───────────────────────────────────────────────────────────────────
    df["rsi14"] = ta.rsi(c, length=14) / 100
    df["rsi7"]  = ta.rsi(c, length=7)  / 100

    # ── MACD ──────────────────────────────────────────────────────────────────
    macd = ta.macd(c, fast=12, slow=26, signal=9)
    df["macd_ratio"]  = macd["MACD_12_26_9"]  / (c + 1e-8)
    df["macd_signal"] = macd["MACDs_12_26_9"] / (c + 1e-8)
    df["macd_hist"]   = macd["MACDh_12_26_9"] / (c + 1e-8)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb = ta.bbands(c, length=20)
    df["bb_pct"]   = (c - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0] + 1e-8)
    df["bb_width"] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / (bb.iloc[:, 1] + 1e-8)

    # ── Stochastic ────────────────────────────────────────────────────────────
    stoch = ta.stoch(h, l, c)
    df["stoch_k"] = stoch.iloc[:, 0] / 100
    df["stoch_d"] = stoch.iloc[:, 1] / 100

    # ── CCI / ROC / MOM ───────────────────────────────────────────────────────
    df["cci14"] = ta.cci(h, l, c, length=14) / 200
    df["roc12"] = ta.roc(c, length=12)        / 100
    df["mom14"] = ta.mom(c, length=14)        / (c + 1e-8)

    # ── Volume ────────────────────────────────────────────────────────────────
    df["log_vol"]      = np.log1p(v)
    df["vol_ratio_14"] = v / (v.rolling(14).mean() + 1e-8)
    df["obv_norm"]     = ta.obv(c, v).pct_change(12)

    # ── Candle body / wick ────────────────────────────────────────────────────
    df["body"]       = (c - df["open"]).abs() / (c + 1e-8)
    df["upper_wick"] = (h - c.combine(df["open"], max)) / (c + 1e-8)
    df["lower_wick"] = (c.combine(df["open"], min) - l) / (c + 1e-8)

    # ── Regime features ───────────────────────────────────────────────────────
    df["atr_rank"]     = df["atr14"].rank(pct=True)
    df["adx_norm"]     = ta.adx(h, l, c, length=14).iloc[:, 0] / 100
    df["vol_regime"]   = (df["vol_6"] > df["vol_6"].rolling(48).median()).astype(float)
    df["trend_regime"] = (c > c.ewm(span=55).mean()).astype(float)
    df["regime_bars"]  = df["trend_regime"].rolling(12).sum() / 12

    df.dropna(inplace=True)
    return df.reset_index(drop=True)


# ── Inference ─────────────────────────────────────────────────────────────────
def kelly_fraction(conf: float) -> float:
    if conf <= 0.5:
        return 0.0
    return float(np.clip((2 * conf - 1) * 0.5, 0.0, CFG["max_kelly"]))


@torch.no_grad()
def get_signal(model: OmegaV7, scaler, feature_cols: list, df: pd.DataFrame) -> dict:
    """
    Run inference on a fully-engineered DataFrame.

    Parameters
    ----------
    model       : OmegaV7 in eval mode
    scaler      : per-symbol StandardScaler fitted during training
    feature_cols: ordered list of 38 feature column names
    df          : output of engineer_features()
    """
    feat = df[feature_cols].values.astype(np.float32)

    # Apply per-symbol scaler (StandardScaler.transform expects 2-D array)
    feat = scaler.transform(feat)
    feat = np.nan_to_num(feat, nan=0.0, posinf=3.0, neginf=-3.0)
    feat = np.clip(feat, -5.0, 5.0)

    window = feat[-CFG["seq_len"]:]                               # (24, 38)
    x      = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1, 24, 38)

    logits = model(x)                                              # (1, 2)
    probs  = F.softmax(logits, dim=-1)[0].numpy()
    conf   = float(probs.max())
    cls    = int(probs.argmax())

    print(f"  probs: UP={probs[1]:.4f}  DOWN={probs[0]:.4f}  conf={conf:.4f}")

    action = "HOLD"
    if conf >= CFG["conf_threshold"]:
        action = "BUY" if cls == 1 else "SELL"

    return {
        "direction":  "UP" if cls == 1 else "DOWN",
        "prob_up":    float(probs[1]),
        "prob_down":  float(probs[0]),
        "confidence": conf,
        "action":     action,
        "kelly_f":    kelly_fraction(conf) if action != "HOLD" else 0.0,
    }


# ── Outcome checker ───────────────────────────────────────────────────────────
def check_outcomes(current_prices: dict) -> list:
    """
    Resolve entries that are at least TARGET_HORIZON hours old.
    Matches on 'symbol' key (e.g. 'BTC/USDT').
    """
    if not Path(ENTRY_LOG_FILE).exists():
        return []

    with open(ENTRY_LOG_FILE) as f:
        entries = json.load(f)

    now_ts              = datetime.now(timezone.utc).timestamp()
    resolve_after_min   = TARGET_HORIZON * 60 - 5   # 175 min
    resolved, remaining = [], []

    for e in entries:
        age_minutes = (now_ts - e["ts"]) / 60
        if age_minutes >= resolve_after_min:
            exit_price = current_prices.get(e["symbol"])
            if exit_price:
                e["exit_price"] = exit_price
                e["pct_move"]   = (exit_price - e["entry_price"]) / e["entry_price"]
                if e["action"] == "BUY":
                    e["correct"] = "WIN" if exit_price > e["entry_price"] else "LOSS"
                else:
                    e["correct"] = "WIN" if exit_price < e["entry_price"] else "LOSS"
                e["pnl"] = e["alloc_usd"] * abs(e["pct_move"]) * (
                    1 if e["correct"] == "WIN" else -1
                )
                resolved.append(e)
            else:
                remaining.append(e)
        else:
            remaining.append(e)

    if resolved:
        log_path = Path(LOG_FILE)
        df_new   = pd.DataFrame(resolved)
        df_out   = (
            pd.concat([pd.read_csv(log_path), df_new], ignore_index=True)
            if log_path.exists()
            else df_new
        )
        df_out.to_csv(log_path, index=False)
        print(f"✅ Resolved {len(resolved)} outcomes → {LOG_FILE}")
        for r in resolved:
            icon = "✅" if r["correct"] == "WIN" else "❌"
            print(
                f"  {icon} {r['symbol']} {r['action']} | "
                f"entry={r['entry_price']:.4f}  exit={r['exit_price']:.4f} | "
                f"{r['correct']} | P&L ${r['pnl']:.3f}"
            )

    with open(ENTRY_LOG_FILE, "w") as f:
        json.dump(remaining, f, indent=2)

    return resolved


# ── Discord notification ──────────────────────────────────────────────────────
def send_discord(signals: list, resolved: list, now_utc: datetime) -> None:
    action_icons = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⏸️"}

    lines = [
        f"📡 **OMEGA v7** | {now_utc.strftime('%Y-%m-%d %H:%M')} UTC | "
        f"crypto-only · 3h target · 38 features · RegimeGate",
        "```",
        f"{'Symbol':<10} {'Action':<6} {'↑UP':>6} {'↓DN':>6} {'Conf':>6} {'Kelly':>6} {'$':>6} {'Price':>12}",
        "─" * 68,
    ]
    for s in signals:
        lines.append(
            f"{s['symbol']:<10} "
            f"{action_icons[s['action']]} {s['action']:<4} "
            f"{s['prob_up']:>5.1%} {s['prob_down']:>5.1%} "
            f"{s['confidence']:>5.1%} {s['kelly_f']:>5.1%} "
            f"${s['alloc_usd']:>4.2f} {s['price']:>12.4f}"
        )
    lines.append("─" * 68)
    actionable = [s for s in signals if s["action"] != "HOLD"]
    lines.append(f"Actionable : {len(actionable)}/{len(signals)}")
    lines.append(
        f"Deploy     : ${sum(s['alloc_usd'] for s in actionable):.2f}"
        f" / ${CFG['total_capital']:.2f}"
    )
    lines.append("```")

    if resolved:
        lines.append("**Outcomes (3h prior signals):**")
        lines.append("```")
        for r in resolved:
            icon = "✅" if r["correct"] == "WIN" else "❌"
            lines.append(
                f"{icon} {r['symbol']} {r['action']} | "
                f"{r['entry_price']:.4f} → {r['exit_price']:.4f} | "
                f"{r['correct']} | P&L ${r['pnl']:.3f}"
            )
        wins  = sum(1 for r in resolved if r["correct"] == "WIN")
        total = len(resolved)
        lines.append(f"Win rate: {wins}/{total} ({wins/total:.0%})")
        lines.append("```")

    payload = {"content": "\n".join(lines), "username": "OMEGA v7"}
    try:
        resp = requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
        resp.raise_for_status()
        print("✅ Discord notification sent")
    except Exception as exc:
        print(f"⚠️  Discord failed: {exc}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    now_utc = datetime.now(timezone.utc)
    print(f"\n{'='*65}")
    print(f"  📡 OMEGA v7 | {now_utc.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"  Crypto-only · 3h target · 38 features · RegimeGate")
    print(f"{'='*65}")

    model, scalers, feature_cols, regime_cols = load_model()

    signals:        list = []
    current_prices: dict = {}

    for sym in CFG["symbols"]:
        pair = f"{sym}/USDT"
        try:
            print(f"\n  [{sym}] fetching...")
            df = fetch_crypto(sym, limit=150)

            if df.empty:
                print(f"  ⚠️  {sym}: no data returned")
                continue
            if len(df) < CFG["seq_len"] + 60:
                print(f"  ⚠️  {sym}: too few raw bars ({len(df)}) — skipping")
                continue

            df = engineer_features(df)

            if len(df) < CFG["seq_len"]:
                print(
                    f"  ⚠️  {sym}: too few rows after feature engineering "
                    f"({len(df)}) — skipping"
                )
                continue

            price = float(df["close"].iloc[-1])
            current_prices[pair] = price

            # Per-symbol scaler — fall back gracefully if symbol missing
            if sym in scalers:
                scaler = scalers[sym]
            else:
                fallback = next(iter(scalers.keys()))
                print(
                    f"  ⚠️  {sym}: no dedicated scaler — "
                    f"using '{fallback}' scaler as fallback"
                )
                scaler = scalers[fallback]

            s     = get_signal(model, scaler, feature_cols, df)
            alloc = CFG["total_capital"] * CFG["crypto_alloc"] * s["kelly_f"]

            signals.append(
                {
                    **s,
                    "symbol":       pair,
                    "price":        price,
                    "alloc_usd":    alloc,
                    "ts":           now_utc.timestamp(),
                    "datetime_utc": now_utc.strftime("%Y-%m-%d %H:%M"),
                }
            )

        except Exception as exc:
            print(f"  ⚠️  {sym}: {type(exc).__name__}: {exc}")

    # ── Resolve 3h-old entries ────────────────────────────────────────────
    resolved = check_outcomes(current_prices)

    # ── Print table ───────────────────────────────────────────────────────
    action_icons = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⏸️ "}
    print(
        f"\n{'Symbol':<12} {'Action':<6} {'↑UP':>6} {'↓DN':>6} "
        f"{'Conf':>6} {'Kelly':>6} {'$':>6} {'Price':>12}"
    )
    print("─" * 68)
    for s in signals:
        print(
            f"{s['symbol']:<12} "
            f"{action_icons[s['action']]} {s['action']:<4} "
            f"{s['prob_up']:>5.1%} {s['prob_down']:>5.1%} "
            f"{s['confidence']:>5.1%} {s['kelly_f']:>5.1%} "
            f"${s['alloc_usd']:>4.2f} {s['price']:>12.4f}"
        )
    print("─" * 68)
    actionable = [s for s in signals if s["action"] != "HOLD"]
    print(f"  Actionable : {len(actionable)}/{len(signals)}")
    print(
        f"  Deploy     : ${sum(s['alloc_usd'] for s in actionable):.2f}"
        f" / ${CFG['total_capital']:.2f}"
    )

    # ── Save entry prices for open positions ──────────────────────────────
    existing = (
        json.loads(Path(ENTRY_LOG_FILE).read_text())
        if Path(ENTRY_LOG_FILE).exists()
        else []
    )
    new_entries = [
        {
            "symbol":       s["symbol"],
            "action":       s["action"],
            "entry_price":  s["price"],
            "alloc_usd":    s["alloc_usd"],
            "confidence":   s["confidence"],
            "ts":           s["ts"],
            "datetime_utc": s["datetime_utc"],
        }
        for s in actionable
    ]
    with open(ENTRY_LOG_FILE, "w") as f:
        json.dump(existing + new_entries, f, indent=2)
    if new_entries:
        print(
            f"  📝 Logged {len(new_entries)} entry prices → "
            f"outcomes resolved in {TARGET_HORIZON}h"
        )

    # ── Discord ───────────────────────────────────────────────────────────
    send_discord(signals, resolved, now_utc)

    print(f"\n✅ Done | {now_utc.strftime('%Y-%m-%d %H:%M')} UTC")


if __name__ == "__main__":
    main()