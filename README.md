# OMEGA — Omni-Market Trading Signal System

> Transformer-based signal models for crypto, stocks, and forex. Hourly signals via GitHub Actions. Paper trading with outcome tracking.

---

> ⚠️ **Not financial advice.** This project is for research and educational purposes only. No real money is involved.

---

## Models

| Model | Markets | Data Source | Target | HuggingFace |
|---|---|---|---|---|
| OMEGA v3 | Crypto + Stocks | CryptoCompare, Polygon.io | 1h | `sato2ru/omega-v3-trading` |
| OMEGA Forex v3 | EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, GBP/JPY | Twelve Data | 1h | `sato2ru/omega-forex` |
| OMEGA v7 | BTC, ETH, SOL, XRP, ADA | CryptoCompare | 3h | `sato2ru/omega-v7` |

---

## Architecture

### OMEGA v3
```
Input (seq_len=24, 33 features)
  → Linear projection → d_model=64
  → Positional Encoding
  → 3× TransformerEncoderLayer (pre-norm, 4 heads, FFN=256)
  → Last token → LayerNorm → Linear(32) → GELU → Linear(2)
  → Softmax → P(UP), P(DOWN)
```
- **Parameters:** ~158K
- **Features:** 33 technical indicators (EMA 9/21/50, ADX, RSI, MACD, Stoch, CCI, ROC, MOM, ATR, BBands, OBV, body/wick/hl_range, log_volume)
- **Market embedding:** learned embedding per market type (crypto / stock)
- **Confidence threshold:** 0.60

### OMEGA Forex v3
```
Input (seq_len=24, 46 features)
  → Conv1d(k=3) → Conv1d(k=5)
  → RegimeGate (conditioned on 5 regime features)
  → 3× TransformerEncoderLayer (pre-norm, 4 heads, FFN=256)
  → Mean pool → LayerNorm → Linear(32) → GELU → Linear(2)
  → Softmax → P(UP), P(DOWN)
```
- **Parameters:** ~184K
- **Features:** 22 technical + 11 session + 5 Keplerian lag + 5 regime + 3 USD/JPY Standard Candle anchor (46 total)
- **Temperature scaling:** post-hoc calibration fitted on validation set
- **Per-pair thresholds:** calibrated per currency pair
- **Active pairs:** USD/JPY, USD/CHF — EUR/USD, GBP/USD, AUD/USD, GBP/JPY benched pending session feature improvements

### OMEGA v7
```
Input (seq_len=24, 38 features)
  → Conv1d(k=3) → Conv1d(k=3)
  → RegimeGate (conditioned on 5 regime features)
  → Learned positional embedding
  → 3× TransformerEncoderLayer (pre-norm, 4 heads, FFN=128)
  → Mean pool → LayerNorm → Linear(32) → GELU → Linear(2)
  → Softmax → P(UP), P(DOWN)
```
- **Parameters:** 148K
- **Features:** 33 ratio-based base indicators + 5 regime (38 total)
- **Target:** 3h forward close, 0.1% minimum move threshold
- **Confidence threshold:** 0.60

---

## Performance

### OMEGA v3
| Metric | Value |
|---|---|
| Overall test accuracy | ~52% |
| Hi-conf accuracy (≥0.60) | ~54.76% |

### OMEGA Forex v3
| Pair | Hi-Conf Accuracy | Threshold |
|---|---|---|
| USD/JPY | 68.0% | 0.55 |
| USD/CHF | 61.4% | 0.54 |
| Others | ~50% | Benched |

### OMEGA v7
| Metric | Value |
|---|---|
| Overall test accuracy | 54.23% |
| ROC-AUC | 52.84% |
| Hi-conf accuracy (≥0.60) | **58.70%** |
| Hi-conf coverage | 9.4% |
| vs v3 baseline (hi-conf) | +3.94pp |

All evaluation is done on a **strictly chronological held-out test set** — no shuffling, no future leakage.

---

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       ├── hourly.yml                   # OMEGA v3 — runs at :00 every hour
│       ├── omega-forex-signals.yml      # OMEGA Forex v3 — runs at :05 every hour
│       └── omega-v7-signals.yml         # OMEGA v7 — runs at :10 every hour
├── omega/
│   └── run.py                           # v3 inference + outcome tracking
├── omega_forex/
│   └── run.py                           # Forex v3 inference + outcome tracking
├── omega_v7/
│   └── run.py                           # v7 inference + outcome tracking
├── signals_log.csv                      # v3 resolved outcomes
├── entry_prices.json                    # v3 open positions
├── forex_signals_log.csv                # Forex resolved outcomes
├── forex_entry_prices.json              # Forex open positions
├── omega_v7_signals_log.csv             # v7 resolved outcomes (3h)
├── omega_v7_all_signals.csv             # v7 all signals audit trail
└── omega_v7_entry_prices.json           # v7 open positions
```

---

## How It Works

```
Every hour:
  1. Fetch latest 1H OHLC bars
  2. Compute technical indicators
  3. Scale features (train-fitted scaler, no leakage)
  4. Build 24-bar sequence window
  5. Run model → softmax probabilities
  6. Apply confidence threshold → BUY / SELL / HOLD
  7. Size position via half-Kelly (capped at 20%)
  8. Log entry prices for actionable signals
  9. Resolve outcomes from prior signals → write to CSV
 10. Post signal table + outcomes to Discord
 11. Commit updated logs to repo
```

v3 and Forex resolve after **1h**. v7 resolves after **3h**.

---

## Setup

### Prerequisites
- GitHub repository (private recommended)
- HuggingFace account with private model repos
- Twelve Data API key (Forex v3)
- Polygon.io API key (v3 stocks)
- Discord server with a webhook

### GitHub Secrets

| Secret | Used by |
|---|---|
| `HF_TOKEN` | All models — downloads weights from HuggingFace |
| `POLYGON_KEY` | OMEGA v3 — stock data |
| `TWELVE_DATA_API_KEY` | OMEGA Forex v3 — forex data |
| `DISCORD_WEBHOOK_FOREX` | All models — Discord signals |

### Workflow Permissions

**Settings → Actions → General → Workflow permissions → Read and write**
Required for the bot to commit log CSVs back to the repo.

---

## Training

All models are trained on Kaggle (free T4 GPU). Weights are hosted on HuggingFace (private repos — accessed via `HF_TOKEN`).

**Key training decisions:**
- Chronological train/val/test split (70/15/15) — no shuffling
- Scaler fitted on training data only — no future leakage
- Weekend rows stripped from forex data (OTC market gap artifacts)
- Early stopping with patience 15–20
- Warmup + cosine LR decay
- Temperature scaling on validation set (Forex only)
- Label smoothing = 0.05, class weights for imbalance (v7)

---

## Signal Output

```
📡 OMEGA v7 | 2026-03-29 06:10 UTC
Crypto-only · 3h target · 38 features · RegimeGate
════════════════════════════════════════════════════════════════
Symbol     Action    ↑UP    ↓DN   Conf  Kelly      $        Price
────────────────────────────────────────────────────────────────
BTC/USDT   ⏸️ HOLD  43.1% 56.9% 56.9%  0.0% $0.00   66861.58
ETH/USDT   🔴 SELL  39.4% 60.6% 60.6% 10.6% $2.65    2007.66
SOL/USDT   ⏸️ HOLD  44.2% 55.8% 55.8%  0.0% $0.00      82.75
────────────────────────────────────────────────────────────────
Actionable : 1/3   Deploy : $2.65 / $50.00
```

---

## Stack

`PyTorch` · `pandas-ta-classic` · `scikit-learn` · `HuggingFace Hub` · `Twelve Data` · `Polygon.io` · `CryptoCompare` · `GitHub Actions` · `Discord Webhooks`

---

## Roadmap

- [ ] Session-aware features for benched Forex pairs
- [ ] ForexFactory news masking (v8)
- [ ] Extended seq_len=48 for longer-horizon pairs
- [ ] BTC Standard Candle injection once lookahead risk resolved
- [ ] Live execution via OANDA API (after sustained paper trading edge confirmed)

---

*Three models, three markets, one pipeline. Built and iterated from scratch.*