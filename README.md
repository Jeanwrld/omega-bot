# OMEGA — Omni-Market Trading Signal System

> Transformer-based signal models for crypto, stocks, and forex. Hourly signals via GitHub Actions. Paper trading with outcome tracking.

---

## Overview

OMEGA is a collection of neural trading signal models built on a shared Transformer architecture. Each model is trained independently per market domain and deployed as a scheduled GitHub Action that generates hourly BUY/SELL/HOLD signals, tracks entry prices, and resolves outcomes one hour later.

| Model | Markets | Data Source | HuggingFace |
|---|---|---|---|
| OMEGA v3 | Crypto + Stocks | CryptoCompare, Polygon.io | `sato2ru/omega-v3-trading` |
| OMEGA Forex | EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, GBP/JPY | Twelve Data | `sato2ru/omega-forex` |

> ⚠️ **Not financial advice.** This project is for research and educational purposes only. No real money is involved.

---

## Architecture

Both models share the same Transformer backbone:

```
Input (seq_len=24, n_features) 
  → Linear projection → d_model=64
  → Positional Encoding
  → 3× TransformerEncoderLayer (pre-norm, 4 heads, FFN=256)
  → Last token → LayerNorm → Linear(32) → GELU → Linear(2)
  → Softmax → P(UP), P(DOWN)
```

**Signal logic:** if `max(P(UP), P(DOWN)) ≥ conf_threshold` → BUY or SELL, else HOLD.  
**Position sizing:** half-Kelly criterion, capped at 20% of allocated capital.

### OMEGA v3
- **Parameters:** ~158K
- **Features:** 33 technical indicators (EMA 9/21/50, ADX, RSI, MACD, Stoch, CCI, ROC, MOM, ATR, BBands, OBV, body/wick/hl_range, log_volume)
- **Market embedding:** learned embedding per market type (crypto / stock / prediction)
- **Confidence threshold:** 0.60

### OMEGA Forex
- **Parameters:** ~158K  
- **Features:** 22 technical indicators (ratio-normalised EMAs, ADX, RSI, MACD, Stoch, CCI, ROC, MOM, ATR, BBands, price structure — no volume, OTC market)
- **Temperature scaling:** post-hoc calibration fitted on validation set
- **Per-pair thresholds:** calibrated per currency pair on validation set
- **Active pairs:** USD/JPY, USD/CHF (proven edge) — EUR/USD, GBP/USD, AUD/USD, GBP/JPY benched pending session feature improvements

---

## Performance

### OMEGA v3
| Metric | Value |
|---|---|
| Overall test accuracy | ~52% |
| High-confidence accuracy (≥60%) | ~54.76% |
| Split | Chronological 70/15/15 |

### OMEGA Forex
| Pair | Hi-Conf Accuracy | Signals | Threshold |
|---|---|---|---|
| USD/JPY | 68.0% | 50 | 0.55 |
| USD/CHF | 61.4% | 44 | 0.54 |
| Others | ~50% | — | Benched |

All evaluation is done on a **strictly chronological held-out test set** — no shuffling, no future leakage.

---

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       ├── hourly.yml                 # OMEGA v3 — runs at :00 every hour
│       └── omega-forex-signals.yml    # OMEGA Forex — runs at :05 every hour
├── omega/
│   └── run.py                         # v3 inference + outcome tracking
├── omega_forex/
│   └── run.py                         # Forex inference + outcome tracking
├── signals_log.csv                    # v3 resolved outcomes (auto-committed)
├── entry_prices.json                  # v3 open positions
├── forex_signals_log.csv              # Forex resolved outcomes (auto-committed)
└── forex_entry_prices.json            # Forex open positions
```

---

## How It Works

```
Every hour:
  1. Fetch latest 1H OHLC bars
  2. Compute technical indicators
  3. Scale features (train-fitted scaler, no leakage)
  4. Build 24-bar sequence window
  5. Run Transformer → softmax probabilities
  6. Apply confidence threshold → BUY / SELL / HOLD
  7. Size position via half-Kelly
  8. Log entry prices for BUY/SELL signals
  9. Resolve outcomes from ~1hr ago → write to CSV
 10. Post signal table + outcomes to Discord
 11. Commit updated logs to repo
```

---

## Setup

### Prerequisites
- GitHub repository (private recommended)
- HuggingFace account
- Twelve Data API key (free tier, for forex)
- Polygon.io API key (for stocks)
- Discord server with a webhook

### GitHub Secrets

Go to **Settings → Secrets and variables → Actions** and add:

| Secret | Used by |
|---|---|
| `HF_TOKEN` | Both models — downloads weights from HuggingFace |
| `POLYGON_KEY` | OMEGA v3 — stock data via Polygon.io |
| `TWELVE_DATA_API_KEY` | OMEGA Forex — forex data via Twelve Data |
| `DISCORD_WEBHOOK_FOREX` | Both models — posts signals to Discord |

### Workflow Permissions

**Settings → Actions → General → Workflow permissions → Read and write**  
Required for the bot to commit `signals_log.csv` back to the repo.

---

## Training

Both models are trained on Kaggle (free T4 GPU). Training notebooks are not included in this repo but the trained weights are hosted on HuggingFace.

**Key training decisions:**
- Chronological train/val/test split (70/15/15) — no shuffling
- Scaler fitted on training data only — no future leakage
- Weekend rows stripped from forex data (OTC market gap artifacts)
- Early stopping with patience=15
- Warmup (10 epochs) + cosine LR decay
- Temperature scaling on validation set post-training (forex only)

---

## Signal Output

```
📡 OMEGA Forex | 2026-03-14 09:05 UTC
──────────────────────────────────────────────────────────────
Pair       Action    ↑UP    ↓DN   Conf    Thr  Kelly      Price
──────────────────────────────────────────────────────────────
USD/JPY    🔴 SELL  47.4%  52.6%  52.6%  50.0%   2.6%  159.735
USD/CHF    🟢 BUY   56.1%  43.9%  56.1%  54.0%   6.1%    0.796
──────────────────────────────────────────────────────────────
Actionable : 2/2   Deploy : $4.33 / $50.00
```

---

## Roadmap

- [ ] Session-aware features (hour of day, London/NY overlap) for EUR/USD and GBP/USD
- [ ] Extended seq_len (48) for longer-horizon pairs
- [ ] 2+ years training data via paginated Twelve Data fetch
- [ ] Activate benched forex pairs once session features are added
- [ ] Live execution via OANDA API (after sustained paper trading edge confirmed)

---

## Stack

`PyTorch` · `pandas-ta-classic` · `scikit-learn` · `HuggingFace Hub` · `Twelve Data` · `Polygon.io` · `CryptoCompare` · `GitHub Actions` · `Discord Webhooks`

---

*Built and iterated from scratch. OMEGA v3 handles crypto + stocks. OMEGA Forex handles currency pairs. Clean separation, no interference.*
