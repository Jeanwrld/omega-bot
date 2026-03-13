# 🌐 OMEGA v3 — Automated Signal Bot

Runs every hour via GitHub Actions. Fetches live prices, generates BUY/SELL/HOLD signals, and automatically logs outcomes 1 hour later.

## Setup (one time)

### 1. Create the repo
- Go to github.com → New repository → name it `omega-bot` → Public or Private

### 2. Add Secrets
Go to **Settings → Secrets and variables → Actions → New repository secret**

| Secret name   | Value |
|---------------|-------|
| `POLYGON_KEY` | Your Polygon.io API key |
| `HF_TOKEN`    | Your HuggingFace token (huggingface.co/settings/tokens) |

### 3. Push these files
```bash
git init
git add .
git commit -m "initial commit"
git branch -M main
git remote add origin https://github.com/Jeanwrld/omega-bot.git
git push -u origin main
```

### 4. Enable Actions
Go to the **Actions** tab in your repo → click "I understand my workflows, go ahead and enable them"

### 5. Test it manually
Actions tab → "OMEGA v3 Hourly Signals" → "Run workflow" button

---

## Output files (auto-updated every hour)

### `signals_log.csv`
Every resolved BUY/SELL signal with outcome:
| Column | Description |
|--------|-------------|
| datetime_utc | When signal fired |
| symbol | e.g. BTC/USDT |
| action | BUY or SELL |
| entry_price | Price when signal fired |
| exit_price | Price 1 hour later |
| correct | WIN or LOSS |
| pnl | Estimated P&L in $ |
| confidence | Model confidence % |

### `entry_prices.json`
Pending signals waiting for their 1hr exit price check (internal use)

---

## Monitoring
- Check **Actions tab** to see each hourly run log
- Check **signals_log.csv** for paper trade results
- Target: WIN rate > 54.76% over 50+ trades

> ⚠️ Paper trade only. Not financial advice.
