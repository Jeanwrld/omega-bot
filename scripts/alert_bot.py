#!/usr/bin/env python3
"""
OMEGA Alert Bot
Runs every hour at :15 UTC via GitHub Actions.
Checks rolling win rates and fires Discord alerts when models degrade.
"""

import os
import requests
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

DISCORD_WEBHOOK = os.environ["DISCORD_WEBHOOK_FOREX"]

# Alert thresholds
WIN_RATE_WINDOW   = 20    # last N resolved signals to check
WIN_RATE_FLOOR    = 0.48  # alert if win rate drops below this
LOSS_STREAK_ALERT = 5     # alert if N consecutive losses
SIGNAL_GAP_HOURS  = 6     # alert if no signals for this long (model may be stuck)

SOURCES = {
    "Forex v3": {
        "file":       "forex_signals_log.csv",
        "outcome_col":"outcome",
        "dt_col":     "datetime",
    },
    "v7 Crypto": {
        "file":       "omega_v7_signals_log.csv",
        "outcome_col":"correct",
        "dt_col":     "datetime_utc",
    },
}


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(p)


def check_model(name: str, src: dict) -> list[str]:
    """Returns list of alert strings, empty if all clear."""
    alerts = []
    df = load_csv(src["file"])

    if df.empty:
        return []

    outcome_col = src["outcome_col"]
    dt_col      = src["dt_col"]

    if outcome_col not in df.columns:
        return []

    # Parse datetimes
    df["_dt"] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=["_dt"]).sort_values("_dt")

    outcomes = df[outcome_col].tolist()

    # ── Win rate check ────────────────────────────────────────────────────────
    recent = outcomes[-WIN_RATE_WINDOW:]
    if len(recent) >= 10:
        wins    = sum(1 for o in recent if o == "WIN")
        wr      = wins / len(recent)
        if wr < WIN_RATE_FLOOR:
            alerts.append(
                f"⚠️ **{name}** win rate is **{wr*100:.1f}%** "
                f"over last {len(recent)} signals (floor: {WIN_RATE_FLOOR*100:.0f}%)"
            )

    # ── Loss streak check ─────────────────────────────────────────────────────
    streak = 0
    for o in reversed(outcomes):
        if o == "LOSS":
            streak += 1
        else:
            break
    if streak >= LOSS_STREAK_ALERT:
        alerts.append(
            f"🔴 **{name}** is on a **{streak}-loss streak**"
        )

    # ── Signal gap check ──────────────────────────────────────────────────────
    if not df.empty:
        last_signal = df["_dt"].iloc[-1]
        now_utc     = pd.Timestamp.now(tz="UTC").tz_localize(None)
        gap_hours   = (now_utc - last_signal).total_seconds() / 3600
        if gap_hours > SIGNAL_GAP_HOURS:
            alerts.append(
                f"⏰ **{name}** last signal was **{gap_hours:.1f}h ago** — "
                f"model may be stalled"
            )

    return alerts


def main():
    now = datetime.now(timezone.utc)
    print(f"\n=== OMEGA Alert Bot — {now.strftime('%Y-%m-%d %H:%M UTC')} ===\n")

    all_alerts = []
    for name, src in SOURCES.items():
        alerts = check_model(name, src)
        for a in alerts:
            print(f"  ALERT: {a}")
        all_alerts.extend(alerts)

    if not all_alerts:
        print("  ✅ All clear — no alerts triggered")
        return

    # Build Discord message
    lines = [
        f"🚨 **OMEGA Alert** — {now.strftime('%Y-%m-%d %H:%M')} UTC",
        "",
    ]
    lines.extend(all_alerts)
    lines.append("")
    lines.append("*Check the dashboard for details.*")

    payload = {"content": "\n".join(lines), "username": "OMEGA Alerts"}
    try:
        r = requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
        r.raise_for_status()
        print(f"\n✅ {len(all_alerts)} alert(s) posted to Discord")
    except Exception as e:
        print(f"\n⚠️  Discord post failed: {e}")


if __name__ == "__main__":
    main()
