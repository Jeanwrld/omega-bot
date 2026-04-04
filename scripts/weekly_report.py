#!/usr/bin/env python3
"""
OMEGA Weekly Performance Report
Runs every Sunday 00:00 UTC via GitHub Actions.
Reads resolved CSVs, computes rolling stats, posts to Discord.
"""

import os
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path

DISCORD_WEBHOOK = os.environ["DISCORD_WEBHOOK_FOREX"]

SOURCES = {
    "Forex v3": {
        "file":       "forex_signals_log.csv",
        "pair_col":   "pair",
        "outcome_col":"outcome",
        "pnl_col":    "pnl",
        "pnl_pct_col":"pnl_pct",   # already in %
        "conf_col":   "confidence",
        "pnl_is_pct": True,        # pnl_pct is % not decimal
    },
    "v7 Crypto": {
        "file":       "omega_v7_signals_log.csv",
        "pair_col":   "symbol",
        "outcome_col":"correct",
        "pnl_col":    "pnl",
        "pnl_pct_col":"pct_move",  # decimal
        "conf_col":   "confidence",
        "pnl_is_pct": False,
    },
}

WINDOW_DAYS  = 7    # rolling window for weekly stats
LOOKBACK     = 30   # days of history for trend


def load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(p)


def compute_stats(df: pd.DataFrame, src: dict, days: int) -> dict | None:
    if df.empty:
        return None

    # Find datetime column
    for col in ["datetime", "datetime_utc"]:
        if col in df.columns:
            df["_dt"] = pd.to_datetime(df[col], errors="coerce")
            break
    else:
        return None

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    week   = df[df["_dt"] >= cutoff].copy()

    if week.empty:
        return None

    outcome_col = src["outcome_col"]
    pnl_col     = src["pnl_col"]
    pnl_pct_col = src["pnl_pct_col"]
    conf_col    = src["conf_col"]

    wins   = (week[outcome_col] == "WIN").sum()
    losses = (week[outcome_col] == "LOSS").sum()
    total  = wins + losses

    if total == 0:
        return None

    win_rate = wins / total

    # P&L
    pnl_total = week[pnl_col].apply(pd.to_numeric, errors="coerce").sum() if pnl_col in week else 0

    # Move % — normalise to decimal
    if pnl_pct_col in week.columns:
        moves = week[pnl_pct_col].apply(pd.to_numeric, errors="coerce").dropna()
        if src["pnl_is_pct"]:
            moves = moves / 100
    else:
        moves = pd.Series([], dtype=float)

    # Sharpe (annualised, hourly)
    if len(moves) > 1:
        mean  = moves.mean()
        std   = moves.std()
        sharpe = (mean / std) * np.sqrt(24 * 365) if std > 0 else 0
    else:
        sharpe = 0

    # Max drawdown
    cum = moves.cumsum()
    roll_max = cum.cummax()
    drawdown = (cum - roll_max).min()

    # Avg confidence
    avg_conf = week[conf_col].apply(pd.to_numeric, errors="coerce").mean() if conf_col in week else 0

    # Best / worst pair
    pair_col = src["pair_col"]
    by_pair  = {}
    if pair_col in week.columns:
        for pair, grp in week.groupby(pair_col):
            w = (grp[outcome_col] == "WIN").sum()
            t = len(grp)
            by_pair[pair] = {"wins": w, "total": t, "wr": w / t if t else 0}

    return {
        "wins":     wins,
        "losses":   losses,
        "total":    total,
        "win_rate": win_rate,
        "pnl":      pnl_total,
        "sharpe":   sharpe,
        "drawdown": drawdown,
        "avg_conf": avg_conf,
        "by_pair":  by_pair,
    }


def trend_arrow(current: float, previous: float) -> str:
    if current > previous + 0.01:  return "↑"
    if current < previous - 0.01:  return "↓"
    return "→"


def build_message(now: datetime) -> str:
    week_str = now.strftime("%Y-%m-%d")
    lines    = [
        f"📊 **OMEGA Weekly Report** — w/e {week_str}",
        f"*Rolling 7-day performance across all active models*",
        "",
    ]

    any_data = False

    for model_name, src in SOURCES.items():
        df = load_csv(src["file"])
        stats = compute_stats(df, src, days=WINDOW_DAYS)

        if not stats:
            lines.append(f"**{model_name}** — no resolved signals this week")
            lines.append("")
            continue

        any_data   = True
        wr_pct     = stats["win_rate"] * 100
        wr_icon    = "🟢" if wr_pct >= 55 else "🟡" if wr_pct >= 50 else "🔴"
        pnl_sign   = "+" if stats["pnl"] >= 0 else ""
        sh_icon    = "🟢" if stats["sharpe"] >= 1 else "🟡" if stats["sharpe"] >= 0 else "🔴"
        dd_pct     = stats["drawdown"] * 100

        lines.append(f"**{model_name}**")
        lines.append("```")
        lines.append(f"  Signals   : {stats['total']}  ({stats['wins']}W / {stats['losses']}L)")
        lines.append(f"  Win Rate  : {wr_icon} {wr_pct:.1f}%")
        lines.append(f"  P&L       : {pnl_sign}${stats['pnl']:.3f}")
        lines.append(f"  Sharpe    : {sh_icon} {stats['sharpe']:.2f}  (annualised)")
        lines.append(f"  Max DD    : {dd_pct:.3f}%")
        lines.append(f"  Avg Conf  : {stats['avg_conf']*100:.1f}%")

        if stats["by_pair"]:
            lines.append("")
            lines.append("  By pair:")
            for pair, p in sorted(stats["by_pair"].items(), key=lambda x: -x[1]["wr"]):
                icon = "🟢" if p["wr"] >= 0.55 else "🟡" if p["wr"] >= 0.50 else "🔴"
                lines.append(f"    {icon} {pair:<12} {p['wins']}/{p['total']}  ({p['wr']*100:.0f}%)")

        lines.append("```")
        lines.append("")

    if not any_data:
        lines.append("*No resolved signals across any model this week.*")

    lines.append("*Next report: Sunday 00:00 UTC*")
    return "\n".join(lines)


def main():
    now = datetime.now(timezone.utc)
    print(f"\n=== OMEGA Weekly Report — {now.strftime('%Y-%m-%d %H:%M UTC')} ===\n")

    message = build_message(now)
    print(message)

    payload = {"content": message, "username": "OMEGA Weekly"}
    try:
        r = requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
        r.raise_for_status()
        print("\n✅ Weekly report posted to Discord")
    except Exception as e:
        print(f"\n⚠️  Discord post failed: {e}")


if __name__ == "__main__":
    main()
