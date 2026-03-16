import os, time, json, pickle, requests, warnings
import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import torch
import torch.nn as nn
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Constants ─────────────────────────────────────────────────────────────────
HF_REPO        = 'sato2ru/omega-forex'
TWELVE_KEY     = os.environ['TWELVE_DATA_API_KEY']
DISCORD_WEBHOOK= os.environ['DISCORD_WEBHOOK_FOREX']
LOG_FILE       = 'forex_signals_log.csv'
ENTRY_LOG_FILE = 'forex_entry_prices.json'

CFG = {
    # Only the two pairs with proven edge — bench the rest until more data
    'pairs'      : ['USD/JPY', 'USD/CHF'],
    'seq_len'    : 24,
    'd_model'    : 64,
    'n_heads'    : 4,
    'n_layers'   : 3,
    'dropout'    : 0.1,
    'conf_threshold': 0.55,
    'max_kelly'  : 0.20,
    'total_capital': 50.0,
}

device = torch.device('cpu')


# ── Model (identical to notebook) ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class OmegaForex(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=3,
                 dropout=0.1, seq_len=24):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=seq_len + 8, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder    = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 32),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 2))

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.classifier(x[:, -1, :])


# ── Load model from HuggingFace ───────────────────────────────────────────────
def load_model():
    from huggingface_hub import hf_hub_download
    token = os.environ.get('HF_TOKEN')

    model_path  = hf_hub_download(HF_REPO, 'model.pt',    token=token)
    scaler_path = hf_hub_download(HF_REPO, 'scalers.pkl', token=token)
    config_path = hf_hub_download(HF_REPO, 'config.json', token=token)

    with open(config_path) as f:
        cfg = json.load(f)

    feature_cols = cfg['feature_cols']
    pair_thresholds = cfg.get('test_metrics', {}).get('pair_thresholds', {})
    temperature     = cfg.get('test_metrics', {}).get('temperature', 1.0)

    model = OmegaForex(
        n_features=len(feature_cols),
        d_model=cfg['d_model'], n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'], dropout=cfg['dropout'],
        seq_len=cfg['seq_len'],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)

    print(f'✅ OMEGA Forex loaded | T={temperature:.4f}')
    print(f'   pairs      : {list(scalers.keys())}')
    print(f'   features   : {len(feature_cols)}')
    print(f'   thresholds : {pair_thresholds}')

    return model, scalers, feature_cols, temperature, pair_thresholds


# ── Data fetch ────────────────────────────────────────────────────────────────
def fetch_forex(pair, n_bars=100):
    """Fetch recent 1H bars from Twelve Data. Returns clean OHLC DataFrame."""
    url = 'https://api.twelvedata.com/time_series'
    params = {
        'symbol'    : pair,
        'interval'  : '1h',
        'outputsize': n_bars,
        'apikey'    : TWELVE_KEY,
        'format'    : 'JSON',
        'timezone'  : 'UTC',
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if 'values' not in data:
        raise RuntimeError(f'Twelve Data error for {pair}: {data.get("message", data)}')

    df = pd.DataFrame(data['values'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['volume'] = 0.0   # forex has no exchange volume
    df = df.dropna(subset=['open', 'high', 'low', 'close'])

    # Strip weekends
    df = df[df['datetime'].dt.dayofweek < 5].reset_index(drop=True)
    return df


# ── Feature engineering (mirrors notebook exactly) ───────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    d['ema9']  = ta.ema(d['close'], length=9)
    d['ema21'] = ta.ema(d['close'], length=21)
    d['ema50'] = ta.ema(d['close'], length=50)
    d['ema9_ratio']  = d['close'] / d['ema9']  - 1
    d['ema21_ratio'] = d['close'] / d['ema21'] - 1
    d['ema50_ratio'] = d['close'] / d['ema50'] - 1

    adx = ta.adx(d['high'], d['low'], d['close'], length=14)
    d['adx'] = adx['ADX_14']
    d['dmp'] = adx['DMP_14']
    d['dmn'] = adx['DMN_14']

    d['rsi'] = ta.rsi(d['close'], length=14)

    macd = ta.macd(d['close'], fast=12, slow=26, signal=9)
    d['macd']        = macd['MACD_12_26_9']
    d['macd_signal'] = macd['MACDs_12_26_9']
    d['macd_hist']   = macd['MACDh_12_26_9']

    stoch = ta.stoch(d['high'], d['low'], d['close'], k=14, d=3)
    d['stoch_k'] = stoch['STOCHk_14_3_3']
    d['stoch_d'] = stoch['STOCHd_14_3_3']

    d['cci'] = ta.cci(d['high'], d['low'], d['close'], length=14)
    d['roc'] = ta.roc(d['close'], length=10)
    d['mom'] = ta.mom(d['close'], length=10)
    d['atr'] = ta.atr(d['high'], d['low'], d['close'], length=14)

    bb = ta.bbands(d['close'], length=20, std=2)
    bb_cols = bb.columns.tolist()   # [BBL, BBM, BBU, BBB, BBP]
    d['bb_upper']  = bb[bb_cols[2]]
    d['bb_middle'] = bb[bb_cols[1]]
    d['bb_lower']  = bb[bb_cols[0]]
    d['bb_width']  = (d['bb_upper'] - d['bb_lower']) / d['bb_middle']
    d['bb_pct']    = (d['close'] - d['bb_lower']) / (d['bb_upper'] - d['bb_lower'] + 1e-10)

    d['body']     = (d['close'] - d['open']).abs() / (d['high'] - d['low'] + 1e-10)
    d['wick_up']  = (d['high'] - d[['open', 'close']].max(axis=1)) / (d['high'] - d['low'] + 1e-10)
    d['wick_dn']  = (d[['open', 'close']].min(axis=1) - d['low'])  / (d['high'] - d['low'] + 1e-10)
    d['hl_range'] = (d['high'] - d['low']) / d['close']

    return d.dropna().reset_index(drop=True)


# ── Kelly sizing ──────────────────────────────────────────────────────────────
def kelly_fraction(win_prob, max_kelly=CFG['max_kelly']):
    if win_prob <= 0.5:
        return 0.0
    f = (2 * win_prob - 1) * 0.5   # half-Kelly
    return float(np.clip(f, 0, max_kelly))


# ── Inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def get_signal(model, scalers, feature_cols, df, pair, temperature, pair_thresholds):
    scaler = scalers.get(pair)
    feat   = df[feature_cols].copy()
    if scaler:
        feat[feature_cols] = scaler.transform(feat[feature_cols].values)

    window = feat.iloc[-CFG['seq_len']:].values
    if len(window) < CFG['seq_len']:
        raise ValueError(f'Not enough bars for {pair}: {len(window)} < {CFG["seq_len"]}')

    x      = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
    logits = model(x) / temperature
    probs  = torch.softmax(logits, dim=1)[0].numpy()

    conf   = float(probs.max())
    cls    = int(probs.argmax())

    # Use pair-specific threshold if available, else global
    threshold = pair_thresholds.get(pair, CFG['conf_threshold'])
    action = 'HOLD'
    if conf >= threshold:
        action = 'BUY' if cls == 1 else 'SELL'

    return {
        'direction': 'UP' if cls == 1 else 'DOWN',
        'prob_up'  : float(probs[1]),
        'prob_dn'  : float(probs[0]),
        'confidence': conf,
        'threshold' : threshold,
        'action'   : action,
        'kelly_f'  : kelly_fraction(conf) if action != 'HOLD' else 0.0,
    }


# ── Outcome checker ───────────────────────────────────────────────────────────
def check_outcomes(current_prices):
    if not Path(ENTRY_LOG_FILE).exists():
        return []

    with open(ENTRY_LOG_FILE) as f:
        entries = json.load(f)

    now_ts = datetime.now(timezone.utc).timestamp()
    resolved, remaining = [], []

    for e in entries:
        age_min = (now_ts - e['ts']) / 60
        if age_min >= 55:
            exit_price = current_prices.get(e['symbol'])
            if exit_price:
                e['exit_price'] = exit_price
                e['pct_move']   = (exit_price - e['entry_price']) / e['entry_price']
                if e['action'] == 'BUY':
                    e['correct'] = 'WIN' if exit_price > e['entry_price'] else 'LOSS'
                else:
                    e['correct'] = 'WIN' if exit_price < e['entry_price'] else 'LOSS'
                e['pnl'] = e['alloc_usd'] * abs(e['pct_move']) * (1 if e['correct'] == 'WIN' else -1)
                resolved.append(e)
            else:
                remaining.append(e)
        else:
            remaining.append(e)

    if resolved:
        log_path = Path(LOG_FILE)
        df_new = pd.DataFrame(resolved)
        if log_path.exists():
            df_out = pd.concat([pd.read_csv(log_path), df_new], ignore_index=True)
        else:
            df_out = df_new
        df_out.to_csv(log_path, index=False)
        print(f'✅ Resolved {len(resolved)} outcomes → {LOG_FILE}')
        for r in resolved:
            icon = '✅' if r['correct'] == 'WIN' else '❌'
            print(f'  {icon} {r["symbol"]} {r["action"]} | '
                  f'entry={r["entry_price"]:.5f} exit={r["exit_price"]:.5f} | '
                  f'{r["correct"]} | P&L ${r["pnl"]:.3f}')

    with open(ENTRY_LOG_FILE, 'w') as f:
        json.dump(remaining, f)

    return resolved


# ── Discord notification ──────────────────────────────────────────────────────
def send_discord(signals, resolved, now_utc):
    action_icons = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '⏸️'}

    # ── Signal block ──────────────────────────────────────────────────────────
    lines = [
        f'📡 **OMEGA Forex** | {now_utc.strftime("%Y-%m-%d %H:%M")} UTC',
        '```',
        f'{"Pair":<10} {"Action":<6} {"↑UP":>6} {"↓DN":>6} {"Conf":>6} {"Thr":>6} {"Kelly":>6} {"Price":>10}',
        '─' * 60,
    ]
    for s in signals:
        icon = action_icons[s['action']]
        lines.append(
            f'{s["symbol"]:<10} {icon} {s["action"]:<4} '
            f'{s["prob_up"]:>5.1%} {s["prob_dn"]:>5.1%} '
            f'{s["confidence"]:>5.1%} {s["threshold"]:>5.1%} '
            f'{s["kelly_f"]:>5.1%} {s["price"]:>10.5f}'
        )
    lines.append('─' * 60)

    actionable = [s for s in signals if s['action'] != 'HOLD']
    lines.append(f'Actionable : {len(actionable)}/{len(signals)}')
    lines.append(f'Deploy     : ${sum(s["alloc_usd"] for s in actionable):.2f} / ${CFG["total_capital"]:.2f}')
    lines.append('```')

    # ── Outcomes block (if any resolved) ──────────────────────────────────────
    if resolved:
        lines.append('**Outcomes from last hour:**')
        lines.append('```')
        for r in resolved:
            icon = '✅' if r['correct'] == 'WIN' else '❌'
            lines.append(
                f'{icon} {r["symbol"]} {r["action"]} | '
                f'{r["entry_price"]:.5f} → {r["exit_price"]:.5f} | '
                f'{r["correct"]} | P&L ${r["pnl"]:.3f}'
            )
        wins  = sum(1 for r in resolved if r['correct'] == 'WIN')
        total = len(resolved)
        lines.append(f'Win rate: {wins}/{total} ({wins/total:.0%})')
        lines.append('```')

    payload = {'content': '\n'.join(lines), 'username': 'OMEGA Forex'}
    try:
        r = requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
        r.raise_for_status()
        print('✅ Discord notification sent')
    except Exception as e:
        print(f'⚠️ Discord failed: {e}')


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    now_utc = datetime.now(timezone.utc)
    print(f'\n{"="*60}')
    print(f'  📡 OMEGA Forex | {now_utc.strftime("%Y-%m-%d %H:%M")} UTC')
    print(f'{"="*60}')

    model, scalers, feature_cols, T, pair_thresholds = load_model()

    signals        = []
    current_prices = {}

    for pair in CFG['pairs']:
        try:
            print(f'\n  Fetching {pair} ...', end=' ')
            df = fetch_forex(pair, n_bars=100)
            if len(df) < CFG['seq_len'] + 10:
                print(f'not enough bars ({len(df)}), skipping')
                continue
            df = build_features(df)
            price = float(df['close'].iloc[-1])
            current_prices[pair] = price
            print(f'{len(df)} bars | close={price:.5f}')

            s = get_signal(model, scalers, feature_cols, df, pair, T, pair_thresholds)
            alloc = CFG['total_capital'] * s['kelly_f']
            signals.append({
                **s,
                'symbol'      : pair,
                'price'       : price,
                'alloc_usd'   : alloc,
                'ts'          : now_utc.timestamp(),
                'datetime_utc': now_utc.strftime('%Y-%m-%d %H:%M'),
            })
            time.sleep(8)   # Twelve Data rate limit
        except Exception as e:
            print(f'  ⚠️  {pair}: {e}')

    # ── Check outcomes for signals logged ~1hr ago ────────────────────────────
    resolved = check_outcomes(current_prices)

    # ── Print table ───────────────────────────────────────────────────────────
    action_icons = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '⏸️ '}
    print(f"\n{'Pair':<10} {'Action':<6} {'↑UP':>6} {'↓DN':>6} {'Conf':>6} {'Thr':>6} {'Kelly':>6} {'Price':>10}")
    print('─' * 62)
    for s in signals:
        print(f"{s['symbol']:<10} {action_icons[s['action']]} {s['action']:<4} "
              f"{s['prob_up']:>5.1%} {s['prob_dn']:>5.1%} "
              f"{s['confidence']:>5.1%} {s['threshold']:>5.1%} "
              f"{s['kelly_f']:>5.1%} {s['price']:>10.5f}")
    print('─' * 62)

    actionable = [s for s in signals if s['action'] != 'HOLD']
    print(f'  Actionable : {len(actionable)}/{len(signals)}')
    print(f'  Deploy     : ${sum(s["alloc_usd"] for s in actionable):.2f} / ${CFG["total_capital"]:.2f}')

    # ── Log entry prices for BUY/SELL signals ─────────────────────────────────
    existing = json.loads(Path(ENTRY_LOG_FILE).read_text()) if Path(ENTRY_LOG_FILE).exists() else []
    new_entries = [
        {
            'symbol'      : s['symbol'],
            'action'      : s['action'],
            'entry_price' : s['price'],
            'alloc_usd'   : s['alloc_usd'],
            'confidence'  : s['confidence'],
            'ts'          : s['ts'],
            'datetime_utc': s['datetime_utc'],
        }
        for s in actionable
    ]
    Path(ENTRY_LOG_FILE).write_text(json.dumps(existing + new_entries, indent=2))
    if new_entries:
        print(f'  📝 Logged {len(new_entries)} entry prices')

    # ── Discord ───────────────────────────────────────────────────────────────
    send_discord(signals, resolved, now_utc)

    print(f'\n✅ Done | {now_utc.strftime("%Y-%m-%d %H:%M")} UTC')


if __name__ == '__main__':
    main()