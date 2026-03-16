import os, math, time, json, pickle, requests, warnings
import numpy as np
import pandas as pd
import pandas_ta_classic as ta
import torch
import torch.nn as nn
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Constants
MARKET_TYPES   = {'crypto': 0, 'stock': 1, 'prediction': 2}
HF_REPO        = 'sato2ru/omega-v3-trading'
POLYGON_KEY    = os.environ['POLYGON_KEY']
DISCORD_WEBHOOK= os.environ['DISCORD_WEBHOOK_FOREX']
LOG_FILE       = 'signals_log.csv'
ENTRY_LOG_FILE = 'entry_prices.json'

CFG = {
    'crypto_pairs':   ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT'],
    'stock_tickers':  ['SPY', 'QQQ', 'NVDA', 'TSLA', 'AAPL'],
    'seq_len':        24,
    'd_model':        64, 'nhead': 4, 'num_layers': 3,
    'dropout':        0.3, 'num_market_types': 3, 'num_classes': 2,
    'conf_threshold': 0.60, 'max_kelly': 0.20,
    'total_capital':  50.0, 'crypto_alloc': 0.50, 'stock_alloc': 0.30,
}

device = torch.device('cpu')


# ── Model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class OmegaModel(nn.Module):
    def __init__(self, num_features, d_model=64, nhead=4,
                 num_layers=3, dropout=0.3, num_market_types=3, num_classes=2):
        super().__init__()
        self.market_emb = nn.Embedding(num_market_types, d_model)
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_enc   = PositionalEncoding(d_model, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True)
        self.backbone      = nn.TransformerEncoder(enc_layer, num_layers=num_layers,
                                                    norm=nn.LayerNorm(d_model))
        H = d_model // 2
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, H), nn.GELU(), nn.Dropout(dropout), nn.Linear(H, num_classes))
        self.pred_head = nn.Sequential(
            nn.Linear(d_model, H), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(H, 1), nn.Sigmoid())
        self.conf_head = nn.Sequential(
            nn.Linear(d_model, H//2), nn.GELU(), nn.Linear(H//2, 1), nn.Sigmoid())

    def forward(self, x, market_type):
        B = x.size(0)
        h = self.input_proj(x)
        h = torch.cat([self.market_emb(market_type).unsqueeze(1),
                        self.cls_token.expand(B, -1, -1), h], dim=1)
        h = self.backbone(self.pos_enc(h))
        c = h[:, 1]
        return {'direction': self.direction_head(c),
                'pred_prob': self.pred_head(c).squeeze(-1),
                'confidence': self.conf_head(c).squeeze(-1)}


# ── Load model
def load_model():
    from huggingface_hub import hf_hub_download
    model_path  = hf_hub_download(repo_id=HF_REPO, filename='omega_best.pt', token=os.environ.get('HF_TOKEN'))
    scaler_path = hf_hub_download(repo_id=HF_REPO, filename='scalers.pkl',   token=os.environ.get('HF_TOKEN'))
    ckpt = torch.load(model_path, map_location=device)
    feature_cols = ckpt['feature_cols']
    model = OmegaModel(num_features=len(feature_cols), **{
        k: CFG[k] for k in ['d_model','nhead','num_layers','dropout',
                              'num_market_types','num_classes']}).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    with open(scaler_path, 'rb') as f:
        scalers = pickle.load(f)
    val_acc = ckpt.get('val_acc', 0.0)
    print(f"✅ Model loaded | val_acc={val_acc:.2%}")
    return model, scalers, feature_cols, val_acc


# ── Data
def fetch_crypto(pair):
    symbol = pair.split('/')[0]
    r = requests.get(
        'https://min-api.cryptocompare.com/data/v2/histohour',
        params={'fsym': symbol, 'tsym': 'USD', 'limit': 100}, timeout=15).json()
    data = [c for c in r.get('Data', {}).get('Data', []) if c['volumefrom'] > 0]
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df['time'], unit='s')
    df = df[['open','high','low','close','volumefrom']].rename(
        columns={'volumefrom': 'volume'}).astype(float)
    df['market_type'] = MARKET_TYPES['crypto']
    df['symbol'] = pair
    return df


def fetch_stock(ticker):
    from datetime import timedelta
    start = (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d')
    end   = datetime.now().strftime('%Y-%m-%d')
    url = (f'https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/hour'
           f'/{start}/{end}?adjusted=true&sort=asc&limit=200&apikey={POLYGON_KEY}')
    all_results = []
    while url:
        r = requests.get(url, timeout=15).json()
        all_results.extend(r.get('results', []))
        url = r.get('next_url')
        if url: url += f'&apikey={POLYGON_KEY}'
        time.sleep(0.2)
    if not all_results: return pd.DataFrame()
    df = pd.DataFrame(all_results)
    df.index = pd.to_datetime(df['t'], unit='ms')
    df = df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'})
    df = df[['open','high','low','close','volume']].astype(float)
    df['market_type'] = MARKET_TYPES['stock']
    df['symbol'] = ticker
    return df


def add_indicators(df):
    df = df.copy()
    df.ta.ema(length=9, append=True);  df.ta.ema(length=21, append=True)
    df.ta.ema(length=50, append=True); df.ta.adx(append=True)
    df.ta.rsi(length=14, append=True); df.ta.macd(append=True)
    df.ta.stoch(append=True);          df.ta.cci(length=14, append=True)
    df.ta.roc(length=10, append=True); df.ta.mom(length=10, append=True)
    df.ta.atr(length=14, append=True); df.ta.bbands(length=20, append=True)
    df.ta.obv(append=True)
    df['body']       = (df['close'] - df['open']) / (df['open'] + 1e-9)
    df['upper_wick'] = (df['high'] - df[['open','close']].max(axis=1)) / (df['open'] + 1e-9)
    df['lower_wick'] = (df[['open','close']].min(axis=1) - df['low']) / (df['open'] + 1e-9)
    df['hl_range']   = (df['high'] - df['low']) / (df['open'] + 1e-9)
    df['volume']     = np.log1p(df['volume'])
    COLUMN_REMAP = {
    'BBL_20_2.0': 'BBL_20_2.0_2.0',
    'BBM_20_2.0': 'BBM_20_2.0_2.0',
    'BBU_20_2.0': 'BBU_20_2.0_2.0',
    'BBB_20_2.0': 'BBB_20_2.0_2.0',
    'BBP_20_2.0': 'BBP_20_2.0_2.0',
     }
    df.rename(columns=COLUMN_REMAP, inplace=True)
    df.dropna(inplace=True)
    return df


# ── Inference
def kelly_fraction(win_prob):
    if win_prob <= 0.5: return 0.0
    return float(np.clip((2 * win_prob - 1) * 0.5, 0, CFG['max_kelly']))


@torch.no_grad()
def get_signal(model, scalers, feature_cols, df, mtype):
    feat_df = df[[c for c in feature_cols if c in df.columns]].copy()
    for col in feat_df.columns:
        if col in scalers:
            feat_df[col] = scalers[col].transform(feat_df[[col]])
    for c in feature_cols:
        if c not in feat_df.columns:
            feat_df[c] = 0.0
    feat_df = feat_df[feature_cols]
    window = feat_df.iloc[-CFG['seq_len']:].values
    x  = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
    mt = torch.tensor([mtype], dtype=torch.long)
    out   = model(x, mt)
    probs = torch.softmax(out['direction'], dim=-1)[0].numpy()
    conf  = float(probs.max())
    cls   = int(probs.argmax())
    action = 'HOLD'
    if conf >= CFG['conf_threshold']:
        action = 'BUY' if cls == 1 else 'SELL'
    return {
        'direction': 'UP' if cls == 1 else 'DOWN',
        'prob_up': float(probs[1]), 'prob_down': float(probs[0]),
        'confidence': conf, 'action': action,
        'kelly_f': kelly_fraction(conf) if action != 'HOLD' else 0.0,
    }


# ── Outcome checker
def check_outcomes(current_prices):
    if not Path(ENTRY_LOG_FILE).exists():
        return []
    with open(ENTRY_LOG_FILE) as f:
        entries = json.load(f)

    now_ts = datetime.now(timezone.utc).timestamp()
    resolved, remaining = [], []

    for e in entries:
        age_minutes = (now_ts - e['ts']) / 60
        if age_minutes >= 55:
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
            print(f'  {icon} {r["symbol"]} {r["action"]} | entry={r["entry_price"]:.4f} exit={r["exit_price"]:.4f} | {r["correct"]} | P&L ${r["pnl"]:.3f}')

    with open(ENTRY_LOG_FILE, 'w') as f:
        json.dump(remaining, f)

    return resolved


# ── Discord notification
def send_discord(signals, resolved, now_utc, val_acc):
    action_icons = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '⏸️'}
    market_icons = {'CRYPTO': '🔴', 'STOCK': '📈'}

    lines = [
        f'📡 **OMEGA v3** | {now_utc.strftime("%Y-%m-%d %H:%M")} UTC | val_acc={val_acc:.2%}',
        '```',
        f'{"Mkt":<8} {"Symbol":<12} {"Action":<6} {"↑UP":>6} {"↓DN":>6} {"Conf":>6} {"Kelly":>6} {"$":>6} {"Price":>10}',
        '─' * 70,
    ]
    for s in signals:
        lines.append(
            f'{s["market"]:<8} {s["symbol"]:<12} '
            f'{action_icons[s["action"]]} {s["action"]:<4} '
            f'{s["prob_up"]:>5.1%} {s["prob_down"]:>5.1%} '
            f'{s["confidence"]:>5.1%} {s["kelly_f"]:>5.1%} '
            f'${s["alloc_usd"]:>4.2f} {s["price"]:>10.4f}'
        )
    lines.append('─' * 70)
    actionable = [s for s in signals if s['action'] != 'HOLD']
    lines.append(f'Actionable : {len(actionable)}/{len(signals)}')
    lines.append(f'Deploy     : ${sum(s["alloc_usd"] for s in actionable):.2f} / ${CFG["total_capital"]:.2f}')
    lines.append('```')

    if resolved:
        lines.append('**Outcomes from last hour:**')
        lines.append('```')
        for r in resolved:
            icon = '✅' if r['correct'] == 'WIN' else '❌'
            lines.append(
                f'{icon} {r["symbol"]} {r["action"]} | '
                f'{r["entry_price"]:.4f} → {r["exit_price"]:.4f} | '
                f'{r["correct"]} | P&L ${r["pnl"]:.3f}'
            )
        wins  = sum(1 for r in resolved if r['correct'] == 'WIN')
        total = len(resolved)
        lines.append(f'Win rate: {wins}/{total} ({wins/total:.0%})')
        lines.append('```')

    payload = {'content': '\n'.join(lines), 'username': 'OMEGA v3'}
    try:
        r = requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
        r.raise_for_status()
        print('✅ Discord notification sent')
    except Exception as e:
        print(f'⚠️ Discord failed: {e}')


# ── Main
def main():
    now_utc = datetime.now(timezone.utc)
    print(f'\n{"="*65}')
    print(f'  📡 OMEGA v3 | {now_utc.strftime("%Y-%m-%d %H:%M")} UTC')
    print(f'{"="*65}')

    model, scalers, feature_cols, val_acc = load_model()

    signals        = []
    current_prices = {}

    # Crypto
    for pair in CFG['crypto_pairs']:
        try:
            df = fetch_crypto(pair)
            if df.empty or len(df) < CFG['seq_len'] + 50: continue
            df = add_indicators(df)
            price = float(df['close'].iloc[-1])
            current_prices[pair] = price
            print(f'  Features present: {[c for c in feature_cols if c in df.columns]}')
            print(f'  Features missing: {[c for c in feature_cols if c not in df.columns]}')
            s = get_signal(model, scalers, feature_cols, df, MARKET_TYPES['crypto'])
            alloc = CFG['total_capital'] * CFG['crypto_alloc'] * s['kelly_f']
            signals.append({**s, 'market': 'CRYPTO', 'symbol': pair,
                             'price': price, 'alloc_usd': alloc,
                             'ts': now_utc.timestamp(),
                             'datetime_utc': now_utc.strftime('%Y-%m-%d %H:%M')})
        except Exception as e:
            print(f'  ⚠️ {pair}: {e}')

    # Stocks
    for ticker in CFG['stock_tickers']:
        try:
            df = fetch_stock(ticker)
            if df.empty or len(df) < CFG['seq_len'] + 50: continue
            df = add_indicators(df)
            price = float(df['close'].iloc[-1])
            current_prices[ticker] = price
            s = get_signal(model, scalers, feature_cols, df, MARKET_TYPES['stock'])
            alloc = CFG['total_capital'] * CFG['stock_alloc'] * s['kelly_f']
            signals.append({**s, 'market': 'STOCK', 'symbol': ticker,
                             'price': price, 'alloc_usd': alloc,
                             'ts': now_utc.timestamp(),
                             'datetime_utc': now_utc.strftime('%Y-%m-%d %H:%M')})
            time.sleep(15)
        except Exception as e:
            print(f'  ⚠️ {ticker}: {type(e).__name__}: {e}')

    # Check outcomes
    resolved = check_outcomes(current_prices)

    # Print table
    action_icons = {'BUY': '🟢', 'SELL': '🔴', 'HOLD': '⏸️ '}
    market_icons = {'CRYPTO': '🔴', 'STOCK': '📈'}
    print(f"\n{'Mkt':<8} {'Symbol':<12} {'Action':<6} {'↑UP':>6} {'↓DN':>6} {'Conf':>6} {'Kelly':>6} {'$':>6} {'Price':>10}")
    print('─' * 70)
    for s in signals:
        print(f"{market_icons[s['market']]} {s['market']:<6} {s['symbol']:<12} "
              f"{action_icons[s['action']]} {s['action']:<4} "
              f"{s['prob_up']:>5.1%} {s['prob_down']:>5.1%} "
              f"{s['confidence']:>5.1%} {s['kelly_f']:>5.1%} "
              f"${s['alloc_usd']:>4.2f} {s['price']:>10.4f}")
    print('─' * 70)

    actionable = [s for s in signals if s['action'] != 'HOLD']
    print(f'  Actionable: {len(actionable)}/{len(signals)}')
    print(f'  Deploy: ${sum(s["alloc_usd"] for s in actionable):.2f} / ${CFG["total_capital"]:.2f}')

    # Save entry prices
    existing = json.loads(Path(ENTRY_LOG_FILE).read_text()) if Path(ENTRY_LOG_FILE).exists() else []
    new_entries = [{'symbol': s['symbol'], 'action': s['action'],
                    'entry_price': s['price'], 'alloc_usd': s['alloc_usd'],
                    'confidence': s['confidence'], 'ts': s['ts'],
                    'datetime_utc': s['datetime_utc']}
                   for s in actionable]
    with open(ENTRY_LOG_FILE, 'w') as f:
        json.dump(existing + new_entries, f, indent=2)
    if new_entries:
        print(f'  📝 Logged {len(new_entries)} entry prices → outcomes checked next run')

    # Discord
    send_discord(signals, resolved, now_utc, val_acc)

    print(f'\n✅ Done | {now_utc.strftime("%Y-%m-%d %H:%M")} UTC')


if __name__ == '__main__':
    main()