"""
Step 4: abnormal returns around the buyback announcements the model found.

Uses github.com/Yogcarmen/event-study-toolkit (market model vs SPY,
estimation days -250..-11, event window -5..+5).

Announcements that arrive together with earnings (8-K item 2.02) are
reported separately: the price reaction then mixes the buyback news with
the earnings surprise.

    python scripts/event_study.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from eventstudy import EventStudy, load_prices, to_returns  # noqa: E402

POSITIVE = ['new_authorization', 'increased_authorization', 'accelerated_share_repurchase']
MARKET = 'SPY'


def main():
    ext = pd.read_csv(ROOT / 'results' / 'extractions.csv')
    fil = pd.read_csv(ROOT / 'data' / 'filings.csv')
    df = ext.merge(fil[['adsh', 'ticker', 'file_date', 'items']], on='adsh')
    df = df[df['category'].isin(POSITIVE) & df['ticker'].notna()].copy()
    df['with_earnings'] = df['items'].fillna('').str.contains('2.02')
    df['file_date'] = pd.to_datetime(df['file_date'])
    print(f'{len(df)} flagged announcements with a ticker '
          f'({df["with_earnings"].sum()} filed together with earnings)')

    tickers = sorted(set(df['ticker'])) + [MARKET]
    cache = ROOT / 'data' / 'prices'
    prices = load_prices(tickers, start='2023-01-01', end='2024-08-31', cache_dir=cache)
    # Bulk downloads sometimes drop tickers (rate limits); retry those one by one.
    missing = [t for t in tickers if t not in prices or prices[t].isna().all()]
    for t in missing:
        try:
            one = load_prices([t], start='2023-01-01', end='2024-08-31', cache_dir=cache)
            if not one[t].isna().all():
                prices[t] = one[t]
        except Exception:
            pass
    still = [t for t in tickers if t not in prices or prices[t].isna().all()]
    print(f'No price history for {len(still)} tickers (mostly delisted/acquired since): {still}')
    rets = to_returns(prices)

    events = df.rename(columns={'ticker': 'asset', 'file_date': 'date'})
    groups = {'All flagged announcements': events,
              'Stand-alone (no earnings in same 8-K)': events[~events['with_earnings']],
              'Filed with earnings': events[events['with_earnings']]}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    tables = []
    for ax, (name, ev) in zip(axes, groups.items()):
        es = EventStudy(rets, ev[['asset', 'date']], market=MARKET,
                        estimation=(-250, -11), event=(-5, 5))
        t = es.summary(windows=[(0, 0), (-1, 1), (0, 5)])
        t.insert(0, 'sample', name)
        tables.append(t)
        es.plot(ax=ax, title=f'{name}')
        print(f'{name}: N={es.n_events}, skipped={len(es.skipped)}')

    fig.suptitle('Average CAR around share-repurchase announcements found by the screener '
                 '(8-K filings, H1 2024)', x=0.01, ha='left', fontsize=12)
    fig.tight_layout()
    out = ROOT / 'results'
    fig.savefig(out / 'buyback_car.png', dpi=150, bbox_inches='tight')
    summary = pd.concat(tables)
    summary.to_csv(out / 'summary_event_study.csv')
    pd.set_option('display.float_format', lambda v: f'{v:.4f}')
    print(summary.to_string())


if __name__ == '__main__':
    main()
