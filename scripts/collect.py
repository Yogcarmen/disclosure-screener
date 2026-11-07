"""
Step 1: keyword search on EDGAR, download documents, keep repurchase passages.

Output: data/filings.parquet-like CSV (data/filings.csv), one row per filing
(accession number), with the relevant passages from all its documents.

    python scripts/collect.py --start 2024-01-01 --end 2024-06-30 --sample 400
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from screener.edgar import search, fetch_document  # noqa: E402
from screener.text import html_to_text, relevant_passages  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--query', default='share repurchase program')
    p.add_argument('--start', default='2024-01-01')
    p.add_argument('--end', default='2024-06-30')
    p.add_argument('--sample', type=int, default=400,
                   help='random sample of filings to keep (0 = all)')
    p.add_argument('--seed', type=int, default=7)
    args = p.parse_args()

    from screener.edgar import USER_AGENT
    if '@' not in USER_AGENT:
        sys.exit('Set SEC_USER_AGENT to "Your Name your@email" first '
                 '(SEC blocks document downloads without a contact email).')

    hits = search(args.query, args.start, args.end)
    print(f'{len(hits)} matching documents in {hits["adsh"].nunique()} filings')
    hits.to_csv(ROOT / 'data' / 'search_hits.csv', index=False)

    filings = hits['adsh'].drop_duplicates()
    if args.sample and len(filings) > args.sample:
        filings = filings.sample(args.sample, random_state=args.seed)
    docs = hits[hits['adsh'].isin(filings)]

    rows = []
    for i, (adsh, grp) in enumerate(docs.groupby('adsh', sort=False), 1):
        parts = []
        for _, d in grp.iterrows():
            try:
                parts.append(relevant_passages(html_to_text(fetch_document(d, ROOT / 'data' / 'raw'))))
            except Exception as e:  # keep going; record the failure
                print(f'  ! {adsh} {d["filename"]}: {e}')
        first = grp.iloc[0]
        rows.append({'adsh': adsh, 'company': first['company'], 'ticker': first['ticker'],
                     'cik': first['cik'], 'file_date': first['file_date'],
                     'items': first['items'], 'n_docs': len(grp),
                     'url': f"https://www.sec.gov/Archives/edgar/data/{first['cik']}/"
                            f"{adsh.replace('-', '')}/{first['filename']}",
                     'passages': '\n'.join(x for x in parts if x)[:4000]})
        if i % 50 == 0:
            print(f'  fetched {i}/{len(filings)} filings')

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / 'data' / 'filings.csv', index=False)
    print(f'Saved {len(out)} filings to data/filings.csv '
          f'({(out["passages"].str.len() == 0).sum()} with no passage found)')


if __name__ == '__main__':
    main()
