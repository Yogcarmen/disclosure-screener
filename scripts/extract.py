"""
Step 2: run the local LLM on every filing's passages and validate the output.

Output: results/extractions.csv (resumable: already-processed filings are
skipped on re-run).

    python scripts/extract.py --model qwen2.5:7b-instruct
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from screener.llm import extract, DEFAULT_MODEL  # noqa: E402
from screener.validate import check  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--revalidate', action='store_true',
                   help='recompute validation flags for existing results (no LLM calls)')
    p.add_argument('--out', default=str(ROOT / 'results' / 'extractions.csv'),
                   help='output CSV (use one file per model when comparing models)')
    args = p.parse_args()
    out_path = Path(args.out)

    if args.revalidate:
        res = pd.read_csv(out_path)
        src = pd.read_csv(ROOT / 'data' / 'filings.csv').set_index('adsh')['passages'].fillna('')
        clean = res.astype(object).where(res.notna(), None)
        flags = pd.DataFrame([check(r.to_dict(), src[r['adsh']]) for _, r in clean.iterrows()])
        res = pd.concat([res.drop(columns=[c for c in flags.columns if c in res]), flags], axis=1)
        res.to_csv(out_path, index=False)
        print(f"needs_review: {res['needs_review'].sum()} / {len(res)}")
        return

    filings = pd.read_csv(ROOT / 'data' / 'filings.csv').fillna({'passages': ''})
    out_path.parent.mkdir(exist_ok=True)
    done = set(pd.read_csv(out_path)['adsh']) if out_path.exists() else set()
    todo = filings[~filings['adsh'].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f'{len(done)} already done, {len(todo)} to process with {args.model}')

    t_start = time.time()
    for i, (_, f) in enumerate(todo.iterrows(), 1):
        t0 = time.time()
        if not f['passages'].strip():
            res = {'category': 'not_about_buyback', 'amount_usd': None, 'amount_text': None,
                   'expiry': None, 'evidence': '', 'error': 'no passage'}
        else:
            try:
                res = extract(f['passages'], model=args.model)
                res['error'] = None
            except Exception as e:
                res = {'category': None, 'amount_usd': None, 'amount_text': None,
                       'expiry': None, 'evidence': '', 'error': str(e)[:200]}
        flags = check(res, f['passages']) if res.get('category') else {}
        row = {'adsh': f['adsh'], 'model': args.model, 'seconds': round(time.time() - t0, 2),
               **res, **flags}
        pd.DataFrame([row]).to_csv(out_path, mode='a', header=not out_path.exists(), index=False)
        if i % 25 == 0:
            rate = (time.time() - t_start) / i
            print(f'  {i}/{len(todo)}  ({rate:.1f}s per filing)')

    res = pd.read_csv(out_path)
    print(res['category'].value_counts(dropna=False).to_string())
    print(f"needs_review: {res['needs_review'].fillna(True).astype(bool).sum()} / {len(res)}")


if __name__ == '__main__':
    main()
