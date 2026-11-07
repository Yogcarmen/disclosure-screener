"""
Step 3: build a stratified sample for human review, then score the model.

    python scripts/audit.py sample --n 60     # writes audit/audit_sample.csv
    # ...fill human_category / human_amount_usd in a spreadsheet...
    python scripts/audit.py score             # precision / recall / accuracy

Stratifying by predicted category makes sure the rare positive classes
(new / increased authorisations, ASRs) are reviewed, not just the majority
'existing_program_only' class. Scores are reported per stratum and
re-weighted to the full population.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from screener.llm import CATEGORIES  # noqa: E402

POSITIVE = {'new_authorization', 'increased_authorization', 'accelerated_share_repurchase'}
AUDIT = ROOT / 'audit' / 'audit_sample.csv'


def sample(n: int, seed: int):
    ext = pd.read_csv(ROOT / 'results' / 'extractions.csv')
    fil = pd.read_csv(ROOT / 'data' / 'filings.csv')
    df = ext.merge(fil[['adsh', 'company', 'ticker', 'file_date', 'url', 'passages']], on='adsh')
    df = df[df['category'].notna()]
    per = max(1, n // df['category'].nunique())
    parts = [g.sample(min(len(g), per), random_state=seed) for _, g in df.groupby('category')]
    out = pd.concat(parts)
    out = out.sample(frac=1, random_state=seed)            # shuffle so reviewers are not primed
    out['stratum_size'] = out['category'].map(df['category'].value_counts())
    out['human_category'] = ''
    out['human_amount_usd'] = ''
    out['notes'] = ''
    cols = ['adsh', 'company', 'ticker', 'file_date', 'url', 'passages',
            'category', 'amount_usd', 'amount_text', 'evidence', 'needs_review',
            'stratum_size', 'human_category', 'human_amount_usd', 'notes']
    AUDIT.parent.mkdir(exist_ok=True)
    out[cols].to_csv(AUDIT, index=False)
    print(f'Wrote {len(out)} rows to {AUDIT} '
          f'({out["category"].value_counts().to_dict()})')
    print('Allowed human_category values:', ', '.join(CATEGORIES))


def score():
    a = pd.read_csv(AUDIT)
    a = a[a['human_category'].fillna('').str.strip() != '']
    if a.empty:
        print('No reviewed rows yet.'); return
    a['correct'] = a['category'] == a['human_category']
    a['pred_pos'] = a['category'].isin(POSITIVE)
    a['true_pos'] = a['human_category'].isin(POSITIVE)

    by = a.groupby('category').agg(reviewed=('correct', 'size'), accuracy=('correct', 'mean'),
                                   stratum_size=('stratum_size', 'first'))
    print('Per predicted category (accuracy = share the reviewer agreed with):')
    print(by.to_string(float_format=lambda v: f'{v:.2f}'))

    w = by['stratum_size'] / by['stratum_size'].sum()
    print(f"\nPopulation-weighted accuracy: {(by['accuracy'] * w).sum():.1%}")

    # Precision of the 'announcement' flag: among predicted positives, how many are true positives
    pp = a[a['pred_pos']]
    if len(pp):
        print(f"Precision of announcement flag: {pp['true_pos'].mean():.1%} (n={len(pp)})")
    # Recall estimate: weight each stratum's true-positive share by its size
    tp_share = a.groupby('category')['true_pos'].mean()
    est_true_pos = (tp_share * by['stratum_size']).fillna(0)
    found = est_true_pos[est_true_pos.index.isin(POSITIVE)].sum()
    if est_true_pos.sum() > 0:
        print(f"Estimated recall of announcement flag: {found / est_true_pos.sum():.1%}")

    amt = a[a['true_pos'] & a['pred_pos'] & a['human_amount_usd'].notna()]
    if len(amt):
        ok = np.isclose(amt['amount_usd'].astype(float), amt['human_amount_usd'].astype(float), rtol=0.01)
        print(f'Amount exact (within 1%) on correctly flagged announcements: {ok.mean():.1%} (n={len(amt)})')
    if 'needs_review' in a:
        flagged = a[a['needs_review'] == True]  # noqa: E712
        if len(flagged):
            print(f"Validator-flagged rows that were actually wrong: {(~flagged['correct']).mean():.1%} "
                  f'(n={len(flagged)})')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sample'); s.add_argument('--n', type=int, default=60)
    s.add_argument('--seed', type=int, default=11)
    sub.add_parser('score')
    args = p.parse_args()
    sample(args.n, args.seed) if args.cmd == 'sample' else score()


if __name__ == '__main__':
    main()
