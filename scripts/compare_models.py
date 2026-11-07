"""
Compare local models on the same filings.

Reads every results/extractions*.csv (one file per model) and reports:
* category mix, validator failure rate, errors and speed per model
* pairwise agreement and Cohen's kappa on the 5-way category and on the
  binary "is this a repurchase announcement?" flag
* filings where the models disagree (the most useful ones to audit first)
* accuracy against human labels, once audit/audit_sample.csv is filled in

    python scripts/compare_models.py
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

POSITIVE = {'new_authorization', 'increased_authorization', 'accelerated_share_repurchase'}


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a) | set(b))
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def load_models() -> dict[str, pd.DataFrame]:
    out = {}
    for path in sorted((ROOT / 'results').glob('extractions*.csv')):
        df = pd.read_csv(path)
        if df.empty:
            continue
        df['category'] = df['category'].fillna('error')   # invalid / empty JSON from the model
        out[str(df['model'].iloc[0])] = df.set_index('adsh')
    return out


def main():
    models = load_models()
    names = list(models)
    common = sorted(set.intersection(*(set(m.index) for m in models.values())))
    print(f'Models: {names}; filings scored by all: {len(common)}')

    rows = []
    for n, m in models.items():
        m = m.loc[common]
        rows.append({'model': n,
                     'sec_per_filing': m['seconds'].mean(),
                     'errors': int(m['error'].notna().sum() - (m['error'] == 'no passage').sum()),
                     'needs_review_%': 100 * m['needs_review'].fillna(True).astype(bool).mean(),
                     'flagged_announcements': int(m['category'].isin(POSITIVE).sum()),
                     **{f'n_{c}': int((m['category'] == c).sum())
                        for c in ['new_authorization', 'increased_authorization',
                                  'accelerated_share_repurchase', 'existing_program_only',
                                  'not_about_buyback']}})
    per_model = pd.DataFrame(rows).set_index('model')
    print('\nPer model:\n' + per_model.round(2).to_string())

    pairs = []
    for a, b in combinations(names, 2):
        ca, cb = models[a].loc[common, 'category'], models[b].loc[common, 'category']
        pa, pb = ca.isin(POSITIVE), cb.isin(POSITIVE)
        pairs.append({'pair': f'{a} vs {b}',
                      'category_agree_%': 100 * (ca == cb).mean(),
                      'category_kappa': cohen_kappa(ca, cb),
                      'announcement_agree_%': 100 * (pa == pb).mean(),
                      'announcement_kappa': cohen_kappa(pa, pb)})
    pairwise = pd.DataFrame(pairs).set_index('pair')
    print('\nPairwise agreement:\n' + pairwise.round(3).to_string())

    cats = pd.DataFrame({n: models[n].loc[common, 'category'] for n in names})
    unanimous = cats.nunique(axis=1) == 1
    print(f'\nAll models agree on {unanimous.mean():.1%} of filings '
          f'({(~unanimous).sum()} disagreements)')
    cats[~unanimous].to_csv(ROOT / 'results' / 'model_disagreements.csv')

    # Accuracy against human labels, if any
    audit_path = ROOT / 'audit' / 'audit_sample.csv'
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        audit = audit[audit['human_category'].fillna('').str.strip() != ''].set_index('adsh')
        if len(audit):
            acc = {n: (models[n].loc[audit.index, 'category'] == audit['human_category']).mean()
                   for n in names if set(audit.index) <= set(models[n].index)}
            print('\nAccuracy vs human labels (audit sample, not re-weighted):')
            for n, v in acc.items():
                print(f'  {n}: {v:.1%} (n={len(audit)})')
            per_model['audit_accuracy_%'] = pd.Series(acc) * 100

    per_model.to_csv(ROOT / 'results' / 'summary_models.csv')
    pairwise.to_csv(ROOT / 'results' / 'summary_model_agreement.csv')
    plot(per_model, pairwise, names, models, common)


def plot(per_model, pairwise, names, models, common):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={'width_ratios': [1.2, 1]})

    ax = axes[0]
    x = np.arange(len(names))
    ax.bar(x - 0.2, per_model['sec_per_filing'], 0.4, color='#4C72B0', label='Seconds per filing')
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, per_model['needs_review_%'], 0.4, color='#DD8452', label='Failed checks (%)')
    ax.set_xticks(x, names, fontsize=9)
    ax.set_ylabel('Seconds per filing')
    ax2.set_ylabel('Outputs failing automatic checks (%)')
    ax.set_title('Speed and self-consistency by model', loc='left', fontsize=11)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc='upper center',
              bbox_to_anchor=(0.5, -0.1), ncol=2)
    for s in ('top',):
        ax.spines[s].set_visible(False); ax2.spines[s].set_visible(False)

    ax = axes[1]
    k = len(names)
    mat = np.eye(k)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i != j:
                mat[i, j] = (models[a].loc[common, 'category'] == models[b].loc[common, 'category']).mean()
    im = ax.imshow(mat, vmin=0.5, vmax=1, cmap='Blues')
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f'{mat[i, j]:.0%}', ha='center', va='center',
                    color='white' if mat[i, j] > 0.8 else 'black', fontsize=10)
    ax.set_xticks(range(k), names, fontsize=8, rotation=20)
    ax.set_yticks(range(k), names, fontsize=8)
    ax.set_title('Agreement on 5-way category', loc='left', fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(ROOT / 'assets' / 'model_comparison.png', dpi=150, bbox_inches='tight')
    print('Saved assets/model_comparison.png')


if __name__ == '__main__':
    main()
