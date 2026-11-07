"""
Thin client for SEC EDGAR full-text search and filing documents.

SEC fair-access rules: identify yourself with a User-Agent that includes a
contact email, and stay well under 10 requests per second. Document downloads
from www.sec.gov return 403 without an email. Set it in your shell, not in
code, so it never ends up in the repository:

    PowerShell:  $env:SEC_USER_AGENT = "Your Name your@email.com"
    bash:        export SEC_USER_AGENT="Your Name your@email.com"
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

SEARCH_URL = 'https://efts.sec.gov/LATEST/search-index'
ARCHIVE_URL = 'https://www.sec.gov/Archives/edgar/data/{cik}/{adsh}/{filename}'
USER_AGENT = os.environ.get('SEC_USER_AGENT', 'disclosure-screener research project')
PAUSE = 0.25  # seconds between requests (< 10 req/s)

_session = requests.Session()
_session.headers.update({'User-Agent': USER_AGENT, 'Accept-Encoding': 'gzip, deflate'})


def _get(url: str, params: dict | None = None, retries: int = 3) -> requests.Response:
    for attempt in range(retries):
        resp = _session.get(url, params=params, timeout=30)
        time.sleep(PAUSE)
        if resp.status_code == 200:
            return resp
        if resp.status_code in (429, 500, 502, 503):
            time.sleep(2 ** attempt * 2)
            continue
        resp.raise_for_status()
    resp.raise_for_status()
    return resp


def _ticker(display_name: str) -> str | None:
    """'Match Group, Inc.  (MTCH)  (CIK 0000891103)' -> 'MTCH' (first ticker)."""
    m = re.search(r'\(([A-Z][A-Z0-9.\-]*(?:, [A-Z0-9.\-]+)*)\)\s+\(CIK', display_name)
    return m.group(1).split(',')[0].strip() if m else None


def search(query: str, start: str, end: str, forms: str = '8-K',
           max_hits: int | None = None) -> pd.DataFrame:
    """
    Full-text search. Returns one row per matching document with filing
    metadata (company, ticker, CIK, accession number, file date, 8-K items).
    """
    rows, offset = [], 0
    while True:
        params = {'q': f'"{query}"', 'forms': forms, 'dateRange': 'custom',
                  'startdt': start, 'enddt': end, 'from': offset}
        data = _get(SEARCH_URL, params).json()
        hits = data['hits']['hits']
        if not hits:
            break
        for h in hits:
            s = h['_source']
            adsh, filename = h['_id'].split(':', 1)
            name = s['display_names'][0] if s.get('display_names') else ''
            rows.append({
                'doc_id': h['_id'], 'adsh': adsh, 'filename': filename,
                'cik': int(s['ciks'][0]), 'company': re.sub(r'\s+\(.*$', '', name),
                'ticker': _ticker(name), 'form': s.get('form'),
                'file_type': s.get('file_type'), 'file_date': s.get('file_date'),
                'items': ','.join(s.get('items') or []),
            })
        offset += len(hits)
        total = data['hits']['total']['value']
        if offset >= total or (max_hits and offset >= max_hits):
            break
    df = pd.DataFrame(rows)
    return df.head(max_hits) if max_hits else df


def fetch_document(row, cache_dir: str | Path = 'data/raw') -> str:
    """Download one filing document (HTML/text), cached on disk."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{row['adsh']}__{row['filename']}"
    if path.exists():
        return path.read_text(encoding='utf-8', errors='ignore')
    url = ARCHIVE_URL.format(cik=int(row['cik']), adsh=row['adsh'].replace('-', ''),
                             filename=row['filename'])
    text = _get(url).text
    path.write_text(text, encoding='utf-8')
    return text
