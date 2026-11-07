"""
Automatic checks on LLM output against the source text.

These do not prove an answer is right, but they catch the most common
failure modes cheaply:

* evidence_found  - the quoted sentence appears in the excerpt, verbatim or
                    nearly so (catches paraphrase / hallucinated quotes)
* amount_found    - the stated amount text appears in the excerpt
* amount_consistent - amount_usd agrees with amount_text after parsing
"""

from __future__ import annotations

import re

from screener.text import normalise

SCALE = {'thousand': 1e3, 'million': 1e6, 'billion': 1e9, 'trillion': 1e12,
         'k': 1e3, 'm': 1e6, 'mm': 1e6, 'bn': 1e9, 'b': 1e9}


def parse_amount(text: str | None) -> float | None:
    """'$1.5 billion' -> 1.5e9 ; '$750,000,000' -> 7.5e8 ; None if no number."""
    if not text:
        return None
    m = re.search(r'([\d][\d,]*(?:\.\d+)?)\s*(thousand|million|billion|trillion|bn|mm|m|b|k)?\b',
                  text.lower())
    if not m:
        return None
    value = float(m.group(1).replace(',', ''))
    return value * SCALE.get(m.group(2) or '', 1.0)


def _contains(source: str, quote: str | None, min_len: int = 20) -> bool:
    if not quote or len(quote.strip()) < min_len:
        return False
    return normalise(quote) in normalise(source)


def match_score(source: str, quote: str | None) -> float:
    """
    Share of the quote covered by its longest contiguous match in the source
    (after normalisation). 1.0 = verbatim; small edits at the ends (a dropped
    word, a joined clause, a table separator in the source) stay above 0.9,
    while paraphrases and stitched-together "quotes" fall well below.
    """
    if not quote or not quote.strip():
        return 0.0
    q, s = normalise(quote), normalise(source)
    if q in s:
        return 1.0
    from difflib import SequenceMatcher
    m = SequenceMatcher(None, q, s, autojunk=False).find_longest_match(0, len(q), 0, len(s))
    return m.size / len(q)


EVIDENCE_THRESHOLD = 0.9


def check(result: dict, source: str) -> dict:
    score = match_score(source, result.get('evidence'))
    flags = {
        'evidence_score': round(score, 3),
        'evidence_found': score >= EVIDENCE_THRESHOLD and len(str(result.get('evidence') or '')) >= 20,
        'amount_found': None,
        'amount_consistent': None,
    }
    if result.get('amount_text'):
        flags['amount_found'] = _contains(source, result['amount_text'], min_len=1)
        parsed = parse_amount(result['amount_text'])
        usd = result.get('amount_usd')
        if parsed is not None and usd:
            flags['amount_consistent'] = abs(parsed - usd) / max(usd, 1) < 0.01
    flags['needs_review'] = (not flags['evidence_found']
                             or flags['amount_found'] is False
                             or flags['amount_consistent'] is False)
    return flags
