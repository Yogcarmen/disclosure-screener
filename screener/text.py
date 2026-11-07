"""HTML to clean text, and selection of the passages worth sending to the LLM."""

from __future__ import annotations

import re

from bs4 import BeautifulSoup

KEYWORDS = re.compile(r'repurchas|buy[\s-]?back|accelerated share', re.I)


def html_to_text(html: str) -> str:
    """Visible text with one paragraph per line and collapsed whitespace."""
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup(['script', 'style', 'head']):
        tag.decompose()
    # Mark block boundaries, then treat every other line break in the source
    # HTML as ordinary whitespace, so one paragraph stays on one line.
    marker = '\x00PARA\x00'   # must not be whitespace, or the collapse below eats it
    for tag in soup.find_all(['br', 'p', 'div', 'tr', 'li', 'h1', 'h2', 'h3', 'h4', 'table']):
        tag.append(marker)
    text = soup.get_text(' ').replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text)
    lines = [ln.strip() for ln in text.split(marker)]
    return '\n'.join(ln for ln in lines if ln)


def relevant_passages(text: str, window: int = 1, max_chars: int = 4000) -> str:
    """
    Paragraphs mentioning a repurchase keyword, plus ``window`` neighbouring
    paragraphs on each side, in document order, capped at ``max_chars``.

    Sending only these passages keeps prompts short (fast on a laptop GPU) at
    the cost of occasionally missing context far from the keyword.
    """
    paras = text.split('\n')
    keep = set()
    for i, p in enumerate(paras):
        if KEYWORDS.search(p):
            keep.update(range(max(0, i - window), min(len(paras), i + window + 1)))
    out = '\n'.join(paras[i] for i in sorted(keep))
    return out[:max_chars]


def normalise(s: str) -> str:
    """Lower-case, unify quotes/dashes and collapse whitespace (for matching)."""
    s = s.lower().replace('’', "'").replace('“', '"').replace('”', '"')
    s = s.replace('–', '-').replace('—', '-')
    return re.sub(r'\s+', ' ', s).strip()
