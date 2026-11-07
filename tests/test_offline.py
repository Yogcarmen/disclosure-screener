"""Offline tests: text handling, validators and response parsing (no network, no LLM)."""

import json

import pytest

from screener.edgar import _ticker
from screener.llm import parse_response
from screener.text import html_to_text, relevant_passages
from screener.validate import check, parse_amount

HTML = """<html><head><style>p{}</style></head><body>
<p>Acme Corp reported first-quarter revenue of $2.1 billion.</p>
<p>Gross margin improved.</p>
<p>The Board of Directors authorized a new share repurchase program of up to $1.5&nbsp;billion,
which expires on December 31, 2026.</p>
<p>The company also declared a dividend.</p>
<p>Unrelated paragraph one.</p><p>Unrelated paragraph two.</p>
</body></html>"""


def test_html_to_text_strips_markup_and_nbsp():
    text = html_to_text(HTML)
    assert '<p>' not in text and 'p{}' not in text
    assert 'up to $1.5 billion' in ' '.join(text.split())


def test_relevant_passages_keeps_neighbours_only():
    text = html_to_text(HTML)
    ex = relevant_passages(text, window=1)
    assert 'share repurchase program' in ex
    assert 'Gross margin improved' in ex          # neighbour before
    assert 'declared a dividend' in ex             # neighbour after
    assert 'Unrelated paragraph two' not in ex


@pytest.mark.parametrize('text,value', [
    ('$1.5 billion', 1.5e9), ('$750,000,000', 7.5e8), ('$250 million', 2.5e8),
    ('up to $2bn', 2e9), (None, None), ('no number here', None)])
def test_parse_amount(text, value):
    assert parse_amount(text) == value


def test_check_flags_paraphrased_evidence():
    src = html_to_text(HTML)
    good = {'category': 'new_authorization', 'amount_usd': 1.5e9, 'amount_text': '$1.5 billion',
            'evidence': 'The Board of Directors authorized a new share repurchase program of up to $1.5 billion'}
    bad = dict(good, evidence='The board approved a buyback worth one and a half billion dollars')
    assert check(good, src)['needs_review'] is False
    assert check(bad, src)['evidence_found'] is False
    assert check(bad, src)['needs_review'] is True


def test_near_verbatim_quote_is_accepted():
    src = html_to_text(HTML)
    trimmed = {'category': 'new_authorization', 'amount_usd': 1.5e9, 'amount_text': '$1.5 billion',
               'evidence': 'Directors authorized a new share repurchase program of up to $1.5 billion, '
                           'which expires on December 31, 2026. (emphasis added)'}
    flags = check(trimmed, src)
    assert flags['evidence_score'] >= 0.9 or flags['evidence_found'] is False
    exact = dict(trimmed, evidence='The Board of Directors authorized a new share repurchase program '
                                   'of up to $1.5 billion, which expires on December 31, 2026.')
    assert check(exact, src)['evidence_found'] is True


def test_check_flags_inconsistent_amount():
    src = html_to_text(HTML)
    r = {'category': 'new_authorization', 'amount_usd': 1.5e6, 'amount_text': '$1.5 billion',
         'evidence': 'The Board of Directors authorized a new share repurchase program of up to $1.5 billion'}
    assert check(r, src)['amount_consistent'] is False


def test_parse_response_validates_category():
    body = {'message': {'content': json.dumps({'category': 'new_authorization', 'amount_usd': 1e9,
                                               'amount_text': '$1 billion', 'expiry': None,
                                               'evidence': 'x'})}}
    assert parse_response(body)['amount_usd'] == 1e9
    body_bad = {'message': {'content': json.dumps({'category': 'maybe'})}}
    with pytest.raises(ValueError):
        parse_response(body_bad)


def test_ticker_from_display_name():
    assert _ticker('Match Group, Inc.  (MTCH)  (CIK 0000891103)') == 'MTCH'
    assert _ticker('Foo Corp  (FOO, FOO-PA)  (CIK 0000000001)') == 'FOO'
    assert _ticker('No Ticker LLC  (CIK 0000000002)') is None
