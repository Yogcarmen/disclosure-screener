"""
Structured extraction with a local LLM served by Ollama.

The model must return JSON that matches ``SCHEMA`` (Ollama structured
outputs), at temperature 0 so that runs are repeatable.
"""

from __future__ import annotations

import json

import requests

OLLAMA_URL = 'http://localhost:11434/api/chat'
DEFAULT_MODEL = 'qwen2.5:7b-instruct'

CATEGORIES = ['new_authorization', 'increased_authorization',
              'accelerated_share_repurchase', 'existing_program_only', 'not_about_buyback']

SCHEMA = {
    'type': 'object',
    'properties': {
        'category': {'type': 'string', 'enum': CATEGORIES},
        'amount_usd': {'type': ['number', 'null']},
        'amount_text': {'type': ['string', 'null']},
        'expiry': {'type': ['string', 'null']},
        'evidence': {'type': 'string'},
    },
    'required': ['category', 'amount_usd', 'amount_text', 'expiry', 'evidence'],
}

SYSTEM = """You read excerpts from US company filings (SEC Form 8-K and press releases).
Decide whether the excerpt ANNOUNCES a share repurchase decision made now, and extract its terms.

category:
- new_authorization: the board approved a new repurchase program/authorization.
- increased_authorization: an existing authorization was increased, extended or replenished.
- accelerated_share_repurchase: the company entered into an accelerated share repurchase (ASR) agreement.
- existing_program_only: buybacks are only mentioned as past activity or remaining capacity under an existing program (common in earnings releases).
- not_about_buyback: repurchase wording refers to something else (e.g. notes, preferred stock tender, repurchase agreements/repos) or is boilerplate.

amount_usd: the dollar size of the NEW or ADDITIONAL authorization as a number in US dollars
(e.g. "$1.5 billion" -> 1500000000). null if not stated or category is existing_program_only/not_about_buyback.
amount_text: the amount exactly as written in the excerpt, or null.
expiry: expiry date or term exactly as written, or null.
evidence: copy ONE sentence verbatim from the excerpt that supports the category. Do not paraphrase."""


def extract(passages: str, model: str = DEFAULT_MODEL, timeout: int = 180) -> dict:
    """Call Ollama and return the parsed JSON (raises on HTTP/parse errors)."""
    payload = {
        'model': model,
        'messages': [{'role': 'system', 'content': SYSTEM},
                     {'role': 'user', 'content': f'Excerpt:\n"""\n{passages}\n"""'}],
        'format': SCHEMA,
        'stream': False,
        'options': {'temperature': 0, 'num_ctx': 4096},
    }
    if model.startswith(('qwen3', 'deepseek-r1')):
        payload['think'] = False   # reasoning models: answer directly, comparable to the others
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return parse_response(resp.json())


def parse_response(body: dict) -> dict:
    """Parse Ollama's chat response body into a clean result dict."""
    out = json.loads(body['message']['content'])
    if out.get('category') not in CATEGORIES:
        raise ValueError(f"unexpected category: {out.get('category')!r}")
    for key in ('amount_usd', 'amount_text', 'expiry', 'evidence'):
        out.setdefault(key, None)
    return out
