"""
utils.py
--------
Small dependency-light helpers shared across modules.

This file MUST NOT import from any business-logic module (state_manager,
decision_engine, message_normalizer, main). Those modules import from
here, never the other way around — that's what keeps the import graph
acyclic and the responsibilities clear.

Public surface:
    VALID_RESOURCES       — whitelist of resource names allowed in the game
    safe_json_parse(text) — parse text as a JSON dict, return None on error
    clean_qty_dict(d)     — coerce a {resource: qty} dict to positive ints
"""

from __future__ import annotations

import json
from typing import Any


# Canonical list of resources allowed in this trading game. Used as a
# whitelist when parsing LLM output or peer messages so we never trade
# invented resources like "wood" or "gold-bar".
VALID_RESOURCES: frozenset[str] = frozenset({
    "arroz",
    "ladrillos",
    "madera",
    "piedra",
    "queso",
    "tela",
    "trigo",
    "oro",
    "vino",
    "aceite",
})


def safe_json_parse(text: str | None) -> dict | None:
    """
    Parse `text` as JSON and return it only if it's a dict.

    Returns None on any failure — empty input, non-JSON text, JSON that
    isn't an object, etc. Designed for use in fall-back chains where the
    caller has a deterministic Plan B.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def clean_qty_dict(d: Any) -> dict[str, int]:
    """
    Return a clean {resource: positive_int_qty} dict.

    Filters out:
        - non-dict inputs (returns empty)
        - non-integer values
        - zero or negative quantities

    Does NOT validate resource names — keep that decision at the caller's
    boundary so this stays a pure structural cleaner.
    """
    if not isinstance(d, dict):
        return {}
    return {
        str(k): int(v)
        for k, v in d.items()
        if isinstance(v, int) and v > 0
    }
