"""
messaging.py
------------
Builders for the structured JSON messages this agent sends to peers.

Every outbound trade-related message shares the same shape:

    {
      "kind":              "request" | "delivery",
      "from_agent":        <our agent name>,
      "resources":         {<resource>: <qty>, ...},
      "offered_resources": {<resource>: <qty>, ...}   # request only
    }

Centralising the construction here guarantees peers always parse our
intent identically — there is exactly one place to change the wire format
if it ever evolves.
"""

from __future__ import annotations

import json

from config import AGENT_NAME


def build_structured_request(
    resources: dict[str, int],
    offered_resources: dict[str, int] | None = None,
) -> str:
    """
    Build a 'request' message.

    Args:
        resources:         resources WE want from the peer.
        offered_resources: resources we are willing to GIVE in exchange.
                           Use empty/None for a pure ask (no barter side).

    The returned string is JSON, ready to drop into a /buzon POST body.
    """
    return json.dumps({
        "kind": "request",
        "from_agent": AGENT_NAME,
        "resources": dict(resources),
        "offered_resources": dict(offered_resources or {}),
    })


def build_structured_delivery(resources: dict[str, int]) -> str:
    """
    Build a 'delivery' message — resources we are shipping to a peer.

    Note: actual resource transfers go through Butler; this message is
    primarily a human-readable notification that the package is on the
    way and aids dashboard logging on the peer side.
    """
    return json.dumps({
        "kind": "delivery",
        "from_agent": AGENT_NAME,
        "resources": dict(resources),
    })
