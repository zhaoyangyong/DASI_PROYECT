"""
message_normalizer.py
---------------------
Converts any incoming message — JSON or natural language — into a single
NormalizedMessage before business logic runs.

Pipeline (PDF §7 step 1 + §4 intent):
    1. JSON parse short-circuit. If the peer sends a structured message and
       it already has a recognised `kind`, return it directly.
    2. Regex fast path for the canonical Spanish trade formats this project
       uses ("Necesito X. Puedo ofrecer Y.", "Te ofrezco X a cambio de Y.",
       "Te envío X.", "¿Puedes enviarme X, como acordamos?"). Catches ~80%
       of messages in 1ms each and avoids paying for an Ollama round-trip
       plus the llama3.1 "no tool call" intermittent failure.
    3. Otherwise, send the raw text to Ollama's `classify_intent` tool and
       use the model's structured answer.
    4. If Ollama is unreachable, times out, or returns kind="unknown",
       fall back conservatively to kind="clarification" so the upper layer
       asks the peer to restate themselves rather than guess.

Intent classification is a language-understanding task and is delegated to
the LLM. Code remains authoritative on what we DO with the result — the
decision engine still enforces resource whitelists, inventory caps, and
target-resource protection downstream.
"""

import re

from loguru import logger

from models import NormalizedMessage
from ollama_client import call_ollama_tool
from prompt_builder import INTENT_TOOL, build_intent_prompt
from state_manager import state
from utils import VALID_RESOURCES, safe_json_parse


_VALID_KINDS = {
    "request", "delivery", "accept", "reject",
    "counter_offer", "clarification", "unknown",
}


# ---------------------------------------------------------------------------
# Regex fast path — canonical Spanish trade-message formats
# ---------------------------------------------------------------------------

_PLURAL_ALIASES: dict[str, str] = {
    "ladrillo":  "ladrillos",
    "maderas":   "madera",
    "piedras":   "piedra",
    "telas":     "tela",
    "quesos":    "queso",
    "vinos":     "vino",
    "aceites":   "aceite",
    "trigos":    "trigo",
    "arroces":   "arroz",
}

# "3 queso" / "5 de tela" / "2 unidades de oro"
_QTY_WORD = re.compile(
    r"(\d+)\s+(?:de\s+|unidades?\s+(?:de\s+)?)?([a-záéíóúñ]+)",
    re.IGNORECASE,
)

# "Necesito X. Puedo ofrecer Y" or "Necesito X. Ofrezco Y"
_NECESITO_OFREZCO = re.compile(
    r"necesito\s+(.+?)\.\s*(?:puedo\s+ofrecer|ofrezco)\s+(.+?)(?=[\.\?¿]|$)",
    re.IGNORECASE | re.DOTALL,
)

# "Te ofrezco X a cambio de Y" — also handles common synonyms peers (and
# llama-generated text) use: "Puedo ofrecer ... a cambio de ...",
# "Te doy ... a cambio de ...", and the bare "Ofrezco / Doy" variants.
_OFREZCO_A_CAMBIO = re.compile(
    r"(?:te\s+ofrezco|ofrezco|puedo\s+ofrecer|te\s+doy|doy)\s+"
    r"(.+?)\s+a\s+cambio\s+de\s+(.+?)(?=[\.\?¿]|$)",
    re.IGNORECASE | re.DOTALL,
)

# "Te envío X" / "Te mando X" / "Envío X"
_DELIVERY = re.compile(
    r"(?:te\s+env[íi]o|te\s+mando|env[íi]o|mando)\s+(.+?)(?=[\.\?¿]|$)",
    re.IGNORECASE | re.DOTALL,
)

# "¿Puedes enviarme X, como acordamos?" — peer asks us to ship resources from
# a deal we already agreed (the honour-claim leg of a barter). This is the
# EXACT shape decision_engine._generate_trade_message emits for its want-only
# branch, so our own outbound honour requests must round-trip through this
# fast path. Without it the message depends entirely on the intermittently
# unreliable llama3.1 intent tool; a misclassification routes it to
# clarification instead of process_request, and the trade's second leg never
# settles. The trailing "como acordamos" (if present) carries no quantity, so
# _extract_qty_map drops it harmlessly. Matches "enviarme" and "envíame".
_HONOUR_CLAIM = re.compile(
    r"(?:puedes\s+)?(?:envi[aá]rme|env[íi]ame)\s+(.+?)(?=[\.\?¿]|$)",
    re.IGNORECASE | re.DOTALL,
)

# Bare affirmation. Anchored at both ends so "ok te envío 2 piedras" does
# NOT match (it still falls through to _DELIVERY); only pure affirmations
# with optional trailing punctuation are caught here. When we have a
# pending offer to this peer, this is interpreted as acceptance.
_AFFIRMATIVE = re.compile(
    r"^\s*"
    r"(ok+|okay|vale|s[íi]|perfecto|acepto|aceptado|trato|"
    r"hecho|de\s+acuerdo|genial|claro|confirmado|conforme)"
    r"[\s\.\!,]*$",
    re.IGNORECASE,
)


def _canonical_resource(word: str) -> str | None:
    """Map a raw token to a canonical resource name, or None if unknown."""
    word = word.lower()
    word = _PLURAL_ALIASES.get(word, word)
    return word if word in VALID_RESOURCES else None


def _extract_qty_map(segment: str) -> dict[str, int]:
    """Pull all 'N resource' pairs from `segment`; drops unknown words."""
    result: dict[str, int] = {}
    for m in _QTY_WORD.finditer(segment):
        qty = int(m.group(1))
        resource = _canonical_resource(m.group(2))
        if resource is None or qty <= 0:
            continue
        result[resource] = qty
    return result


def _try_fast_parse(
    text: str,
    from_agent: str,
    has_pending: bool = False,
) -> NormalizedMessage | None:
    """
    Cheap regex match for the project's canonical formats. Returns None
    on any ambiguity so the LLM tool can take over.

    `has_pending` controls the bare-affirmation branch: "ok"/"vale"/"sí"
    alone is treated as kind="accept" only when we have a pending offer
    out to this peer, else as kind="clarification" (a bare ok with nothing
    on the table is not actionable on its own).
    """
    # Bare affirmation — highest priority. Anchored regex so longer text
    # like "ok te envío 2 piedras" does NOT hit this branch.
    if _AFFIRMATIVE.match(text):
        return NormalizedMessage(
            from_agent=from_agent,
            kind="accept" if has_pending else "clarification",
            resources={},
            offered_resources={},
            raw_text=text,
            metadata={"fast_path": "affirmative"},
        )

    # Counter — most specific (anchored on 'a cambio de')
    m = _OFREZCO_A_CAMBIO.search(text)
    if m:
        offered = _extract_qty_map(m.group(1))
        wanted = _extract_qty_map(m.group(2))
        if offered and wanted:
            return NormalizedMessage(
                from_agent=from_agent,
                kind="counter_offer",
                resources=wanted,
                offered_resources=offered,
                raw_text=text,
                metadata={"fast_path": "counter"},
            )

    # Standard proactive / request: "Necesito X. Puedo ofrecer Y."
    m = _NECESITO_OFREZCO.search(text)
    if m:
        wanted = _extract_qty_map(m.group(1))
        offered = _extract_qty_map(m.group(2))
        if wanted or offered:
            return NormalizedMessage(
                from_agent=from_agent,
                kind="request",
                resources=wanted,
                offered_resources=offered,
                raw_text=text,
                metadata={"fast_path": "request"},
            )

    # Delivery announcement
    m = _DELIVERY.search(text)
    if m:
        resources = _extract_qty_map(m.group(1))
        if resources:
            return NormalizedMessage(
                from_agent=from_agent,
                kind="delivery",
                resources=resources,
                offered_resources={},
                raw_text=text,
                metadata={"fast_path": "delivery"},
            )

    # Honour-claim — peer asks us to deliver on a previously agreed trade.
    # Classified as a plain `request` with no counter-offer attached: that is
    # exactly the shape process_request's honour-pending fast path expects
    # (requested subset of pending.we_give, empty offered_resources), so it
    # routes straight to _honour_pending_claim. Checked AFTER _DELIVERY so a
    # genuine "Te envío X" delivery is never misread as a request. Falls
    # through to the LLM if no concrete quantity is present (e.g. an open
    # "¿Puedes enviarme algo de oro?"), keeping ambiguous asks conservative.
    m = _HONOUR_CLAIM.search(text)
    if m:
        resources = _extract_qty_map(m.group(1))
        if resources:
            return NormalizedMessage(
                from_agent=from_agent,
                kind="request",
                resources=resources,
                offered_resources={},
                raw_text=text,
                metadata={"fast_path": "honour_claim"},
            )

    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _from_json(data: dict, raw_text: str) -> NormalizedMessage:
    """
    Build a NormalizedMessage from a dict — either a peer's JSON payload or
    the arguments returned by the `classify_intent` tool. Filters resources
    to non-negative integers and clamps unknown kinds to "unknown".
    """
    kind = data.get("kind", "unknown")
    if kind == "offer":
        kind = "request"
    if kind not in _VALID_KINDS:
        kind = "unknown"

    # Defensive: Ollama's tool-call schema is not strictly enforced, so the
    # model occasionally returns a STRING in place of the resources object
    # (e.g. `"resources": "lots of stuff"`), which would crash `.items()`.
    # Coerce any non-dict value to an empty dict before iterating.
    raw_resources = data.get("resources")
    if not isinstance(raw_resources, dict):
        raw_resources = {}
    raw_offered = data.get("offered_resources")
    if not isinstance(raw_offered, dict):
        raw_offered = {}

    # Quantities <=0 are dropped: llama3.1 sometimes returns {resource: 0}
    # when the source text only listed resource names without numbers (e.g.
    # "Necesito [trigo, madera]"). Those should be treated as "no info",
    # not as an explicit zero, so we strip them out here.
    resources: dict[str, int] = {}
    for k, v in raw_resources.items():
        if isinstance(v, (int, float)) and v > 0:
            resources[k] = int(v)

    offered_resources: dict[str, int] = {}
    for k, v in raw_offered.items():
        if isinstance(v, (int, float)) and v > 0:
            offered_resources[k] = int(v)

    # If after dropping zero/invalid quantities a request/counter has nothing
    # actionable left, downgrade to clarification so we ask the peer for
    # concrete numbers rather than silently rejecting an "empty" trade.
    if kind in ("request", "counter_offer") and not resources and not offered_resources:
        kind = "clarification"

    return NormalizedMessage(
        from_agent=str(data.get("from_agent", "unknown")),
        kind=kind,
        resources=resources,
        offered_resources=offered_resources,
        raw_text=raw_text,
        metadata=data.get("metadata") or {},
    )


def _clarification_fallback(raw: str, from_agent: str, reason: str) -> NormalizedMessage:
    """
    Conservative fallback when the intent tool cannot give us a confident
    answer. The upper layer will ask the peer to restate themselves.
    """
    logger.info(f"Falling back to 'clarification' ({reason}) for: {raw[:80]}")
    return NormalizedMessage(
        from_agent=from_agent,
        kind="clarification",
        resources={},
        offered_resources={},
        raw_text=raw,
        metadata={"fallback_reason": reason},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def normalize(raw: str, from_agent: str = "unknown") -> NormalizedMessage:
    raw = raw.strip()
    logger.debug(f"Normalizing: {raw[:120]}")

    pending = state.get_pending_offer(from_agent) if from_agent != "unknown" else None
    has_pending = pending is not None

    # --- 1. JSON short-circuit -------------------------------------------------
    # If the peer sent a well-formed structured message there is no need to
    # spend an Ollama round-trip understanding it. Only short-circuit when
    # the kind is already recognised; otherwise fall through to the LLM.
    data = safe_json_parse(raw)
    if data is not None:
        msg = _from_json(data, raw)
        if msg.from_agent == "unknown" and from_agent != "unknown":
            msg = msg.model_copy(update={"from_agent": from_agent})
        if msg.kind != "unknown":
            # A JSON-declared accept with no pending offer is suspicious;
            # PDF §5 says treat it as clarification.
            if msg.kind == "accept" and not has_pending:
                logger.info(
                    "JSON message claims 'accept' with no pending offer — "
                    "treating as clarification."
                )
                msg = msg.model_copy(update={"kind": "clarification"})
            logger.info(
                f"Normalized via JSON | kind={msg.kind} "
                f"resources={msg.resources} offered={msg.offered_resources}"
            )
            return msg
        logger.debug("JSON parsed but kind=unknown — delegating to intent tool.")

    # --- 2. Regex fast path for canonical Spanish formats ---------------------
    # Catches "Necesito X. Puedo ofrecer Y.", "Te ofrezco X a cambio de Y.",
    # "Te envío X.", and bare affirmations ("ok", "vale", "sí" …) in
    # microseconds, avoiding an Ollama round-trip. Any ambiguity falls
    # through to the LLM tool below. `has_pending` is forwarded so the
    # bare-ok branch can distinguish "accepting our pending offer" from
    # "stray ok with nothing on the table".
    fast = _try_fast_parse(raw, from_agent, has_pending=has_pending)
    if fast is not None:
        # Bare counter_offer with no pending offer to counter is suspicious —
        # treat the message as a fresh request so downstream evaluation runs.
        if fast.kind == "counter_offer" and not has_pending:
            fast = fast.model_copy(update={"kind": "request"})
        logger.info(
            f"Normalized via fast path ({fast.metadata.get('fast_path')}) | "
            f"kind={fast.kind} resources={fast.resources} "
            f"offered={fast.offered_resources}"
        )
        return fast

    # --- 3. LLM intent tool (primary path for natural language) ---------------
    try:
        history = state.get_history(from_agent) if from_agent != "unknown" else []
        prompt = build_intent_prompt(raw, history=history, pending=pending)
        ollama_data = await call_ollama_tool(prompt, INTENT_TOOL)
    except Exception as exc:
        logger.warning(f"Intent tool raised: {exc}")
        ollama_data = None

    if ollama_data is None:
        return _clarification_fallback(raw, from_agent, "ollama_unavailable")

    msg = _from_json(ollama_data, raw)
    if msg.from_agent == "unknown":
        msg = msg.model_copy(update={"from_agent": from_agent})

    # PDF §5 — demote stray accepts that have no pending offer to clarify.
    if msg.kind == "accept" and not has_pending:
        logger.info(
            "Intent tool returned 'accept' with no pending offer — "
            "treating as clarification."
        )
        msg = msg.model_copy(update={"kind": "clarification"})

    if msg.kind == "unknown":
        return _clarification_fallback(raw, from_agent, "intent_tool_unknown")

    logger.info(
        f"Normalized via intent tool | kind={msg.kind} "
        f"resources={msg.resources} offered={msg.offered_resources}"
    )
    return msg
