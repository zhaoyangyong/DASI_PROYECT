"""
prompt_builder.py
-----------------
All prompts and tool schemas sent to Ollama are constructed here.

Design philosophy (PDF "Prompt 设计思想"):
    The LLM is NOT a chatbot — it is a *win-oriented trade decision maker*.
    Every prompt is split by task so the model only ever has to decide ONE
    thing at a time. Each task is bound to a single function-calling tool
    that enforces the output schema:

        - build_intent_prompt          + INTENT_TOOL          → classify intent
        - build_evaluate_prompt        + EVALUATE_TOOL        → trade decision
        - build_counter_offer_prompt   + COUNTER_OFFER_TOOL   → re-price barter
        - build_clarification_prompt   + CLARIFICATION_TOOL   → ask peer to clarify
        - build_trade_message_prompt   + TRADE_MESSAGE_TOOL   → Spanish sentence

    Each prompt states the resource gap, the surplus, conversation context,
    and the action menu. The model's only job is to call its tool with a
    legal choice — code enforces every business constraint afterwards.
"""

from __future__ import annotations

from typing import Iterable

from models import ConversationTurn
from utils import VALID_RESOURCES


_VALID_RESOURCES_ENUM = sorted(VALID_RESOURCES)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _format_history(history: Iterable[ConversationTurn], limit: int = 6) -> str:
    """Render the last `limit` dialogue turns in a compact form."""
    turns = list(history)[-limit:]
    if not turns:
        return "(no prior turns)"
    lines = []
    for t in turns:
        speaker = "PEER" if t.role == "peer" else "ME"
        kind = f"[{t.kind}]" if t.kind else ""
        lines.append(f"{speaker}{kind}: {t.text}")
    return "\n".join(lines)


def _format_pending(pending) -> str:
    if pending is None:
        return "(no pending offer)"
    return (
        f"we_give={pending.we_give} | we_want={pending.we_want}"
    )


def _state_block(state_snapshot: dict) -> str:
    """
    The state block embodies the bottom-line judgment dimensions from PDF §2:
    what I have, what I need, what I lack, what I can spare, how close to win.
    """
    inv = state_snapshot["inventory"]
    needs = state_snapshot["goal_needs"]
    surplus = state_snapshot["surplus"]
    goal_targets = state_snapshot["initial_goal"]
    progress = state_snapshot["progress"]
    near_victory = state_snapshot["near_victory"]
    chain_opps = state_snapshot.get("chain_opportunities", {})
    chain_line = ""
    if chain_opps:
        chain_line = f"\nChain trades available: {chain_opps}  ← accept these intermediates to re-trade for goal resources"
    return (
        f"Inventory (have):       {inv}\n"
        f"Goal gap (still need):  {needs}\n"
        f"Goal targets (must hold ≥ these qty at end): {goal_targets}\n"
        f"Surplus (safe to give): {surplus}\n"
        f"Progress to victory:    {progress:.2f}  near_victory={near_victory}"
        f"{chain_line}"
    )


def _qty_dict_schema(description: str) -> dict:
    """JSON schema for a {resource: positive_int_qty} mapping."""
    return {
        "type": "object",
        "description": description,
        "propertyNames": {"enum": _VALID_RESOURCES_ENUM},
        "additionalProperties": {"type": "integer", "minimum": 0},
    }


# ---------------------------------------------------------------------------
# 1. Intent recognition  (PDF §4 — intent classification)
# ---------------------------------------------------------------------------

def build_intent_prompt(
    raw_text: str,
    history: Iterable[ConversationTurn] = (),
    pending=None,
) -> str:
    """
    Classify what the peer is doing. The LLM does NOT decide a trade here.
    """
    return f"""You are an intent classifier for a competitive resource-trading
agent. Your single job is to label the peer's latest message and extract any
numeric resources mentioned — you do NOT decide whether to trade.

=== RECENT DIALOGUE ===
{_format_history(history)}

=== PENDING OFFER WE SENT ===
{_format_pending(pending)}

=== LATEST PEER MESSAGE ===
"{raw_text}"

=== INTENT KINDS ===
request        — peer asks us for resources (maybe offering some back)
delivery       — peer says they are sending us resources right now
accept         — peer accepts a pending offer (only meaningful if one exists)
reject         — peer declines a pending offer
counter_offer  — peer proposes different quantities/resources than our pending
clarification  — peer asks a question or the message is too vague to act on
unknown        — none of the above

=== HOW TO CLASSIFY — WORKED EXAMPLES ===

Example 1 — explicit barter request
  pending: (no pending offer)
  text: "Quiero 2 trigo, te ofrezco 1 vino a cambio."
  → classify_intent(kind="request", resources={{"trigo":2}},
                    offered_resources={{"vino":1}},
                    confidence="high", reason="Clear request with barter terms.")

Example 2 — bare 'ok' with no pending offer
  pending: (no pending offer)
  text: "vale"
  → classify_intent(kind="clarification", resources={{}},
                    offered_resources={{}},
                    confidence="medium",
                    reason="Bare affirmation but nothing pending to accept.")

Example 3 — counter to a pending offer
  pending: we_give={{madera:2}} | we_want={{oro:1}}
  text: "Mejor 3 madera por 1 oro."
  → classify_intent(kind="counter_offer", resources={{"madera":3}},
                    offered_resources={{"oro":1}},
                    confidence="high",
                    reason="Peer changes the quantities of the pending barter.")

Example 4 — delivery announcement
  text: "Te envío 2 piedras ahora."
  → classify_intent(kind="delivery", resources={{"piedra":2}},
                    offered_resources={{}},
                    confidence="high", reason="Peer is sending resources now.")

Example 5 — open question
  text: "¿Tienes oro?"
  → classify_intent(kind="clarification", resources={{}},
                    offered_resources={{}},
                    confidence="medium",
                    reason="Peer is asking, not proposing terms.")

Example 6 — resource names listed without quantities
  text: "Necesito [trigo, madera, piedra]. Puedo ofrecer [aceite, tela, oro]. ¿Qué propones?"
  → classify_intent(kind="clarification", resources={{}},
                    offered_resources={{}},
                    confidence="medium",
                    reason="Resources are mentioned but no quantities given.")
  IMPORTANT: When the text lists resource names without numbers, leave the
  dicts EMPTY. Do NOT fill quantities with 0 — an empty dict means "unknown",
  while {{resource: 0}} would falsely claim the peer offered zero of it.

Example 7 — claim on an already-agreed trade
  pending: we_give={{piedra:2}} | we_want={{}}
  text: "¿Puedes enviarme 2 piedra, como acordamos?"
  → classify_intent(kind="request", resources={{"piedra":2}},
                    offered_resources={{}},
                    confidence="high",
                    reason="Peer claims resources from a trade already agreed.")
  IMPORTANT: A question phrased as "¿Puedes enviarme X...?" — especially with
  "como acordamos" — is a REQUEST for X, not a clarification. It carries a
  concrete quantity, so the decision engine can act on it directly.

=== FIELD MEANINGS ===
kind              — one of the seven intent labels above
resources         — what peer wants FROM us, keyed by resource name
offered_resources — what peer offers TO us, keyed by resource name
confidence        — high / medium / low
reason            — one short sentence justifying the label

When in doubt between a confident label and clarification, choose
"clarification" — the decision engine can always re-prompt the peer.

Call the `classify_intent` tool with the result.
"""


# ---------------------------------------------------------------------------
# 2. Trade evaluation  (PDF §3 + §4 — does this move us closer to winning?)
# ---------------------------------------------------------------------------

def build_evaluate_prompt(
    state_snapshot: dict,
    requested: dict[str, int],
    offered_to_us: dict[str, int],
    exchangeable: dict[str, int],
    peer: str,
    history: Iterable[ConversationTurn] = (),
    pending=None,
) -> str:
    """
    Decide whether the current request advances victory.
    The LLM may only emit one of {accept, offer, counter, reject, clarify}.
    """
    return f"""You are a win-oriented trade decision agent for a competitive
resource-trading game. Every choice is judged by ONE question:
    "Does this step shorten my distance to the victory goal?"

=== MY STATE ===
{_state_block(state_snapshot)}

=== PEER ===
peer={peer}

=== RECENT DIALOGUE ===
{_format_history(history)}

=== PENDING OFFER WE PREVIOUSLY SENT TO THIS PEER ===
{_format_pending(pending)}

=== THIS REQUEST ===
They want from us:   {requested}
They offer to us:    {offered_to_us}
exchangeable (caps I may give): {exchangeable}

=== HARD CONSTRAINT ===
Never let your inventory of any "Goal targets" resource drop below its
required quantity. Trade only what is shown in "Surplus" — that line
already excludes the goal-floor amounts you must keep.

=== HOW TO DECIDE — WORKED EXAMPLES ===

Example 1 — peer offers a goal resource I still need
  state: inventory={{vino:4, oro:0}}, goal_needs={{oro:1}}, surplus={{vino:2}}
  exchangeable={{vino:2}}
  peer wants {{vino:2}}, offers {{oro:1}}
  → evaluate_trade(decision="accept", resources={{"vino":2}},
                   counter_request={{}}, clarify_text="",
                   reason="Receiving oro closes the goal gap.")

Example 2 — peer wants a goal resource I still need, offers nothing useful
  state: inventory={{trigo:1}}, goal_needs={{trigo:1}}, surplus={{}}
  exchangeable={{}}
  peer wants {{trigo:1}}, offers {{madera:3}}
  → evaluate_trade(decision="reject", resources={{}},
                   counter_request={{}}, clarify_text="",
                   reason="Trigo is still needed and madera does not advance the goal.")

Example 3 — direction right, price too high → counter
  state: inventory={{madera:6}}, goal_needs={{piedra:2}}, surplus={{madera:4}}
  exchangeable={{madera:4}}
  peer wants {{madera:4}}, offers {{piedra:1}}
  → evaluate_trade(decision="counter", resources={{"madera":2}},
                   counter_request={{"piedra":2}}, clarify_text="",
                   reason="Same direction but their offer covers only half the gap.")

Example 4 — chain trade
  state: inventory={{madera:5}}, goal_needs={{oro:1}},
         chain_opportunities={{piedra: ["oro"]}}
  exchangeable={{madera:2}}
  peer wants {{madera:2}}, offers {{piedra:1}}
  → evaluate_trade(decision="accept", resources={{"madera":2}},
                   counter_request={{}}, clarify_text="",
                   reason="Piedra is re-tradeable for oro via chain.")

Example 5 — message too vague
  peer text: "tienes algo de oro?"
  No concrete quantity or barter terms.
  → evaluate_trade(decision="clarify", resources={{}},
                   counter_request={{}},
                   clarify_text="¿Cuánto oro quieres y qué ofreces a cambio?",
                   reason="Cannot evaluate without quantities.")

=== FIELD MEANINGS ===
decision:
  accept   — give the exchangeable amounts; receiving advances goal
  offer    — give a smaller subset of exchangeable (partial grant)
  counter  — give from surplus and request at least one item from goal_needs
  reject   — peer offers nothing useful, or trade harms goal
  clarify  — peer's message lacks resource/quantity info needed to evaluate
resources       — what WE give; non-empty only for accept/offer/counter
counter_request — what we ask back; non-empty only for counter; must overlap goal_needs
clarify_text    — Spanish question; only for clarify
reason          — one sentence stating why this serves victory

=== TIE-BREAKERS ===
- If `near_victory` is true, prefer SPEED — accept slightly bad trades that
  finish the goal quickly.
- A chain-trade intermediate counts as a goal resource for accept purposes.

Call the `evaluate_trade` tool with your decision.
"""


# ---------------------------------------------------------------------------
# 3. Counter-offer generation  (PDF §4 — propose_trade)
# ---------------------------------------------------------------------------

def build_counter_offer_prompt(
    state_snapshot: dict,
    peer: str,
    their_request: dict[str, int],
    their_offer: dict[str, int],
    history: Iterable[ConversationTurn] = (),
) -> str:
    """
    The direction is right but the price is wrong. Generate a tighter barter.
    """
    return f"""You are a win-oriented trade agent generating a COUNTER-OFFER.
The peer's barter goes in the right direction but the price is wrong for us;
propose a tighter version that draws our give from Surplus and our ask from
Goal gap.

=== MY STATE ===
{_state_block(state_snapshot)}

=== PEER ===
peer={peer}

=== RECENT DIALOGUE ===
{_format_history(history)}

=== THEIR LATEST BARTER ===
They want from us:  {their_request}
They offer to us:   {their_offer}

=== HARD CONSTRAINT ===
Never let your inventory of any "Goal targets" resource drop below its
required quantity. Trade only what is shown in "Surplus" — that line
already excludes the goal-floor amounts you must keep.

=== HOW TO PRICE — WORKED EXAMPLES ===

Example 1 — halve the cost, match the gap
  state: surplus={{madera:5}}, goal_needs={{oro:2}}
  peer wants {{madera:4}}, offers {{oro:1}}
  → generate_counter_offer(we_give={{"madera":2}}, we_want={{"oro":2}},
                           reason="Match goal gap and halve our cost.")

Example 2 — redirect ask to a resource we actually need
  state: surplus={{tela:3}}, goal_needs={{piedra:1}}
  peer wants {{tela:2}}, offers {{madera:2}}  ← madera not in goal_needs
  → generate_counter_offer(we_give={{"tela":2}}, we_want={{"piedra":1}},
                           reason="Redirect ask to a goal resource.")

Example 3 — keep the trade small enough to be plausible
  state: surplus={{vino:6}}, goal_needs={{trigo:3}}
  peer wants {{vino:5}}, offers {{trigo:1}}
  → generate_counter_offer(we_give={{"vino":2}}, we_want={{"trigo":2}},
                           reason="Small symmetric trade is easier to close.")

=== FIELD MEANINGS ===
we_give — what WE hand over; drawn from Surplus, positive integers
we_want — what we ask back; must overlap Goal gap, positive integers
reason  — one short sentence stating why this serves victory

Call the `generate_counter_offer` tool with the result.
"""


# ---------------------------------------------------------------------------
# 4. Clarification request  (PDF §4 — send_message)
# ---------------------------------------------------------------------------

def build_clarification_prompt(
    state_snapshot: dict,
    peer: str,
    raw_text: str,
    history: Iterable[ConversationTurn] = (),
) -> str:
    """
    Produce a short Spanish question that pins down what the peer wants.
    The LLM is generating natural-language, but only one targeted question.
    """
    return f"""You are a trade agent. The peer's message is too ambiguous to
act on safely. Write ONE short Spanish question (max 20 words) that pins down
the concrete resources and quantities they want, without committing to any
trade yet.

=== MY STATE ===
{_state_block(state_snapshot)}

=== RECENT DIALOGUE ===
{_format_history(history)}

=== AMBIGUOUS PEER MESSAGE ===
"{raw_text}"

=== HOW TO ASK — WORKED EXAMPLES ===

Example 1 — missing quantities and counterpart
  peer text: "Te ofrezco madera."
  → ask_clarification(text="¿Cuánta madera ofreces y qué quieres a cambio?",
                      reason="Missing quantity and the requested counterpart.")

Example 2 — vague resource reference
  peer text: "Quiero algo de comida."
  → ask_clarification(text="¿Trigo o queso? ¿Cuántas unidades necesitas?",
                      reason="Resource and quantity unspecified.")

Example 3 — open one-word message
  peer text: "intercambio?"
  → ask_clarification(text="¿Qué recursos quieres y qué ofreces a cambio?",
                      reason="No terms at all; ask for both sides of the barter.")

=== FIELD MEANINGS ===
text   — one Spanish sentence, ≤20 words, ends with '?'
reason — one short sentence summarising what is unclear

The question should focus on resources and quantities; do not propose a trade
in this question — that happens in a separate step once the peer answers.

Call the `ask_clarification` tool with the question.
"""


# ---------------------------------------------------------------------------
# 5. Legacy normalization wrapper  (kept for back-compat — uses intent prompt)
# ---------------------------------------------------------------------------

def build_normalization_prompt(raw_text: str) -> str:
    """
    Thin wrapper kept so message_normalizer can fall back on this when no
    history/pending context is available. Internally identical to the intent
    prompt with empty context.
    """
    return build_intent_prompt(raw_text)


# ---------------------------------------------------------------------------
# 6. Legacy decision wrapper  (kept so older callers keep compiling)
# ---------------------------------------------------------------------------

def build_decision_prompt(
    inventory: dict[str, int],
    goal_needs: dict[str, int],
    target_resources: set[str],
    requested: dict[str, int],
    exchangeable: dict[str, int],
) -> str:
    """
    Backwards-compatible wrapper. Builds an evaluation prompt with a minimal
    snapshot when callers haven't been migrated to build_evaluate_prompt.
    """
    initial_goal = dict(goal_needs)
    surplus = {
        r: q for r, q in inventory.items()
        if r not in target_resources and q > 0
    }
    snapshot = {
        "inventory": inventory,
        "goal_needs": goal_needs,
        "initial_goal": initial_goal,
        "target_resources": set(target_resources),
        "surplus": surplus,
        "progress": 0.0,
        "near_victory": False,
    }
    return build_evaluate_prompt(
        state_snapshot=snapshot,
        requested=requested,
        offered_to_us={},
        exchangeable=exchangeable,
        peer="unknown",
    )


# ---------------------------------------------------------------------------
# 7. Natural-language trade message  (replaces outbound JSON)
# ---------------------------------------------------------------------------

def build_trade_message_prompt(
    we_give: dict[str, int],
    we_want: dict[str, int],
    state_snapshot: dict,
    peer: str,
    history: Iterable[ConversationTurn] = (),
) -> str:
    """
    Turn a structured barter decision into a single natural-language sentence
    in Spanish that ANY agent (or human) can understand unambiguously.
    """
    return f"""You are a trade agent. Convert the structured barter below into
ONE clear Spanish sentence (max 30 words) using the exact resource names and
quantities provided. Any agent must immediately understand both sides.

=== MY STATE ===
{_state_block(state_snapshot)}

=== RECENT DIALOGUE WITH THIS PEER ===
{_format_history(history)}

=== BARTER TO EXPRESS ===
I will give:    {we_give}
I want back:    {we_want}

=== HOW TO PHRASE — WORKED EXAMPLES ===

Example 1 — symmetric barter
  we_give={{madera:2}}, we_want={{oro:1}}
  → compose_trade_message(text="Te ofrezco 2 madera a cambio de 1 oro.",
                          reason="Direct symmetric barter.")

Example 2 — multiple resources on one side
  we_give={{vino:2, tela:1}}, we_want={{piedra:2}}
  → compose_trade_message(text="Te doy 2 vino y 1 tela a cambio de 2 piedras.",
                          reason="Bundle two surplus items for one ask.")

Example 3 — request only (no give)
  we_give={{}}, we_want={{trigo:1}}
  → compose_trade_message(text="¿Puedes enviarme 1 trigo, como acordamos?",
                          reason="Ask peer to honour a promised delivery.")

Example 4 — give only (no ask)
  we_give={{queso:3}}, we_want={{}}
  → compose_trade_message(text="Te envío 3 queso ahora, según lo acordado.",
                          reason="Announce delivery of an honoured promise.")

=== FIELD MEANINGS ===
text   — one Spanish sentence; use the exact resources/quantities from the
         BARTER block above; preferred phrasing is
         "Te ofrezco X a cambio de Y" or an equivalent clear variant
reason — one short sentence summarising the intent of the sentence

Call the `compose_trade_message` tool with the sentence.
"""


# ---------------------------------------------------------------------------
# Tool definitions — passed to Ollama's chat API via the `tools` field
# ---------------------------------------------------------------------------

INTENT_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "classify_intent",
        "description": (
            "Classify what the peer is doing in their latest message. "
            "Do not decide a trade — only extract intent kind and any "
            "numeric resources mentioned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": [
                        "request", "delivery", "accept", "reject",
                        "counter_offer", "clarification", "unknown",
                    ],
                },
                "resources": _qty_dict_schema(
                    "Resources the peer wants FROM us, keyed by resource name."
                ),
                "offered_resources": _qty_dict_schema(
                    "Resources the peer offers TO us, keyed by resource name."
                ),
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                },
                "reason": {"type": "string"},
            },
            "required": [
                "kind", "resources", "offered_resources",
                "confidence", "reason",
            ],
        },
    },
}

EVALUATE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "evaluate_trade",
        "description": (
            "Decide whether the peer's trade request advances our victory "
            "goal and choose accept / offer / counter / reject / clarify."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {
                    "type": "string",
                    "enum": ["accept", "offer", "counter", "reject", "clarify"],
                },
                "resources": _qty_dict_schema(
                    "What WE will give to the peer. Must be a subset of the "
                    "'exchangeable' allow-list with quantities <= caps. Empty "
                    "for reject/clarify."
                ),
                "counter_request": _qty_dict_schema(
                    "Only for decision='counter': what we want back from the "
                    "peer; must overlap with Goal gap. Empty otherwise."
                ),
                "clarify_text": {
                    "type": "string",
                    "description": (
                        "Only for decision='clarify': one Spanish question to "
                        "ask the peer. Empty otherwise."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "One short sentence stating WHY this serves victory.",
                },
            },
            "required": [
                "decision", "resources", "counter_request",
                "clarify_text", "reason",
            ],
        },
    },
}

COUNTER_OFFER_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "generate_counter_offer",
        "description": (
            "Produce a tighter counter-offer: what WE give from surplus and "
            "what we want back from the peer's goal-relevant resources."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "we_give": _qty_dict_schema(
                    "Resources WE give the peer — from surplus, never blocked."
                ),
                "we_want": _qty_dict_schema(
                    "Resources we want back — subset of Goal gap, positive integers."
                ),
                "reason": {"type": "string"},
            },
            "required": ["we_give", "we_want", "reason"],
        },
    },
}

CLARIFICATION_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "ask_clarification",
        "description": "Compose one short Spanish question to disambiguate the peer's intent.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "One Spanish sentence, max 20 words.",
                },
                "reason": {"type": "string"},
            },
            "required": ["text", "reason"],
        },
    },
}

TRADE_MESSAGE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "compose_trade_message",
        "description": "Compose one Spanish sentence stating an unambiguous barter offer.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "One Spanish sentence, max 30 words.",
                },
                "reason": {"type": "string"},
            },
            "required": ["text", "reason"],
        },
    },
}
