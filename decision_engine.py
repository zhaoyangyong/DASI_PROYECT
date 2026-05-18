"""
decision_engine.py
------------------
Core business logic for processing incoming messages.

Follows the PDF §7 decision flow:

    1. Save message to per-peer conversation history.
    2. Check rule-based fast accept (explicit accept + pending offer matches
       our inventory).
    3. Split requested resources into forbidden vs exchangeable.
    4. Ask the EVALUATE prompt for a victory-oriented decision.
    5. Validate the LLM output (action, inventory caps, target-resource
       protection, JSON shape).
    6. If invalid, attempt one format-repair pass; otherwise rule fallback.
    7. Branch by decision: accept/offer → ship resources; counter → send a
       counter-proposal message and store pending_offer; clarify → send a
       Spanish question; reject → optionally counter-request.

Rules: the LLM only *suggests* strategy; code enforces all constraints.
"""

from loguru import logger

import butler
import agents
from messaging import build_structured_request
from models import (
    ChainPlan,
    DecisionResponse,
    DeliveryResponse,
    NormalizedMessage,
    StateSnapshot,
)
from state_manager import state
from prompt_builder import (
    CLARIFICATION_TOOL,
    COUNTER_OFFER_TOOL,
    EVALUATE_TOOL,
    build_clarification_prompt,
    build_counter_offer_prompt,
    build_evaluate_prompt,
)
from ollama_client import call_ollama_tool
from utils import clean_qty_dict


# ---------------------------------------------------------------------------
# Public entry — request
# ---------------------------------------------------------------------------

async def process_request(msg: NormalizedMessage) -> DecisionResponse:
    """Process a 'request' (or 'counter_offer') from a peer."""
    snap = state.snapshot()
    inventory: dict[str, int] = snap["inventory"]
    # Two distinct semantics — keep them separate:
    #   target_resources  → PROTECTION (never trade below the goal qty)
    #   still_wanted      → ACQUISITION (still missing for the goal)
    # Most call sites that historically passed `target_resources` actually
    # meant "what we still want", which now lives in `still_wanted`.
    target_resources: set[str] = snap["target_resources"]
    goal_needs: dict[str, int] = snap["goal_needs"]
    still_wanted: set[str] = set(goal_needs.keys())
    requested: dict[str, int] = msg.resources
    peer = msg.from_agent

    # Honour-pending fast path. If the peer is just claiming what we already
    # promised — same-shape request with no fresh barter terms attached —
    # treat it as the second leg of a pre-agreed trade rather than running
    # it through the LLM and the counter-probe ladder. Without this, two
    # same-codebase agents could never complete a barter: peer A delivers
    # X, peer A asks for the promised Y, our process_request sees a bare
    # request for Y (which is in our goal/target) and refuses.
    pending = state.get_pending_offer(peer)
    if (
        pending is not None
        and pending.we_give
        and requested
        and not msg.offered_resources
        and all(
            r in pending.we_give and requested[r] <= pending.we_give[r]
            for r in requested
        )
    ):
        return await _honour_pending_claim(msg, pending)

    # Empty request: probably they're offering something but didn't list
    # what they want. Counter-request if their offer overlaps our goal,
    # else reject quietly.
    if not requested:
        await _counter_request_if_valuable(msg, still_wanted)
        return DecisionResponse(
            decision="reject", resources={}, reason="Empty request."
        )

    # --- Split: forbidden vs exchangeable (code-enforced; PDF §6) -----------
    # `snap["surplus"]` already encodes goal-target protection and chain
    # reservation accounting, so it is the right cap for what we can give.
    forbidden, exchangeable = _split_exchangeable(requested, snap["surplus"])
    if forbidden:
        logger.info(f"Forbidden resources (will not trade): {list(forbidden.keys())}")

    if not exchangeable:
        # Before hard-rejecting, try to flip the dead end into a probe:
        # "I can't give what you asked, but I have X — do you have Y?"
        # Falls through to reject if no useful probe is possible.
        logger.info("No exchangeable resources available — probing before reject.")
        countered = await _counter_request_if_valuable(msg, still_wanted)
        if not countered:
            state.set_last_decision(peer, "reject")
        return DecisionResponse(
            decision="reject",
            resources={},
            reason=f"Cannot provide: {list(forbidden.keys())}",
        )

    # If they offer nothing we still NEED to acquire, the trade can't help us
    # progress — short-circuit reject unless we have NO goal left (in which
    # case any trade is acceptable as long as it doesn't shrink protected
    # goal-resource inventory).
    incoming_goal = {
        r: q for r, q in msg.offered_resources.items()
        if r in still_wanted and q > 0
    }
    if goal_needs and not incoming_goal:
        # Before hard-rejecting, check if a chain trade is possible:
        # the peer's offered resources might be wanted by another peer
        # who has goal resources — accept now, re-trade later.
        if not state.find_chain_opportunity(msg.offered_resources, still_wanted):
            logger.info(
                "Peer offers no goal resources and no chain opportunity — "
                "probing before reject."
            )
            countered = await _counter_request_if_valuable(msg, still_wanted)
            if not countered:
                state.set_last_decision(peer, "reject")
            return DecisionResponse(
                decision="reject",
                resources={},
                reason="No goal resources offered in exchange.",
            )
        logger.info("No direct goal resources but chain trade opportunity detected — evaluating.")

    # --- LLM evaluation (PDF §4 trade-eval) --------------------------------
    decision_data = await _evaluate_with_llm(
        snap, msg, exchangeable, peer
    )

    decision = decision_data["decision"]
    resources = decision_data["resources"]
    counter_request = decision_data.get("counter_request", {})
    clarify_text = decision_data.get("clarify_text", "")
    reason = decision_data["reason"]

    # --- Execute ------------------------------------------------------------
    if decision in ("accept", "offer") and resources:
        success = await state.deduct_resources(resources)
        if not success:
            logger.warning("State deduction failed after decision — rejecting.")
            return DecisionResponse(
                decision="reject", resources={}, reason="Inventory check failed."
            )

        alias = await butler.get_alias_for_ip(peer)
        if alias:
            await butler.send_resources(alias, resources)
        else:
            logger.warning(
                f"Could not resolve alias for '{peer}'; "
                "resources deducted locally but Butler send skipped."
            )

        # If they promised goal resources, immediately request them.
        # `still_wanted` (not all protected goals): we only chase what's
        # still missing for the goal, not what's already satisfied.
        await _request_promised_resources(msg, still_wanted)
        # Chain trade second leg: if the resources peer A is sending us are
        # not in our goal set, look for a peer B who wants them in exchange
        # for one of our still-wanted resources, and fire the second leg
        # right now. `target_resources` is the right filter for "is this an
        # intermediate vs a goal resource", because peer offering an
        # already-met goal resource is not an intermediate either.
        if not incoming_goal and msg.offered_resources:
            await _setup_chain_second_leg(msg, target_resources, still_wanted)
        state.clear_pending_offer(peer)
        state.set_last_decision(peer, decision, we_gave=resources, we_wanted={})

    elif decision == "counter":
        sent = await _send_counter_offer(
            peer=peer,
            we_give=resources,
            we_want=counter_request,
            reason=reason,
        )
        if not sent:
            # Counter blocked by reject-loop guard or invalid payload — fall
            # through to reject so we don't silently no-op.
            logger.info("Counter blocked — falling through to reject.")
            decision = "reject"
            resources, counter_request, clarify_text = {}, {}, ""
            reason = f"{reason} (counter blocked by reject-loop guard)"
            # _counter_request_if_valuable will set last_decision="counter"
            # if it actually fires; otherwise stamp it as a real reject.
            if not await _counter_request_if_valuable(msg, still_wanted):
                state.set_last_decision(peer, "reject")

    elif decision == "clarify":
        await _send_clarification(peer=peer, text=clarify_text)
        state.set_last_decision(peer, "clarify")

    else:  # reject
        if not await _counter_request_if_valuable(msg, still_wanted):
            state.set_last_decision(peer, "reject")

    logger.info(f"Decision: {decision} | resources={resources} | reason={reason}")
    return DecisionResponse(
        decision=decision,
        resources=resources,
        counter_request=counter_request,
        clarify_text=clarify_text,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Public entry — accept (for messages classified as accept)
# ---------------------------------------------------------------------------

async def process_accept(msg: NormalizedMessage) -> DecisionResponse:
    """
    Handle a peer's "accept" of a pending offer we previously sent.

    PDF §5: only act if a pending offer exists AND inventory is still
    sufficient to honour it.
    """
    peer = msg.from_agent
    pending = state.get_pending_offer(peer)
    if pending is None:
        # No pending offer means this is just a stray "ok" — bring nothing.
        logger.info(f"[ACCEPT] from {peer} with no pending offer — ignoring.")
        return DecisionResponse(
            decision="reject", resources={},
            reason="No pending offer to accept.",
        )

    # Re-validate at acceptance time against the live surplus, which already
    # protects the goal-target floor + chain reservations. A single check
    # against surplus encompasses both "we'd drop below the goal qty" and
    # "we no longer have enough stock".
    snap = state.snapshot()
    surplus = snap["surplus"]
    promised = pending.we_give

    for r, q in promised.items():
        if surplus.get(r, 0) < q:
            logger.warning(
                f"[ACCEPT] cannot honour promised '{r}' "
                f"(need {q}, free surplus {surplus.get(r, 0)}) — cancelling."
            )
            state.clear_pending_offer(peer)
            return DecisionResponse(
                decision="reject", resources={},
                reason="Pending offer no longer honourable against current surplus.",
            )

    success = await state.deduct_resources(promised)
    if not success:
        state.clear_pending_offer(peer)
        return DecisionResponse(
            decision="reject", resources={},
            reason="Inventory check failed at accept time.",
        )

    alias = await butler.get_alias_for_ip(peer)
    if alias:
        await butler.send_resources(alias, promised)

    # Ask them to honour their side if it overlaps our goal
    if pending.we_want:
        request_msg = await _generate_trade_message({}, pending.we_want, peer)
        delivered = await agents.send_message_to_agent(peer, request_msg)
        if delivered is not None:
            state.record_turn(peer, "self", request_msg, kind="request")
        else:
            logger.warning(
                f"[ACCEPT] honour-request to {peer} failed — not recording turn."
            )

    state.clear_pending_offer(peer)
    state.mark_last_response(peer, "accepted")
    return DecisionResponse(
        decision="accept",
        resources=promised,
        reason="Honoured pending offer after peer accept.",
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _split_exchangeable(
    requested: dict[str, int],
    surplus: dict[str, int],
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Split a peer request into `(forbidden, exchangeable)`:

        forbidden    — quantities we refuse to even consider
        exchangeable — the maximum we are allowed to give

    `surplus` already encodes every protection rule: goal-target floors are
    subtracted, chain-trade reservations are subtracted, non-positive amounts
    are excluded. A resource is forbidden iff the qty is invalid or exceeds
    what's left in surplus.
    """
    forbidden: dict[str, int] = {}
    exchangeable: dict[str, int] = {}
    for resource, qty in requested.items():
        if not isinstance(qty, int) or qty <= 0:
            forbidden[resource] = qty
        elif surplus.get(resource, 0) < qty:
            forbidden[resource] = qty
        else:
            exchangeable[resource] = qty
    return forbidden, exchangeable


async def _evaluate_with_llm(
    snap: dict,
    msg: NormalizedMessage,
    exchangeable: dict[str, int],
    peer: str,
) -> dict:
    """Ask Ollama for a victory-oriented decision; fall back on rules."""
    history = state.get_history(peer)
    pending = state.get_pending_offer(peer)
    prompt = build_evaluate_prompt(
        state_snapshot=snap,
        requested=msg.resources,
        offered_to_us=msg.offered_resources,
        exchangeable=exchangeable,
        peer=peer,
        history=history,
        pending=pending,
    )
    args = await call_ollama_tool(prompt, EVALUATE_TOOL)

    if args is not None:
        validated = _validate_decision(args, snap, exchangeable, msg)
        if validated is not None:
            logger.info(f"Using Ollama decision: {validated['decision']}")
            return validated
        logger.warning("Ollama decision failed business validation — using rule fallback.")

    logger.info("Falling back to rule-based decision.")
    return _rule_based_decision(snap, msg, exchangeable)


def _validate_decision(
    data: dict,
    snap: StateSnapshot,
    exchangeable: dict[str, int],
    msg: NormalizedMessage,
) -> dict | None:
    """
    Verify every code-side invariant on the tool-call arguments:
        - valid `decision` enum value
        - `resources` ⊆ exchangeable, ≤ inventory caps, ≠ blocked
        - counter requests at least one resource still in `goal_needs`
        - clarify text non-empty (fall back to a default)
        - reject zeros everything out

    Returns the cleaned decision dict, or None on any violation so the
    caller can fall through to the rule-based decision.
    """
    decision = data.get("decision")
    if decision not in ("accept", "offer", "counter", "reject", "clarify"):
        logger.warning(f"Ollama invalid decision value: '{decision}'")
        return None

    resources = data.get("resources") or {}
    counter_request = data.get("counter_request") or {}
    clarify_text = str(data.get("clarify_text") or "")
    reason = str(data.get("reason") or "")

    if not isinstance(resources, dict) or not isinstance(counter_request, dict):
        return None

    surplus = snap["surplus"]
    goal_needs = snap["goal_needs"]

    # ---- accept / offer: must come from `exchangeable`, capped by surplus --
    # `exchangeable` is built from surplus, so it already excludes anything
    # earmarked for the goal floor or held in chain reservation. We just
    # need to verify the LLM didn't sneak in something outside the cap.
    if decision in ("accept", "offer"):
        if not resources:
            return None
        for r, q in resources.items():
            if not isinstance(q, int) or q < 0:
                return None
            if r not in exchangeable:
                logger.warning(f"Ollama included non-exchangeable resource: '{r}'")
                return None
            if q > exchangeable[r]:
                logger.warning(f"Ollama exceeded cap for '{r}'")
                return None

    # ---- counter: must give from surplus & ask for something we still need
    if decision == "counter":
        if not resources or not counter_request:
            return None
        for r, q in resources.items():
            if not isinstance(q, int) or q <= 0:
                return None
            if surplus.get(r, 0) < q:
                logger.warning(
                    f"Ollama counter exceeds surplus for '{r}' "
                    f"(asked {q}, surplus {surplus.get(r, 0)})"
                )
                return None
        # we_want must overlap with goal gap (resources we still need)
        any_useful = any(
            isinstance(q, int) and q > 0 and r in goal_needs
            for r, q in counter_request.items()
        )
        if not any_useful:
            logger.warning("Ollama counter asks for nothing in goal gap.")
            return None
        resources = clean_qty_dict(resources)
        counter_request = clean_qty_dict(counter_request)

    # ---- clarify: requires non-empty text ----------------------------------
    if decision == "clarify":
        resources, counter_request = {}, {}
        if not clarify_text:
            clarify_text = "¿Puedes especificar qué recursos y cuántas unidades quieres?"

    # ---- reject: clear payloads --------------------------------------------
    if decision == "reject":
        resources, counter_request, clarify_text = {}, {}, ""

    return {
        "decision": decision,
        "resources": resources,
        "counter_request": counter_request,
        "clarify_text": clarify_text,
        "reason": reason,
    }


def _rule_based_decision(
    snap: StateSnapshot,
    msg: NormalizedMessage,
    exchangeable: dict[str, int],
) -> dict:
    """
    Last-resort fallback when Ollama is unavailable or repeatedly wrong.

    Rule: if peer offers something we still need for the goal, give them all
    of the exchangeable amounts. Otherwise reject. Matches PDF §3 — only
    accept when it shortens the goal gap.
    """
    still_wanted = set(snap["goal_needs"].keys())
    incoming_goal = {
        r: q for r, q in msg.offered_resources.items()
        if r in still_wanted and q > 0
    }
    if incoming_goal:
        return {
            "decision": "accept",
            "resources": exchangeable,
            "counter_request": {},
            "clarify_text": "",
            "reason": "Rule fallback: peer offers goal resources, exchanging surplus.",
        }
    return {
        "decision": "reject",
        "resources": {},
        "counter_request": {},
        "clarify_text": "",
        "reason": "Rule fallback: peer offers nothing toward our goal.",
    }


# ---------------------------------------------------------------------------
# Outbound helpers — counter / clarify / counter-request
# ---------------------------------------------------------------------------

_SWEETEN_GIVE_RATIO = 3  # ceiling on (sum we_give) / (sum we_want) per trade
# Minimum gap between two speculative counter-request probes to the SAME
# peer. The speculative path is stateless — without this throttle a peer
# that replies fast drives one probe per inbound message (message storm).
_SPECULATIVE_PROBE_COOLDOWN = 30.0  # seconds


def _sweeten_against_loop(
    snap: StateSnapshot | dict,
    we_give: dict[str, int],
    we_want: dict[str, int],
) -> tuple[dict[str, int], dict[str, int]] | None:
    """
    PDF §2.3 — when we'd be repeating an offer that the peer already
    rejected, try ONE adjustment before giving up:

        1. Lower demand: reduce the largest `we_want` quantity by 1
           (only if the remainder stays positive).
        2. Else sweeten the deal: increase the smallest `we_give` quantity
           by 1, as long as
              · we still have surplus for it, AND
              · the total give would not exceed _SWEETEN_GIVE_RATIO × total
                want (so we never bid 14 oro for 1 queso even after many
                sweeten rounds).

    Returns the adjusted (we_give, we_want) tuple, or None if neither path
    is viable (caller should fall through to reject).
    """
    surplus = snap.get("surplus") or {}

    # 1. Try lowering demand
    if we_want:
        biggest = max(we_want.items(), key=lambda kv: kv[1])
        r, q = biggest
        if q > 1:
            new_want = dict(we_want)
            new_want[r] = q - 1
            return clean_qty_dict(we_give), clean_qty_dict(new_want)

    # 2. Try sweetening give (with global ratio ceiling)
    want_total = sum(we_want.values()) or 1
    give_total = sum(we_give.values())
    give_ceiling = want_total * _SWEETEN_GIVE_RATIO
    if we_give and give_total < give_ceiling:
        # smallest current give whose surplus allows +1
        candidates = sorted(we_give.items(), key=lambda kv: kv[1])
        for r, q in candidates:
            if surplus.get(r, 0) > q:
                new_give = dict(we_give)
                new_give[r] = q + 1
                return clean_qty_dict(new_give), clean_qty_dict(we_want)

    return None


def _net_overlap(
    give: dict[str, int],
    want: dict[str, int],
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Cancel out same-resource quantities on both sides of a trade.

    Without this we can end up emitting `Te ofrezco 1 queso a cambio de 1
    queso.` when our surplus contains an excess goal resource and the peer
    happens to offer the same resource. Netting also keeps the reject-loop
    guard comparing the *effective* shape rather than a redundant inflated
    one.
    """
    overlap = set(give.keys()) & set(want.keys())
    if not overlap:
        return give, want
    give = dict(give)
    want = dict(want)
    for r in overlap:
        common = min(give[r], want[r])
        give[r] -= common
        want[r] -= common
    return (
        {r: q for r, q in give.items() if q > 0},
        {r: q for r, q in want.items() if q > 0},
    )


async def _send_counter_offer(
    peer: str,
    we_give: dict[str, int],
    we_want: dict[str, int],
    reason: str,
) -> bool:
    """
    Emit a counter-proposal as a structured request and record it.

    Returns False (without sending) if the same offer would repeat one we
    already sent to this peer that ended in reject/pending — PDF §2.3
    reject-loop guard. Caller should then fall through to a different
    decision (typically reject).
    """
    if not we_give or not we_want or peer in ("unknown", ""):
        return False
    we_give = clean_qty_dict(we_give)
    we_want = clean_qty_dict(we_want)
    we_give, we_want = _net_overlap(we_give, we_want)
    if not we_give or not we_want:
        return False

    # PDF §2.3 reject-loop guard — sweeten repeatedly until the shape is
    # genuinely new (not in the recent window), or abort. Sweetening is
    # monotone, so the loop is bounded.
    snap = state.snapshot()
    sweetened_any = False
    while state.is_repeat_offer(peer, we_give, we_want):
        sweetened = _sweeten_against_loop(snap, we_give, we_want)
        if sweetened is None:
            logger.warning(
                f"[COUNTER] exhausted sweetening for {peer} — aborting counter."
            )
            return False
        we_give, we_want = sweetened
        sweetened_any = True
    if sweetened_any:
        reason = f"{reason} (sweetened to avoid reject-loop)"

    msg = await _generate_trade_message(we_give, we_want, peer)
    logger.info(f"[COUNTER] -> {peer}: give={we_give} want={we_want} ({reason})")
    delivered = await agents.send_message_to_agent(peer, msg)
    if delivered is None:
        # HTTP send failed — don't pretend the counter is in flight, or the
        # reject-loop guard + pending TTL will block fresh shapes for minutes
        # and chain second-leg reservations would never get released on time.
        logger.warning(f"[COUNTER] send to {peer} failed — not recording pending.")
        return False
    state.set_pending_offer(peer, we_give=we_give, we_want=we_want)
    state.set_last_decision(peer, "counter", we_gave=we_give, we_wanted=we_want)
    # Past the send guard above — record the CONFIRMED-sent shape.
    state.record_outbound_offer(peer, we_give, we_want)
    state.record_turn(peer, "self", msg, kind="counter_offer")
    return True


def _fmt_qty_list(d: dict[str, int]) -> str:
    """Format a {resource: qty} dict as 'N r, N r, ...' (drops non-positive)."""
    return ", ".join(f"{q} {r}" for r, q in d.items() if isinstance(q, int) and q > 0)


async def _generate_trade_message(
    we_give: dict[str, int],
    we_want: dict[str, int],
    peer: str,
) -> str:
    """
    Compose a Spanish trade message deterministically.

    Used to be an Ollama round-trip with a template fallback — but the LLM
    was the dominant source of malformed outputs (one-char replies, missing
    quantities, occasionally inventing resource names) and our own fast-path
    regex on the receiving end already requires the canonical phrasings.
    A deterministic template is faster, always parseable by the peer's
    fast path, and removes a whole class of bug.

    Three shapes covered:
      - both sides → "Te ofrezco {give} a cambio de {want}."   (Format B)
      - want only  → "¿Puedes enviarme {want}, como acordamos?"
      - give only  → "Te envío {give} ahora."                   (Format C)
    """
    give_str = _fmt_qty_list(we_give)
    want_str = _fmt_qty_list(we_want)

    if give_str and want_str:
        return f"Te ofrezco {give_str} a cambio de {want_str}."
    if want_str:
        return f"¿Puedes enviarme {want_str}, como acordamos?"
    if give_str:
        return f"Te envío {give_str} ahora."
    return ""


async def _send_clarification(peer: str, text: str) -> None:
    if not text or peer in ("unknown", ""):
        return
    logger.info(f"[CLARIFY] -> {peer}: {text}")
    delivered = await agents.send_message_to_agent(peer, text)
    if delivered is None:
        logger.warning(f"[CLARIFY] send to {peer} failed — not recording turn.")
        return
    state.record_turn(peer, "self", text, kind="clarification")


async def _counter_request_if_valuable(
    msg: NormalizedMessage,
    still_wanted: set[str],
) -> bool:
    """
    Turn a hard reject into an active probe so the conversation actually
    advances. `still_wanted` is the set of resources we still need to
    ACQUIRE (i.e. `goal_needs.keys()`) — not the full protected-goal set.
    Three paths:

      Concrete — peer's offer already includes a still-wanted resource:
        match that ask 1:1 with a small slice of our largest surplus.

      Chain — peer's offer doesn't directly include something still wanted,
        but it's an intermediate a known third party would trade for one of
        our still-wanted resources.

      Speculative — peer's offer has nothing useful and no chain hook;
        probe with one surplus item for one still-wanted item, capped at 3.

    Returns True iff a counter-request was actually sent. Honours the
    PDF §2.3 reject-loop guard so we don't endlessly resend the same shape.
    """
    if msg.from_agent in ("unknown", ""):
        return False

    snap = state.snapshot()
    goal_needs = snap["goal_needs"]
    tradeable = {k: v for k, v in snap["surplus"].items() if v > 0}
    if not tradeable or not goal_needs:
        return False

    # Concrete path: peer offered something we still need. We probe with
    # a SINGLE surplus item sized roughly to the want — never the whole
    # surplus, otherwise we end up promising "4 ladrillos + 3 piedra + 14
    # oro a cambio de 1 queso" which gives the farm away on the first try.
    # Subsequent rounds sweeten one unit at a time (capped by the ceiling
    # in _sweeten_against_loop) until peer accepts.
    goal_offers = {
        r: int(qty) for r, qty in (msg.offered_resources or {}).items()
        if r in still_wanted and qty > 0
    }

    if goal_offers:
        want = clean_qty_dict(goal_offers)
        want_total = sum(want.values())
        # Largest pile first — that's where we have the most slack — but
        # cap qty so the initial probe stays roughly 1:1 with the ask.
        give_r, give_qty_avail = max(tradeable.items(), key=lambda kv: kv[1])
        probe_qty = min(want_total, give_qty_avail, 3)
        give = clean_qty_dict({give_r: probe_qty})
        path = "concrete"
    elif (chain_match := state.find_chain_target(
            # Intermediate ≠ any goal resource (protected OR satisfied).
            # `still_wanted` would be too narrow here; use the full set.
            {r: int(q) for r, q in (msg.offered_resources or {}).items()
             if q > 0 and r not in snap["target_resources"]},
            still_wanted,
        )) is not None:
        # Chain-aware path: peer didn't offer a goal resource directly, but
        # what they offered is something a known third-party peer wants in
        # exchange for one of our goal resources. Ask peer A for the
        # intermediate; once it arrives, _setup_chain_second_leg (or a fresh
        # _counter_request_if_valuable call) closes the second leg.
        #
        # Sized like the speculative probe — this bet has TWO failure modes
        # (peer A may decline, peer B may decline later), so we keep the
        # initial offer small and let the reject-loop guard sweeten it.
        _, intermediate, intermediate_qty, _, _ = chain_match
        give_r, give_qty = next(iter(tradeable.items()))
        probe_qty = min(int(intermediate_qty), int(give_qty), 3)
        want = clean_qty_dict({intermediate: probe_qty})
        give = clean_qty_dict({give_r: probe_qty})
        path = "chain"
    else:
        # Speculative probe: pick ONE goal need + ONE surplus item, capped
        # at 3 each so we leave room for the reject-loop guard to sweeten.
        #
        # This branch is stateless — it recomputes the same shape on every
        # inbound message. Throttle per peer so a fast-replying peer cannot
        # turn each of its messages into one of ours (message storm).
        since = state.seconds_since_speculative_probe(msg.from_agent)
        if since is not None and since < _SPECULATIVE_PROBE_COOLDOWN:
            logger.info(
                f"[COUNTER-REQ:speculative] throttled for {msg.from_agent} "
                f"({since:.0f}s < {_SPECULATIVE_PROBE_COOLDOWN:.0f}s) — "
                "rejecting instead of re-probing."
            )
            return False
        want_r, want_qty = next(iter(goal_needs.items()))
        give_r, give_qty = next(iter(tradeable.items()))
        probe_qty = min(int(want_qty), int(give_qty), 3)
        want = clean_qty_dict({want_r: probe_qty})
        give = clean_qty_dict({give_r: probe_qty})
        path = "speculative"

    # Defuse the "X for X" trap: when the same resource appears on both
    # sides (e.g. peer offers queso and our surplus already has queso),
    # cancel the overlapping qty before deciding anything else.
    give, want = _net_overlap(give, want)
    if not want or not give:
        return False

    # PDF §2.3 reject-loop guard — don't resend a shape already in the
    # recent window. Sweeten repeatedly until the shape is genuinely new,
    # or give up: sweetening is monotone (demand down / give up to the
    # ratio ceiling), so this loop terminates in a bounded number of steps.
    while state.is_repeat_offer(msg.from_agent, give, want):
        sweetened = _sweeten_against_loop(snap, give, want)
        if sweetened is None:
            logger.warning(
                f"[COUNTER-REQ] exhausted sweetening for {msg.from_agent} "
                "— skipping."
            )
            return False
        give, want = sweetened

    request_msg = await _generate_trade_message(give, want, msg.from_agent)
    logger.info(
        f"[COUNTER-REQ:{path}] -> {msg.from_agent}: want={want} give={give}"
    )
    delivered = await agents.send_message_to_agent(msg.from_agent, request_msg)
    if delivered is None:
        # Same rationale as _send_counter_offer: failed sends must not leave
        # a phantom pending behind. Caller will treat False as "no counter
        # actually went out" and fall through to a plain reject.
        logger.warning(
            f"[COUNTER-REQ:{path}] send to {msg.from_agent} failed — "
            "not recording pending."
        )
        return False
    state.set_pending_offer(msg.from_agent, we_give=give, we_want=want)
    state.set_last_decision(
        msg.from_agent, "counter", we_gave=give, we_wanted=want
    )
    # Past the send guard above — these record a CONFIRMED-sent offer.
    state.record_outbound_offer(msg.from_agent, give, want)
    if path == "speculative":
        state.mark_speculative_probe(msg.from_agent)
    state.record_turn(msg.from_agent, "self", request_msg, kind="request")
    return True


async def _honour_pending_claim(
    msg: NormalizedMessage,
    pending,
) -> DecisionResponse:
    """
    Settle our giving side of a previously-agreed trade.

    The peer is asking for exactly what we already promised (matched against
    pending.we_give). Skip the LLM and the counter-probe ladder, deduct the
    promised quantity, ship it via Butler, and decrement the pending so the
    obligation tracker stays consistent. Any leftover `we_want` keeps the
    pending alive until we receive what the peer still owes us.
    """
    peer = msg.from_agent
    promised = pending.we_give
    honour: dict[str, int] = {}
    for r, q in clean_qty_dict(msg.resources).items():
        promised_qty = promised.get(r, 0)
        if promised_qty <= 0:
            continue
        honour[r] = min(int(q), int(promised_qty))
    if not honour:
        return DecisionResponse(
            decision="reject", resources={},
            reason="Pending promise no longer applies.",
        )

    success = await state.deduct_resources(honour)
    if not success:
        # Promise outlived our inventory (e.g. resources got synced away).
        # Clear the stale pending so we don't loop, and let the peer
        # re-negotiate from scratch.
        state.clear_pending_offer(peer)
        return DecisionResponse(
            decision="reject", resources={},
            reason="Insufficient stock to honour pending promise.",
        )

    alias = await butler.get_alias_for_ip(peer)
    if alias:
        await butler.send_resources(alias, honour)
    else:
        logger.warning(
            f"[HONOUR] could not resolve alias for {peer}; "
            "resources deducted but Butler send skipped."
        )

    state.consume_pending(peer, delivered_to_peer=honour)
    state.set_last_decision(peer, "accept", we_gave=honour, we_wanted={})

    logger.info(f"[HONOUR] -> {peer}: settled {honour} from pending promise")
    return DecisionResponse(
        decision="accept",
        resources=honour,
        reason="Honoured pre-agreed pending promise.",
    )


async def _request_promised_resources(
    msg: NormalizedMessage,
    still_wanted: set[str],
) -> None:
    """
    After accepting, ask the peer to deliver the goal resources they offered.
    `still_wanted` is the set we are still missing for the goal (i.e.
    `goal_needs.keys()`); we only chase resources we still need, not ones
    we already hold enough of.
    """
    if not msg.offered_resources or msg.from_agent in ("unknown", ""):
        return
    wanted = {
        r: qty for r, qty in msg.offered_resources.items()
        if r in still_wanted and qty > 0
    }
    if not wanted:
        return

    request_msg = await _generate_trade_message({}, wanted, msg.from_agent)
    logger.info(f"Requesting promised goal resources from {msg.from_agent}: {wanted}")
    delivered = await agents.send_message_to_agent(msg.from_agent, request_msg)
    if delivered is None:
        logger.warning(
            f"Goal-resource follow-up to {msg.from_agent} failed — "
            "not recording turn."
        )
        return
    state.record_turn(msg.from_agent, "self", request_msg, kind="request")


async def _setup_chain_second_leg(
    msg: NormalizedMessage,
    target_resources: set[str],
    still_wanted: set[str],
) -> None:
    """
    Triggered right after we accept an intermediate (non-goal) resource from
    peer A. Find a known peer B who both wants the intermediate AND holds
    something we still need to acquire, reserve the incoming quantity, and
    immediately fire a counter-offer at peer B to close the second leg.

    Two arg sets — different semantics:
      target_resources — the full protected-goal set. Used to FILTER
        intermediates: a resource that is part of our goal (whether or not
        already met) is not an intermediate.
      still_wanted — the resources we still need to acquire. Passed to
        find_chain_target so we only chain toward something we actually
        still want; chaining toward an already-met goal is pointless.

    The reservation prevents another peer from pulling the intermediate out
    of free surplus before peer B can claim it. If the counter-offer cannot
    be sent (peer B unreachable, reject-loop guard, etc.) we release the
    reservation right away so the intermediate flows back to normal surplus.
    """
    intermediate_resources = {
        r: q for r, q in (msg.offered_resources or {}).items()
        if q > 0 and r not in target_resources
    }
    if not intermediate_resources:
        return

    match = state.find_chain_target(intermediate_resources, still_wanted)
    if match is None:
        return

    target_peer, intermediate, intermediate_qty, goal_resource, peer_has_qty = match
    if target_peer == msg.from_agent:
        # The same peer can't sensibly be both legs of the chain.
        return

    target_qty = min(peer_has_qty, intermediate_qty)
    if target_qty <= 0 or intermediate_qty <= 0:
        return

    plan = ChainPlan(
        intermediate=intermediate,
        intermediate_qty=intermediate_qty,
        from_peer=msg.from_agent,
        target_peer=target_peer,
        target_resource=goal_resource,
        target_qty=target_qty,
    )
    state.reserve_intermediate(plan)

    sent = await _send_counter_offer(
        peer=target_peer,
        we_give={intermediate: intermediate_qty},
        we_want={goal_resource: target_qty},
        reason=f"chain second leg: just accepted {intermediate} from {msg.from_agent}",
    )
    if not sent:
        state.release_intermediate(intermediate, reason="second_leg_send_blocked")


# ---------------------------------------------------------------------------
# Delivery
# ---------------------------------------------------------------------------

async def process_delivery(msg: NormalizedMessage) -> DeliveryResponse:
    """Add received resources to inventory and update obligation tracking."""
    resources = msg.resources
    if not resources:
        return DeliveryResponse(status="error", message="Delivery contained no resources.")

    await state.add_resources(resources)
    # Decrement pending.we_want by what we just received. Pending only
    # auto-clears when BOTH sides of the trade are settled, so the peer's
    # still-outstanding "owe" survives the first delivery and the second
    # leg can be honoured by _honour_pending_claim below.
    state.consume_pending(msg.from_agent, received_from_peer=resources)
    snap = state.snapshot()
    logger.info(f"Delivery processed | new state: {snap}")
    return DeliveryResponse(status="ok", message="Resources received and state updated.")


# ---------------------------------------------------------------------------
# Clarification handler — when peer asks US for clarification
# ---------------------------------------------------------------------------

async def process_clarification(msg: NormalizedMessage) -> None:
    """
    Respond to an ambiguous peer message with a tiered, loop-safe strategy:

      1st consecutive clarify → ask one targeted Spanish question (LLM).
      2nd consecutive clarify → re-state our position with full quantities
                                ("Necesito X. Puedo ofrecer Y."). This is a
                                self-state broadcast, NOT a commitment, so
                                we never set pending_offer or last_decision
                                here — it must not interfere with the
                                reject-loop guard or fresh-pending skip.
      3rd+ consecutive        → silent; rely on the proactive loop to
                                re-engage the peer later.

    The counter is reset in `_handle_inbox_locked` the moment the peer
    sends any concrete message kind.
    """
    peer = msg.from_agent
    count = state.bump_clarify_count(peer)

    if count >= 3:
        logger.info(
            f"[CLARIFY-LOOP] silent on {peer} (count={count}) — "
            "letting the proactive loop re-engage."
        )
        return

    if count == 2:
        snap = state.snapshot()
        tradeable = {k: v for k, v in snap["surplus"].items() if v > 0}
        goal = snap["goal_needs"]
        if not tradeable or not goal:
            logger.info(
                f"[CLARIFY-LOOP] re-state to {peer} skipped (nothing to say)"
            )
            return
        give_str = ", ".join(f"{q} {r}" for r, q in tradeable.items())
        want_str = ", ".join(f"{q} {r}" for r, q in goal.items())
        text = f"Necesito {want_str}. Puedo ofrecer {give_str}."
        logger.info(f"[CLARIFY-LOOP] re-state -> {peer}: {text}")
        delivered = await agents.send_message_to_agent(peer, text)
        if delivered is None:
            logger.warning(
                f"[CLARIFY-LOOP] re-state to {peer} failed — not recording turn."
            )
            return
        state.record_turn(peer, "self", text, kind="request")
        # Intentionally do NOT touch last_decision or pending_offer here:
        # this message is a state broadcast, not a counter commitment.
        return

    # First clarification — ask one focused question via the LLM.
    snap = state.snapshot()
    history = state.get_history(peer)
    prompt = build_clarification_prompt(
        state_snapshot=snap,
        peer=peer,
        raw_text=msg.raw_text,
        history=history,
    )
    data = await call_ollama_tool(prompt, CLARIFICATION_TOOL)
    text = str((data or {}).get("text") or "").strip()

    if not text:
        # Deterministic Spanish fallback summarising what we want / offer
        need = ", ".join(snap["goal_needs"].keys()) or "nada"
        offer = ", ".join(snap["surplus"].keys()) or "nada"
        text = f"Necesito [{need}]. Puedo ofrecer [{offer}]. ¿Qué propones?"

    delivered = await agents.send_message_to_agent(peer, text)
    if delivered is None:
        logger.warning(
            f"[CLARIFY] reply to {peer} failed — not recording turn."
        )
        return
    state.record_turn(peer, "self", text, kind="clarification")


# ---------------------------------------------------------------------------
# Counter-offer generation entry — used when peer counter-offers us
# ---------------------------------------------------------------------------

async def process_counter_offer(msg: NormalizedMessage) -> DecisionResponse:
    """
    A peer is countering our previous offer. Evaluate as a fresh request,
    but if we want to re-counter ourselves, use the counter-offer prompt
    to tighten the price.
    """
    # First run normal evaluation
    decision = await process_request(msg)

    # If we chose 'counter' inside process_request, the engine already sent a
    # counter back. If we rejected but the trade is in the right direction,
    # try generating a softer counter via the dedicated counter prompt.
    if decision.decision != "reject":
        return decision

    snap = state.snapshot()
    # "Useful overlap" means peer offers something we still NEED to acquire.
    still_wanted = set(snap["goal_needs"].keys())
    goal_overlap = {
        r: q for r, q in msg.offered_resources.items()
        if r in still_wanted and q > 0
    }
    if not goal_overlap:
        return decision

    prompt = build_counter_offer_prompt(
        state_snapshot=snap,
        peer=msg.from_agent,
        their_request=msg.resources,
        their_offer=msg.offered_resources,
        history=state.get_history(msg.from_agent),
    )
    data = await call_ollama_tool(prompt, COUNTER_OFFER_TOOL)
    if data is None:
        return decision

    we_give = clean_qty_dict(data.get("we_give") or {})
    we_want = clean_qty_dict(data.get("we_want") or {})

    # Re-validate counter against state: every give item must fit in surplus
    # (surplus already excludes goal-floor and chain reservation); every
    # want item must overlap with what we still need.
    surplus = snap["surplus"]
    for r, q in we_give.items():
        if surplus.get(r, 0) < q:
            return decision
    if not any(r in still_wanted for r in we_want):
        return decision

    await _send_counter_offer(
        peer=msg.from_agent,
        we_give=we_give,
        we_want=we_want,
        reason=str(data.get("reason") or "counter-after-reject"),
    )
    return DecisionResponse(
        decision="counter",
        resources=we_give,
        counter_request=we_want,
        reason="Re-countered after initial reject.",
    )
