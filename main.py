"""
main.py
-------
FastAPI entry point. Handles startup/shutdown, route definitions, and
high-level orchestration of the PDF §7 decision flow:

    inbox  →  normalize  →  intent branch  →  decision_engine
                                       │
                                       ├─ request       → process_request
                                       ├─ counter_offer → process_counter_offer
                                       ├─ accept        → process_accept
                                       ├─ delivery      → process_delivery
                                       ├─ clarification → process_clarification
                                       └─ reject / unknown → record only
"""

import asyncio
import json
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from loguru import logger
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

import butler
import agents
from config import (
    AGENT_NAME,
    CHAIN_RESERVE_TTL,
    MY_PORT,
    PENDING_OFFER_TTL,
    SERVER_URL,
)
from decision_engine import (
    process_accept,
    process_clarification,
    process_counter_offer,
    process_delivery,
    process_request,
)
from events import emit, recent, stream as event_stream
from message_normalizer import normalize
from messaging import build_structured_request
from models import IncomingMessage
from state_manager import state

_DASHBOARD_HTML = (Path(__file__).parent / "dashboard.html").read_text()


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------

async def _sync_from_butler_loop() -> None:
    while True:
        await asyncio.sleep(20)
        info = await butler.get_agent_info()
        if not info:
            continue
        inventory = (
            info.get("Recursos") or info.get("recursos") or info.get("resources") or {}
        )
        before = state.snapshot()
        await state.sync_from_butler(
            {k: int(v) for k, v in inventory.items() if isinstance(v, (int, float))}
        )
        after = state.snapshot()

        if after["inventory"] != before["inventory"] or after["goal_needs"] != before["goal_needs"]:
            gained = {
                r: after["inventory"].get(r, 0) - before["inventory"].get(r, 0)
                for r in after["inventory"]
                if after["inventory"].get(r, 0) > before["inventory"].get(r, 0)
            }
            if gained:
                emit("delivery", from_="Butler", resources=gained)
                logger.info(f"[BUTLER SYNC] Received via Butler: {gained}")


async def _proactive_request_loop() -> None:
    await asyncio.sleep(15)

    while True:
        # Drop any offers that have been sitting unanswered too long, so the
        # reject-loop guard can reshape and we don't keep counting dead deals
        # as "in flight".
        expired = state.expire_stale_pending_offers(PENDING_OFFER_TTL)
        if expired:
            emit("pending_expired", peers=expired)

        # Same idea for chain reservations: if peer B never replies to a
        # second-leg counter, free the intermediate so it can be retraded.
        expired_chain = state.expire_stale_intermediates(CHAIN_RESERVE_TTL)
        if expired_chain:
            emit("chain_expired", intermediates=expired_chain)

        snap = state.snapshot()
        goal = snap["goal_needs"]

        # Exit only when we are both done collecting AND nobody owes us a reply.
        if not goal and not state.pending_summary():
            logger.info("All goals satisfied and no pending offers — stopping proactive requests.")
            return

        if not goal:
            # Goals met but still waiting on a reply — idle a cycle instead of
            # broadcasting new requests we don't need.
            await asyncio.sleep(45)
            continue

        tradeable = {k: v for k, v in snap["surplus"].items() if v > 0}

        give_str = ", ".join(f"{q} {r}" for r, q in tradeable.items()) or "nada"
        want_str = ", ".join(f"{q} {r}" for r, q in goal.items())
        request_msg = f"Necesito {want_str}. Puedo ofrecer {give_str}. ¿Alguien tiene lo que busco?"
        logger.info(
            f"[PROACTIVE] Requesting {list(goal.keys())} | "
            f"offering {list(tradeable.keys())}"
        )
        emit("proactive", resources=goal, offered=tradeable)
        await _broadcast_and_track(request_msg, we_give=tradeable, we_want=goal)
        await asyncio.sleep(45)


async def _broadcast_and_track(
    request_msg: str,
    we_give: dict[str, int],
    we_want: dict[str, int],
) -> None:
    """
    Fan out a structured request to every active peer in parallel, and
    register a pending offer per peer whose delivery actually succeeded.

    Sequential sends were getting stuck behind any peer that timed out
    (10s × N peers per cycle). With asyncio.gather the wall-clock is the
    slowest single peer, not the sum.
    """
    active = await butler.get_active_agents() or []
    targets: list[str] = []
    skipped_fresh: list[str] = []
    # We consider a pending offer "fresh" if it hasn't been sitting unanswered
    # for at least half its TTL — fresh ones get a grace period so we don't
    # spam the peer with the exact same shape every proactive cycle.
    freshness_window = PENDING_OFFER_TTL / 2
    now = time.time()
    for agent in active:
        alias = agent.get("alias", "")
        ip = agent.get("ip")
        if alias == AGENT_NAME or not ip or ip == "127.0.0.1":
            continue
        pending = state.get_pending_offer(ip)
        if pending is not None and (now - pending.created_at) < freshness_window:
            skipped_fresh.append(ip)
            continue
        targets.append(ip)

    if skipped_fresh:
        logger.info(
            f"[PROACTIVE] Skipping {len(skipped_fresh)} peer(s) with fresh "
            f"pending offers: {skipped_fresh}"
        )

    if not targets:
        return

    results = await asyncio.gather(
        *(agents.send_message_to_agent(ip, request_msg) for ip in targets),
        return_exceptions=True,
    )

    for ip, result in zip(targets, results):
        if isinstance(result, Exception) or result is None:
            # Peer unreachable — skip pending bookkeeping so the reject-loop
            # guard doesn't think we already pitched this shape to them.
            continue
        if we_give and we_want:
            state.set_pending_offer(ip, we_give=we_give, we_want=we_want)
            # Treat proactive requests as our latest decision toward this peer
            # so PDF §2.3 loop-detection covers broadcast paths too.
            state.set_last_decision(
                ip, "counter", we_gave=we_give, we_wanted=we_want
            )
        state.record_turn(ip, "self", request_msg, kind="request")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting agent '{AGENT_NAME}' on port {MY_PORT}")

    logger.info("Registering with Butler...")
    await butler.register_agent()

    info = await butler.get_agent_info()
    if info:
        inventory = (
            info.get("Recursos") or info.get("recursos") or info.get("resources") or {}
        )
        goal = (
            info.get("Objetivo") or info.get("objetivo") or info.get("goal") or {}
        )
        state.initialize(inventory, goal)
        logger.info(f"Agent info: inventory={inventory} | goal={goal}")
    else:
        logger.warning("Could not fetch agent info from Butler — starting with empty state.")

    async def _broadcast():
        active = await butler.get_active_agents()
        logger.info(f"Active agents: {active}")
        snap = state.snapshot()
        tradeable = {k: v for k, v in snap["surplus"].items() if v > 0}
        # Use the canonical "Necesito N r. Puedo ofrecer N r." layout so the
        # fast-path regex in message_normalizer can parse incoming greetings
        # without burning an Ollama round-trip. Quantities on BOTH sides.
        inv_str = ", ".join(f"{q} {r}" for r, q in tradeable.items()) or "nada"
        goal_str = ", ".join(f"{q} {r}" for r, q in snap["goal_needs"].items()) or "nada"
        opening = (
            f"Hola! Soy {AGENT_NAME}. "
            f"Necesito {goal_str}. "
            f"Puedo ofrecer {inv_str}."
        )
        # Opening greeting is a broadcast, not a structured offer — don't
        # store a pending offer for it.
        await agents.broadcast_message(opening)

    asyncio.create_task(_broadcast())
    asyncio.create_task(_proactive_request_loop())
    asyncio.create_task(_sync_from_butler_loop())

    yield

    logger.warning("Agent shutting down.")
    # Give pending /buzon background handlers up to 5s to drain so we don't
    # cut off a half-finished decision (state already mutated, counter
    # message half-sent). Past that, anything still running gets cancelled.
    if _BG_TASKS:
        logger.info(f"Waiting for {len(_BG_TASKS)} background /buzon task(s) to finish.")
        try:
            await asyncio.wait_for(
                asyncio.gather(*_BG_TASKS, return_exceptions=True),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            logger.warning("Shutdown drain timed out — some /buzon tasks may be cancelled.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(lifespan=lifespan)


# Track every background /buzon-handler task so Python's GC can't collect
# them prematurely (asyncio docs: "Save a reference to the result of this
# function, to avoid a task disappearing mid-execution"). Tasks remove
# themselves from the set on completion via add_done_callback.
_BG_TASKS: set[asyncio.Task] = set()

# One asyncio.Lock per peer IP — serializes /buzon handlers for the SAME
# peer while still letting different peers process in parallel. Without
# this, two messages from the same peer arriving within milliseconds
# would race on pending_offer / last_decision writes and cause the agent
# to fire duplicate counters (issue: "对方很短时间内发来多条消息").
_PEER_LOCKS: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


@app.post("/buzon")
async def receive_message(message: IncomingMessage, request: Request):
    """
    Acknowledge the inbound message immediately and process it in the
    background. The full decision pipeline (normalize → branch → LLM →
    counter / clarify / chain second leg) can easily take 5-10s on CPU
    Ollama, and the peer's `send_message_to_agent` would ReadTimeout long
    before we finish. None of the response body fields are ever consumed
    by senders — they only care about HTTP success — so a fast ack is a
    strict upgrade. All side effects (decisions, outbound messages,
    Butler resource transfers) continue to flow through their existing
    out-of-band channels.
    """
    sender_ip = request.client.host
    raw = message.msg
    logger.info(f"[INBOX] {sender_ip}: {raw[:120]}")
    emit("inbox", from_=sender_ip, text=raw[:120])

    task = asyncio.create_task(_handle_inbox(raw, sender_ip))
    _BG_TASKS.add(task)
    task.add_done_callback(_BG_TASKS.discard)

    return {"status": "queued"}


async def _handle_inbox(raw: str, sender_ip: str) -> None:
    """
    Run the original /buzon decision pipeline asynchronously. Exceptions are
    caught and logged so a single bad message can't kill the task tracking
    set or leak as a "Task exception was never retrieved" warning.

    Serialised per sender_ip via _PEER_LOCKS so a burst of messages from the
    same peer is processed one at a time. This prevents two concurrent
    handlers from racing on pending_offer / last_decision writes and from
    spamming the peer with duplicate counters. Different peers are
    unaffected — they still process in parallel.
    """
    async with _PEER_LOCKS[sender_ip]:
        await _handle_inbox_locked(raw, sender_ip)


async def _handle_inbox_locked(raw: str, sender_ip: str) -> None:
    """Original decision pipeline. Called only with the per-peer lock held."""
    try:
        normalized = await normalize(raw, from_agent=sender_ip)
        logger.info(
            f"[NORMALIZED] kind={normalized.kind} | resources={normalized.resources}"
        )

        # PDF §7 step 1: save to conversation history before any branching
        state.record_turn(sender_ip, "peer", raw, kind=normalized.kind)

        # Peer engaged with something concrete — reset the clarify-loop
        # counter so a future ambiguous reply gets the full ladder again
        # (question → re-state → silent) instead of jumping straight to
        # silent.
        if normalized.kind not in ("clarification", "unknown"):
            state.reset_clarify_count(sender_ip)

        # Record what this peer needs and has for chain-trade analysis
        if normalized.kind in ("request", "counter_offer") and normalized.offered_resources:
            state.record_peer_knowledge(
                sender_ip,
                wants=normalized.resources,
                has=normalized.offered_resources,
            )

        if normalized.kind == "request":
            result = await process_request(normalized)
            logger.info(f"[DECISION] {result.decision} | resources={result.resources}")
            emit(
                result.decision,
                to=sender_ip,
                resources=result.resources,
                reason=result.reason,
            )
            return

        if normalized.kind == "counter_offer":
            result = await process_counter_offer(normalized)
            logger.info(f"[COUNTER-IN] decision={result.decision}")
            emit(
                result.decision,
                to=sender_ip,
                resources=result.resources,
                reason=result.reason,
            )
            return

        if normalized.kind == "accept":
            result = await process_accept(normalized)
            logger.info(f"[ACCEPT] decision={result.decision}")
            emit(
                result.decision,
                to=sender_ip,
                resources=result.resources,
                reason=result.reason,
            )
            return

        if normalized.kind == "delivery":
            result = await process_delivery(normalized)
            logger.info(f"[DELIVERY] status={result.status}")
            emit("delivery", from_=sender_ip, resources=normalized.resources)
            return

        if normalized.kind == "clarification":
            await process_clarification(normalized)
            return

        if normalized.kind == "reject":
            logger.info(f"[REJECT] received from {sender_ip} — clearing pending offer.")
            # PDF §2.3 — mark the last outbound offer as rejected so future
            # counter generation won't repeat the same shape.
            state.mark_last_response(sender_ip, "rejected")
            pending = state.get_pending_offer(sender_ip)
            if pending is not None:
                # Free any chain reservation that was tied to this rejected
                # second-leg offer so the intermediate can return to surplus.
                for resource in pending.we_give:
                    state.release_intermediate(resource, reason="peer_rejected_second_leg")
            state.clear_pending_offer(sender_ip)
            return

        logger.warning(f"[UNKNOWN] Unrecognizable message from {sender_ip}")
        if normalized.raw_text and len(normalized.raw_text.strip()) > 10:
            await process_clarification(normalized)
    except Exception as exc:
        logger.exception(
            f"[BUZON-BG] async handler failed for {sender_ip}: {exc}"
        )
        emit("error", from_=sender_ip, message=str(exc))


# ---------------------------------------------------------------------------
# Dashboard endpoints
# ---------------------------------------------------------------------------

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    html = (
        _DASHBOARD_HTML
        .replace("__AGENT_NAME__", AGENT_NAME)
        .replace("__MY_PORT__", str(MY_PORT))
        .replace("__BUTLER_URL__", SERVER_URL)
    )
    return HTMLResponse(html)


@app.get("/api/state")
async def api_state():
    snap = state.snapshot()
    snap["target_resources"] = sorted(snap["target_resources"])
    return snap


@app.get("/api/agents")
async def api_agents():
    return await butler.get_active_agents() or []


@app.get("/api/events")
async def api_events():
    return recent()


@app.get("/api/pending")
async def api_pending():
    """All outstanding offers we are awaiting a peer response on."""
    return state.pending_summary()


@app.get("/api/conversation")
async def api_conversation(peer: str | None = None):
    """Conversation history per peer (or just one peer when ?peer=... is set)."""
    return state.conversation_summary(peer)


@app.get("/api/stream")
async def api_stream():
    async def _generator():
        async for ev in event_stream():
            yield f"data: {json.dumps(ev)}\n\n"
    return StreamingResponse(_generator(), media_type="text/event-stream")


@app.post("/api/send")
async def api_send(payload: dict):
    ip = payload.get("ip", "").strip()
    message = payload.get("message", "").strip()
    if not ip or not message:
        return {"status": "error", "message": "Missing ip or message"}
    result = await agents.send_message_to_agent(ip, message)
    status = "ok" if result is not None else "send_failed"
    if result is not None:
        state.record_turn(ip, "self", message)
    emit("send", to=ip, text=message[:100], status=status)
    return {"status": status, "result": result}


@app.get("/state")
async def get_state():
    snap = state.snapshot()
    snap["target_resources"] = sorted(snap["target_resources"])
    return snap


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=MY_PORT, log_level="warning")
