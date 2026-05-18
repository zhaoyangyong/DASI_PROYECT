"""
state_manager.py
----------------
Centralized, concurrency-safe state for the agent.

State surfaces (PDF §2 + §5 + §7):
  - inventory          : resources currently owned
  - goal_needs         : units of each target resource still required
  - target_resources   : resources blocked from trading (still needed for goal)
  - initial_goal       : original goal at startup — used to estimate progress
  - pending_offers     : barter proposals outstanding per peer (PDF §5)
  - conversation       : last N dialogue turns per peer (PDF §7 step 1)
"""

import asyncio
import time
from collections import defaultdict, deque
from loguru import logger

from models import (
    ChainPlan,
    ConversationTurn,
    DecisionAction,
    LastDecision,
    MessageKind,
    PendingOffer,
    StateSnapshot,
)


# Keep at most this many turns per peer — enough for context, small enough
# to fit any LLM prompt window comfortably.
_MAX_HISTORY_PER_PEER = 8


class StateManager:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._inventory: dict[str, int] = {}
        self._goal_needs: dict[str, int] = {}
        self._initial_goal: dict[str, int] = {}
        self._target_resources: set[str] = set()
        self._pending_offers: dict[str, PendingOffer] = {}
        self._conversation: dict[str, deque[ConversationTurn]] = defaultdict(
            lambda: deque(maxlen=_MAX_HISTORY_PER_PEER)
        )
        # PDF §2.3 / §3 last_decision — per-peer signature of our most recent
        # outbound decision, used for reject-loop detection.
        self._last_decisions: dict[str, LastDecision] = {}
        # Chain-trade knowledge: what each peer needs and has, inferred from
        # their incoming messages. Used to evaluate indirect trade paths.
        self._peer_knowledge: dict[str, dict] = {}
        # Chain-trade reservations: an intermediate we accepted (or will
        # receive) is earmarked for a planned second leg. Keyed by the
        # intermediate resource name; subtracts from free surplus until
        # released or expired.
        self._intermediates_pending: dict[str, ChainPlan] = {}
        # Consecutive clarification count per peer — drives the clarify
        # loop guard. Incremented on each clarification we send, reset
        # the moment the peer engages with anything concrete.
        self._clarify_count: defaultdict[str, int] = defaultdict(int)
        # PDF §2.3 reject-loop guard — per-peer window of recent outbound
        # counter shapes. Wider than _last_decisions (one entry) so the
        # guard catches 2-cycles (A/B/A/B), not just immediate repeats.
        # Only confirmed-sent offers land here (see record_outbound_offer).
        self._recent_offer_sigs: defaultdict[str, deque] = defaultdict(
            lambda: deque(maxlen=6)
        )
        # Timestamp of the last speculative counter-request probe we sent
        # to each peer — throttles the stateless speculative path so a
        # rapid-fire peer cannot drive a message storm.
        self._last_speculative_probe: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self, inventory: dict[str, int], goal: dict[str, int]) -> None:
        self._inventory = {
            k: v for k, v in inventory.items() if isinstance(v, (int, float)) and v >= 0
        }
        # initial_goal preserves the absolute target so the dashboard can show
        # received-vs-total. goal_needs is what we still need to ACQUIRE —
        # subtract resources already in our starting inventory.
        self._initial_goal = {
            k: int(v) for k, v in goal.items() if isinstance(v, (int, float)) and v > 0
        }
        self._goal_needs = {
            k: total - self._inventory.get(k, 0)
            for k, total in self._initial_goal.items()
            if total - self._inventory.get(k, 0) > 0
        }
        # target_resources is the PROTECTED set: every resource that appears
        # in the goal, even if we already hold enough — we still need to
        # carry the goal qty to the finish line. Previously this was
        # `goal_needs.keys()`, which silently dropped already-satisfied
        # goals and let them be traded away.
        # FIXME(rename): a follow-up PR will rename this to
        # `_protected_goal_resources` everywhere (and the snapshot key) for
        # clarity. The semantic is fixed here; the name catches up later.
        self._target_resources = set(self._initial_goal.keys())
        logger.info(
            f"State initialized | inventory={self._inventory} "
            f"| initial_goal={self._initial_goal} "
            f"| goal_needs={self._goal_needs} "
            f"| target_resources={sorted(self._target_resources)}"
        )

    # ------------------------------------------------------------------
    # Snapshot — the canonical view used to build prompts (PDF §2)
    # ------------------------------------------------------------------

    def snapshot(self) -> StateSnapshot:
        """
        Return a safe copy of state. `surplus` and `progress` are derived
        here so prompts get the victory-oriented dimensions for free.

        The return type is the canonical `StateSnapshot` TypedDict — every
        prompt and decision branch reads from exactly these fields.
        """
        # Surplus = what we are free to trade away.
        # For a goal-protected resource, only the qty BEYOND the goal target
        # is surplus; we must always hold at least `initial_goal[r]` until
        # the game ends, even if we currently meet the target exactly.
        # Chain-trade reservations are also subtracted from free surplus.
        surplus: dict[str, int] = {}
        for resource, qty in self._inventory.items():
            reserved = self.reserved_intermediate_qty(resource)
            available = qty - reserved
            if available <= 0:
                continue
            if resource in self._target_resources:
                goal_qty = self._initial_goal.get(resource, 0)
                excess = available - goal_qty
                if excess > 0:
                    surplus[resource] = excess
            else:
                surplus[resource] = available

        # Progress in [0, 1]
        total_goal = sum(self._initial_goal.values()) or 1
        total_remaining = sum(self._goal_needs.values())
        progress = max(0.0, min(1.0, 1.0 - (total_remaining / total_goal)))

        # Chain opportunities: non-goal resource -> peers who want it and have goal resources
        chain_opportunities: dict[str, list[str]] = {}
        for peer, knowledge in self._peer_knowledge.items():
            peer_wants = knowledge.get("wants", {})
            peer_has = knowledge.get("has", {})
            if not any(r in self._target_resources for r in peer_has):
                continue
            for r in peer_wants:
                if r not in self._target_resources:
                    chain_opportunities.setdefault(r, []).append(peer)

        return {
            "inventory": dict(self._inventory),
            "goal_needs": dict(self._goal_needs),
            "initial_goal": dict(self._initial_goal),
            "target_resources": set(self._target_resources),
            "surplus": surplus,
            "progress": round(progress, 3),
            "near_victory": progress >= 0.75,
            "chain_opportunities": chain_opportunities,
        }

    # ------------------------------------------------------------------
    # Inventory writes
    # ------------------------------------------------------------------

    async def deduct_resources(self, resources: dict[str, int]) -> bool:
        async with self._lock:
            for resource, qty in resources.items():
                if not isinstance(qty, int) or qty <= 0:
                    logger.warning(f"Deduct rejected: invalid qty for '{resource}': {qty}")
                    return False
                if self._inventory.get(resource, 0) < qty:
                    logger.warning(
                        f"Deduct rejected: insufficient '{resource}' "
                        f"(have {self._inventory.get(resource, 0)}, need {qty})"
                    )
                    return False

            for resource, qty in resources.items():
                self._inventory[resource] -= qty
                # If this deduct fulfils a chain reservation, retire the plan
                # so the reservation doesn't linger and double-block surplus.
                plan = self._intermediates_pending.get(resource)
                if plan is not None:
                    remaining = plan.intermediate_qty - qty
                    if remaining <= 0:
                        del self._intermediates_pending[resource]
                        logger.info(
                            f"[CHAIN] Released reservation on {resource} "
                            "(deduct fulfilled the plan)"
                        )
                    else:
                        self._intermediates_pending[resource] = plan.model_copy(
                            update={"intermediate_qty": remaining}
                        )

            logger.info(f"Deducted {resources} | inventory now: {self._inventory}")
            return True

    async def sync_from_butler(self, butler_inventory: dict[str, int]) -> None:
        async with self._lock:
            for resource, raw_qty in butler_inventory.items():
                butler_qty = int(raw_qty) if isinstance(raw_qty, (int, float)) else 0
                local_qty = self._inventory.get(resource, 0)

                if butler_qty > local_qty:
                    gained = butler_qty - local_qty
                    self._inventory[resource] = butler_qty
                    if resource in self._goal_needs:
                        self._goal_needs[resource] = max(0, self._goal_needs[resource] - gained)
                        if self._goal_needs[resource] == 0:
                            del self._goal_needs[resource]
                            self._target_resources.discard(resource)
                            logger.info(f"[SYNC] Goal satisfied for '{resource}'")
                elif butler_qty < local_qty:
                    self._inventory[resource] = butler_qty

            for resource in list(self._inventory.keys()):
                if resource not in butler_inventory:
                    self._inventory[resource] = 0

            logger.info(
                f"[SYNC] inventory={dict(self._inventory)} | goal_needs={dict(self._goal_needs)}"
            )

    async def add_resources(self, resources: dict[str, int]) -> None:
        async with self._lock:
            for resource, qty in resources.items():
                if not isinstance(qty, int) or qty <= 0:
                    logger.warning(f"Add skipped: invalid qty for '{resource}': {qty}")
                    continue

                self._inventory[resource] = self._inventory.get(resource, 0) + qty

                if resource in self._goal_needs:
                    self._goal_needs[resource] = max(0, self._goal_needs[resource] - qty)
                    if self._goal_needs[resource] == 0:
                        del self._goal_needs[resource]
                        self._target_resources.discard(resource)
                        logger.info(
                            f"Goal satisfied for '{resource}': removed from target_resources"
                        )

            logger.info(
                f"Added {resources} | inventory={self._inventory} "
                f"| goal_needs={self._goal_needs}"
            )

    # ------------------------------------------------------------------
    # Pending offers (PDF §5 — rule-based fast accept)
    # ------------------------------------------------------------------

    def set_pending_offer(
        self,
        peer: str,
        we_give: dict[str, int],
        we_want: dict[str, int],
    ) -> None:
        """Record an offer we sent to `peer` while we await their response."""
        if not peer or peer == "unknown":
            return
        self._pending_offers[peer] = PendingOffer(
            peer=peer, we_give=dict(we_give), we_want=dict(we_want)
        )
        logger.info(
            f"[PENDING] Stored offer for '{peer}': give={we_give} want={we_want}"
        )

    def get_pending_offer(self, peer: str) -> PendingOffer | None:
        return self._pending_offers.get(peer)

    def clear_pending_offer(self, peer: str) -> None:
        if peer in self._pending_offers:
            del self._pending_offers[peer]
            logger.info(f"[PENDING] Cleared offer for '{peer}'")

    def consume_pending(
        self,
        peer: str,
        delivered_to_peer: dict[str, int] | None = None,
        received_from_peer: dict[str, int] | None = None,
    ) -> None:
        """
        Apply a partial settlement to an outstanding pending offer.

        `delivered_to_peer` decrements `we_give` (we just gave them X, owe
        less). `received_from_peer` decrements `we_want` (we just got X,
        expect less). The pending is removed only when BOTH sides reach
        zero — a single-leg delivery doesn't end the obligation, which is
        what previously caused "we gave 4 tela but the peer never recognised
        the queso debt" because process_delivery used to wipe pending
        outright on the first delivery.

        Every call emits a log line showing the trigger and the before→after
        of both sides, so a multi-step trade is easy to follow end to end
        in the agent log.
        """
        delivered = delivered_to_peer or {}
        received = received_from_peer or {}

        pending = self._pending_offers.get(peer)
        if pending is None:
            if delivered or received:
                logger.debug(
                    f"[PENDING] consume noop on '{peer}' (no active pending) | "
                    f"delivered={delivered} received={received}"
                )
            return

        before_give = dict(pending.we_give)
        before_want = dict(pending.we_want)

        new_give = dict(before_give)
        new_want = dict(before_want)
        for r, q in delivered.items():
            if r in new_give:
                new_give[r] = max(0, new_give[r] - q)
        for r, q in received.items():
            if r in new_want:
                new_want[r] = max(0, new_want[r] - q)
        new_give = {r: q for r, q in new_give.items() if q > 0}
        new_want = {r: q for r, q in new_want.items() if q > 0}

        if not new_give and not new_want:
            del self._pending_offers[peer]
            logger.info(
                f"[PENDING] '{peer}' SETTLED | "
                f"delivered={delivered} received={received} | "
                f"we_give {before_give} → {{}} | we_want {before_want} → {{}}"
            )
            return

        self._pending_offers[peer] = pending.model_copy(
            update={"we_give": new_give, "we_want": new_want}
        )
        logger.info(
            f"[PENDING] '{peer}' updated | "
            f"delivered={delivered} received={received} | "
            f"we_give {before_give} → {new_give} | "
            f"we_want {before_want} → {new_want}"
        )

    def expire_stale_pending_offers(self, max_age_seconds: float) -> list[str]:
        """
        Cancel pending offers older than `max_age_seconds`. Returns the list
        of peer IDs whose offers were dropped so callers can react (log,
        re-broadcast, etc.). Also flips the corresponding last_decision
        peer_response to "rejected" so the §2.3 reject-loop guard treats
        the next proposal as a fresh shape.
        """
        if max_age_seconds <= 0 or not self._pending_offers:
            return []
        now = time.time()
        expired = [
            peer for peer, offer in self._pending_offers.items()
            if now - offer.created_at > max_age_seconds
        ]
        for peer in expired:
            del self._pending_offers[peer]
            last = self._last_decisions.get(peer)
            if last is not None and last.peer_response == "pending":
                self._last_decisions[peer] = last.model_copy(
                    update={"peer_response": "rejected"}
                )
        if expired:
            logger.info(f"[PENDING] Expired stale offers from: {expired}")
        return expired

    # ------------------------------------------------------------------
    # Conversation history (PDF §7 step 1)
    # ------------------------------------------------------------------

    def record_turn(
        self,
        peer: str,
        role: str,
        text: str,
        kind: MessageKind | None = None,
    ) -> None:
        """Append a single dialogue turn for `peer` (capped FIFO)."""
        if not peer or peer == "unknown":
            return
        if role not in ("peer", "self"):
            return
        self._conversation[peer].append(
            ConversationTurn(role=role, text=text[:200], kind=kind)
        )

    def get_history(self, peer: str) -> list[ConversationTurn]:
        return list(self._conversation.get(peer, []))

    # ------------------------------------------------------------------
    # Last decision tracking (PDF §2.3, §3 last_decision)
    # ------------------------------------------------------------------

    def set_last_decision(
        self,
        peer: str,
        decision: DecisionAction,
        we_gave: dict[str, int] | None = None,
        we_wanted: dict[str, int] | None = None,
    ) -> None:
        """Record the most recent decision we sent to `peer`."""
        if not peer or peer == "unknown":
            return
        self._last_decisions[peer] = LastDecision(
            peer=peer,
            decision=decision,
            we_gave=dict(we_gave or {}),
            we_wanted=dict(we_wanted or {}),
        )

    def get_last_decision(self, peer: str) -> LastDecision | None:
        return self._last_decisions.get(peer)

    def mark_last_response(self, peer: str, response: str) -> None:
        """Update the peer_response field on the last decision (PDF §2.3)."""
        last = self._last_decisions.get(peer)
        if last is None or response not in ("pending", "accepted", "rejected"):
            return
        self._last_decisions[peer] = last.model_copy(update={"peer_response": response})

    def record_peer_knowledge(
        self,
        peer: str,
        wants: dict[str, int],
        has: dict[str, int],
    ) -> None:
        """Record what a peer needs and has, inferred from their trade messages."""
        if not peer or peer == "unknown":
            return
        self._peer_knowledge[peer] = {"wants": dict(wants), "has": dict(has)}

    def find_chain_opportunity(
        self,
        intermediate: dict[str, int],
        target_resources: set[str],
    ) -> bool:
        """
        Return True if any known peer would accept resources from `intermediate`
        AND has at least one of our target resources to offer in return.

        This detects indirect trade paths: accept a non-goal resource now,
        then use it to acquire a goal resource from a different peer.
        """
        for knowledge in self._peer_knowledge.values():
            peer_wants = knowledge.get("wants", {})
            peer_has = knowledge.get("has", {})
            wants_intermediate = any(r in peer_wants for r in intermediate)
            has_goal = any(r in target_resources for r in peer_has)
            if wants_intermediate and has_goal:
                return True
        return False

    def find_chain_target(
        self,
        intermediate_resources: dict[str, int],
        target_resources: set[str],
    ) -> tuple[str, str, int, str, int] | None:
        """
        Return the best chain match for the resources peer A is offering:
            (target_peer_ip, intermediate, intermediate_qty,
             goal_resource, goal_qty)

        Or None when no known peer simultaneously wants any of the offered
        intermediates AND holds any of our target resources.
        """
        for peer_ip, knowledge in self._peer_knowledge.items():
            peer_wants = knowledge.get("wants", {})
            peer_has = knowledge.get("has", {})
            for r, offered_qty in intermediate_resources.items():
                if offered_qty <= 0 or r not in peer_wants:
                    continue
                for goal_r, goal_qty in peer_has.items():
                    if goal_r in target_resources and goal_qty > 0:
                        return (
                            peer_ip,
                            r,
                            int(min(offered_qty, peer_wants[r])),
                            goal_r,
                            int(goal_qty),
                        )
        return None

    # ------------------------------------------------------------------
    # Chain trade intermediate reservations
    # ------------------------------------------------------------------

    def reserved_intermediate_qty(self, resource: str) -> int:
        """Quantity of `resource` currently earmarked for an in-flight chain."""
        plan = self._intermediates_pending.get(resource)
        return plan.intermediate_qty if plan else 0

    def reserve_intermediate(self, plan: ChainPlan) -> None:
        """
        Earmark a quantity of an intermediate resource for a planned chain
        second-leg. Last-write-wins on the resource key — if a fresh chain
        accept supersedes an older plan, the new target takes priority.
        """
        self._intermediates_pending[plan.intermediate] = plan
        logger.info(
            f"[CHAIN] Reserved {plan.intermediate_qty} {plan.intermediate} "
            f"for second leg to {plan.target_peer} "
            f"(want {plan.target_qty} {plan.target_resource})"
        )

    def release_intermediate(self, resource: str, reason: str = "released") -> None:
        if resource in self._intermediates_pending:
            del self._intermediates_pending[resource]
            logger.info(f"[CHAIN] Released reservation on {resource} ({reason})")

    # ------------------------------------------------------------------
    # Clarify-loop guard
    # ------------------------------------------------------------------

    def bump_clarify_count(self, peer: str) -> int:
        """Increment and return the consecutive clarification count for peer."""
        if not peer or peer == "unknown":
            return 0
        self._clarify_count[peer] += 1
        return self._clarify_count[peer]

    def reset_clarify_count(self, peer: str) -> None:
        """Forget the clarify-loop counter for peer (call when peer engages)."""
        if peer in self._clarify_count:
            del self._clarify_count[peer]

    def expire_stale_intermediates(self, max_age_seconds: float) -> list[str]:
        """Drop chain reservations older than max_age_seconds."""
        if max_age_seconds <= 0 or not self._intermediates_pending:
            return []
        now = time.time()
        expired = [
            r for r, plan in self._intermediates_pending.items()
            if now - plan.created_at > max_age_seconds
        ]
        for r in expired:
            del self._intermediates_pending[r]
        if expired:
            logger.info(f"[CHAIN] Expired stale intermediates: {expired}")
        return expired

    def is_repeat_offer(
        self,
        peer: str,
        we_give: dict[str, int],
        we_want: dict[str, int],
    ) -> bool:
        """
        True iff this exact `we_give` / `we_want` shape appears anywhere in
        our recent confirmed-sent counter window for `peer`. PDF §2.3
        reject-loop guard.

        Window-based (not just the single last decision) so an A/B/A/B
        oscillation is caught — comparing against only the last offer lets
        a 2-cycle slip through forever.

        Short-circuits to False if the peer accepted our most recent
        decision: a fresh round after a successful trade is not a loop.
        """
        last = self._last_decisions.get(peer)
        if last is not None and last.peer_response == "accepted":
            return False
        sig = (frozenset(we_give.items()), frozenset(we_want.items()))
        return sig in self._recent_offer_sigs.get(peer, ())

    def record_outbound_offer(
        self,
        peer: str,
        we_give: dict[str, int],
        we_want: dict[str, int],
    ) -> None:
        """
        Log a counter shape into the §2.3 repeat-detection window.

        MUST be called only after agents.send_message_to_agent succeeded —
        recording a shape that never left the process would make the guard
        block a later genuine send of the same shape ("phantom offer").
        """
        if not peer or peer == "unknown":
            return
        sig = (frozenset(we_give.items()), frozenset(we_want.items()))
        self._recent_offer_sigs[peer].append(sig)

    def mark_speculative_probe(self, peer: str) -> None:
        """
        Stamp the time of a confirmed-sent speculative counter-request.

        Call only after the send succeeded — a failed send puts nothing on
        the wire and so must not consume the throttle budget.
        """
        if not peer or peer == "unknown":
            return
        self._last_speculative_probe[peer] = time.time()

    def seconds_since_speculative_probe(self, peer: str) -> float | None:
        """Seconds since the last speculative probe to `peer`, or None."""
        ts = self._last_speculative_probe.get(peer)
        return None if ts is None else time.time() - ts

    # ------------------------------------------------------------------
    # Dashboard helpers — JSON-safe summaries
    # ------------------------------------------------------------------

    def pending_summary(self) -> list[dict]:
        """Return all pending offers as JSON-serializable dicts."""
        return [
            {
                "peer": offer.peer,
                "we_give": dict(offer.we_give),
                "we_want": dict(offer.we_want),
                "created_at": offer.created_at,
            }
            for offer in self._pending_offers.values()
        ]

    def conversation_summary(self, peer: str | None = None) -> dict:
        """
        Return conversation history.

        If `peer` is given, returns just that peer's turns.
        Otherwise returns {peer: [turns]} for every peer that has activity.
        """
        def _serialize(turns) -> list[dict]:
            return [
                {"role": t.role, "text": t.text, "kind": t.kind, "at": t.at}
                for t in turns
            ]

        if peer is not None:
            return {peer: _serialize(self._conversation.get(peer, []))}
        return {p: _serialize(turns) for p, turns in self._conversation.items()}


# Module-level singleton — all other modules import this instance
state = StateManager()
