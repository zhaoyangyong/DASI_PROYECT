"""
models.py
---------
Internal datatypes shared across the agent.

Design philosophy (see prompt 设计思想):
    Every datatype here serves the "win-the-game" loop: identify intent,
    track pending state, decide whether a step shortens the distance to the
    victory goal, then validate before executing.
"""

from time import time
from typing import Literal, TypedDict
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Wire-level
# ---------------------------------------------------------------------------

class IncomingMessage(BaseModel):
    """Raw payload received on /buzon."""
    msg: str


# ---------------------------------------------------------------------------
# Normalized message + intent
# ---------------------------------------------------------------------------

MessageKind = Literal[
    "request",       # peer wants resources from us
    "delivery",      # peer is sending resources to us
    "accept",        # peer accepts a pending offer
    "reject",        # peer rejects a pending offer
    "counter_offer", # peer is countering with different quantities
    "clarification", # peer is asking us to clarify something
    "unknown",
]


class NormalizedMessage(BaseModel):
    """Unified internal representation of any incoming message."""
    from_agent: str
    kind: MessageKind
    resources: dict[str, int] = {}          # what peer WANTS from us
    offered_resources: dict[str, int] = {}  # what peer is willing to GIVE us
    raw_text: str
    metadata: dict = {}


class IntentResult(BaseModel):
    """
    Output of the intent-recognition prompt.

    The LLM's job is purely to *classify* the intent and extract the resources
    mentioned in it — never to decide a trade. Trade decisions live in the
    evaluation prompt.
    """
    kind: MessageKind
    resources: dict[str, int] = {}
    offered_resources: dict[str, int] = {}
    confidence: Literal["high", "medium", "low"] = "medium"
    reason: str = ""


# ---------------------------------------------------------------------------
# Decision output
# ---------------------------------------------------------------------------

DecisionAction = Literal[
    "accept",   # full grant of the request
    "offer",    # partial grant (subset / lower qty)
    "counter",  # propose a different barter back to the peer
    "reject",   # give nothing
    "clarify",  # ask the peer for clarification before committing
]


class DecisionResponse(BaseModel):
    """Response returned after processing a resource request."""
    decision: DecisionAction
    resources: dict[str, int] = {}          # what we will give them
    counter_request: dict[str, int] = {}    # what we want back (counter only)
    clarify_text: str = ""                  # natural-language ask (clarify only)
    reason: str


class DeliveryResponse(BaseModel):
    """Confirmation returned after processing a delivery."""
    status: Literal["ok", "error"]
    message: str


# ---------------------------------------------------------------------------
# Pending offer + conversation tracking
# ---------------------------------------------------------------------------

class PendingOffer(BaseModel):
    """
    Represents an outstanding barter we proposed to a peer.

    While a pending offer exists with a peer, plain "ok / vale / sí" replies
    can be interpreted as acceptance via rules (PDF §5).
    """
    peer: str
    we_give: dict[str, int] = {}           # what we promised to give
    we_want: dict[str, int] = {}           # what we asked for in return
    created_at: float = Field(default_factory=time)


class ConversationTurn(BaseModel):
    """A single dialogue turn with a peer."""
    role: Literal["peer", "self"]
    text: str
    kind: MessageKind | None = None
    at: float = Field(default_factory=time)


class ChainPlan(BaseModel):
    """
    An intermediate resource we have accepted (or are about to receive) as
    the first leg of a chain trade. Reserved from regular surplus so other
    peers don't accidentally pull it out from under the planned second leg.

    Lifecycle:
        - Created when process_request accepts a non-goal "intermediate" the
          peer is offering, and a chain target peer is known.
        - Cleared on: successful second-leg execution, second-leg send
          failure, peer-B reject, or TTL expiry.
    """
    intermediate: str          # resource we will receive from `from_peer`
    intermediate_qty: int      # how much we earmark for the chain
    from_peer: str             # peer A — delivering the intermediate
    target_peer: str           # peer B — who will receive intermediate in leg 2
    target_resource: str       # goal resource we expect from peer B
    target_qty: int            # how much of target_resource we ask for
    created_at: float = Field(default_factory=time)


class StateSnapshot(TypedDict):
    """
    Frozen view of agent state, returned by `state.snapshot()`.

    Embodies the PDF §2 decision dimensions — what I have, what I still
    need, what I can spare, and how close I am to victory. Every prompt
    and every decision branch reads from exactly these fields, so they
    are documented here as the canonical input contract.
    """
    inventory: dict[str, int]          # resources currently owned
    goal_needs: dict[str, int]         # PDF §3 missing_resources — still needed
    initial_goal: dict[str, int]       # original goal at startup
    target_resources: set[str]         # resources blocked from trading
    surplus: dict[str, int]            # PDF §3 surplus_resources — safe to give
    progress: float                    # 0.0 → 1.0 fraction of goal completed
    near_victory: bool                 # progress ≥ 0.75 — switch to fast mode
    chain_opportunities: dict[str, list[str]]  # intermediate resource -> peers who want it and have goal resources


class LastDecision(BaseModel):
    """
    Record of the most recent decision the agent sent to a peer.

    Used to detect and break PDF §2.3 reject-loops: "if the peer rejects,
    do NOT repeat the same invalid offer — swap resources or lower the
    demand."
    """
    peer: str
    decision: DecisionAction
    we_gave: dict[str, int] = {}
    we_wanted: dict[str, int] = {}
    peer_response: Literal["pending", "accepted", "rejected"] = "pending"
    at: float = Field(default_factory=time)
