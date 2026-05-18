# fdi-dasi-jackson

## Team

* Zhaoyang Qi
* Jingyuan Wang

---

## What this project does

An autonomous resource-trading agent that competes with other agents in a multi-agent
barter game, coordinated through a central **Butler** server. The agent registers,
loads its inventory and victory goal from Butler, and then engages in continuous
negotiation with peers — broadcasting needs, evaluating incoming offers, countering
when terms aren't acceptable, and chaining intermediate trades to reach the goal
faster.

**Architecture in one line:** the LLM (Ollama, via function-calling tools) suggests
intent classification and strategy; code enforces every constraint and executes
all state transitions.

---

## Prerequisites

- [`uv`](https://github.com/astral-sh/uv) — Python package manager
- [`ollama`](https://ollama.com) — local LLM runtime
- An Ollama model with **tool-calling** support (we use `llama3.1` by default;
  `qwen2.5:7b-instruct` is a faster alternative)

---

## Setup

```bash
# 1. Install Python deps
uv sync --extra dev

# 2. Pull the LLM model (one-time)
ollama pull llama3.1
```

---

## Running

```bash
# Start the Butler server (provided separately — fdi-pln-butler)
fdi-pln-butler

# Start one agent
uv run main.py
```

For multi-agent local testing on the same host, give each agent a unique port
and alias:

```bash
MY_PORT=7720 AGENT_NAME=AgentA uv run main.py
MY_PORT=7721 AGENT_NAME=AgentB uv run main.py
```

For two-machine setups, ensure each agent points `SERVER_URL` at the LAN IP of
the Butler host (not `127.0.0.1`), so Butler records each agent under its
reachable address.

---

## Configuration

All values are environment-variable-overridable. Most useful ones:

| Variable            | Default                       | Description                                         |
|---------------------|-------------------------------|-----------------------------------------------------|
| `SERVER_URL`        | `http://192.168.1.153:7719/`  | Butler base URL — set to Butler host's LAN IP       |
| `AGENT_NAME`        | `FC1111129`                   | Alias registered with Butler                        |
| `MY_PORT`           | `7720`                        | Port this agent listens on for `/buzon`             |
| `OLLAMA_MODEL`      | `llama3.1`                    | Ollama model name (must support tool calling)       |
| `OLLAMA_URL`        | `http://localhost:11434`      | Ollama server URL                                   |
| `OLLAMA_TIMEOUT`    | `30.0`                        | Seconds before giving up on an Ollama call          |
| `HTTP_TIMEOUT`      | `60.0`                        | Seconds for agent-to-agent HTTP                     |
| `BUTLER_TIMEOUT`    | `5.0`                         | Seconds for Butler startup calls                    |
| `PENDING_OFFER_TTL` | `300.0`                       | Seconds until an unanswered offer is auto-cancelled |
| `CHAIN_RESERVE_TTL` | `180.0`                       | Seconds an unfulfilled chain-trade plan is held     |
| `LOCAL_TEST_MODE`   | `false`                       | Skip Butler registration (mock data)                |

---

## Module layout

```
main.py                FastAPI app, lifespan, /buzon endpoint, dashboard routes
agents.py              Agent-to-agent HTTP client (concurrent broadcast)
butler.py              Butler HTTP client (register, info, peer list, deliveries)
config.py              Centralised env-var configuration
decision_engine.py     Core business logic: process_request, counter, accept, chain
events.py              In-memory pub/sub for dashboard /api/stream
message_normalizer.py  JSON / NL → NormalizedMessage (regex fast path + Ollama)
messaging.py           Outbound structured-message builders
models.py              Pydantic schemas (NormalizedMessage, ChainPlan, etc.)
ollama_client.py       Async Ollama chat client with function-calling tools
prompt_builder.py      All Ollama prompts and tool schemas
state_manager.py       Async-safe state: inventory, goals, pendings, chain plans
utils.py               Shared helpers (resource whitelist, JSON parsing)
dashboard.html         Embedded SPA served at /dashboard
```

---

## Decision pipeline

```
HTTP POST /buzon
   │
   │   (1) instant ack — return {"status": "queued"} in <50ms
   ▼
asyncio.create_task(_handle_inbox)
   │
   ▼ (background)
message_normalizer.normalize()
   ├─ JSON parse short-circuit
   ├─ Regex fast path:    "Necesito X. Puedo ofrecer Y."        → request
   │                      "Te ofrezco X a cambio de Y."          → counter_offer
   │                      "Te envío X."                          → delivery
   │                      "¿Puedes enviarme X, como acordamos?"  → request (honour claim)
   │                      "ok" / "vale" / "sí" …                 → accept / clarification
   └─ Fallback: Ollama `classify_intent` tool
   ▼
NormalizedMessage { kind, resources, offered_resources, ... }
   │
   ├── request        → decision_engine.process_request()
   │     0. Honour-pending fast path: if the peer simply claims what we
   │        already promised, settle it now and skip the LLM entirely
   │     1. Snapshot state (inventory, goal_needs, surplus, chain_opportunities)
   │     2. Split forbidden vs exchangeable (target-resource protection,
   │        intermediate-reservation aware)
   │     3. Early-reject branches each TRY A COUNTER FIRST (see below)
   │     4. Otherwise run Ollama `evaluate_trade` tool
   │     5. Validate every field of the LLM output; rule-based fallback on fail
   │     6. Execute decision atomically; trigger chain second leg if applicable
   │
   ├── counter_offer → process_counter_offer()  runs the request flow, then may
   │                   generate a tighter re-counter via the Ollama counter tool
   ├── accept    → process_accept()      honour pending, deliver, request goal back
   ├── delivery  → process_delivery()    add to inventory, update goal_needs
   ├── clarify   → process_clarification() send a Spanish question
   └── reject    → release chain reservation, clear pending, mark response
```

---

## Counter-before-reject

A peer's request never falls through to a silent reject without first attempting
a counter-offer. `_counter_request_if_valuable` picks one of three strategies:

| Strategy        | When                                                    | Shape                                          |
|-----------------|---------------------------------------------------------|------------------------------------------------|
| **concrete**    | Peer's offer already includes a goal resource           | Match their offered qty, give full surplus     |
| **chain**       | Peer's offer is an intermediate a known peer wants for a goal resource | Single surplus item × single intermediate, capped at 3 |
| **speculative** | Peer's offer contains nothing useful                    | Single surplus × single goal_need, capped at 3 |

If the same counter shape would repeat against the same peer, the reject-loop
guard sweetens the next attempt: drop the largest `want` by 1 first, otherwise
bump the smallest `give` by 1. Once both directions are exhausted, the counter
is dropped and the conversation truly ends in reject.

---

## Chain trades

When a peer offers a non-goal resource that a third party would accept in
exchange for a goal resource, the agent automatically:

1. **Reserves** the incoming intermediate via `state.reserve_intermediate(plan)`
   so other peers can't pull it from free surplus.
2. **Fires** an immediate second-leg counter-offer to the third party.
3. **Releases** the reservation when one of: deduct fulfils the plan, second-leg
   is rejected, second-leg send fails, or `CHAIN_RESERVE_TTL` elapses.

The reservation is subtracted from `surplus` and from `_split_exchangeable`'s
inventory cap, so it cannot be accidentally traded away on another track.

---

## Proactive broadcast

Every 45 s the agent broadcasts its current needs and surplus:

```
Necesito 3 queso, 5 aceite. Puedo ofrecer 1 queso, 3 tela, 3 aceite, 16 oro.
```

Peers with a "fresh" pending offer (younger than `PENDING_OFFER_TTL / 2`) are
skipped that cycle, so a single peer isn't spammed with the same shape.

Broadcasts go out **concurrently** (`asyncio.gather`) so one unreachable peer
doesn't block the rest. Pending bookkeeping is recorded only for peers whose
delivery actually succeeded.

---

## Dashboard

Open `http://localhost:7720/dashboard` after starting the agent. Single-page UI
with:

- **Inventory & Goal panels** — live counts with progress bars
- **Pending offers** — what we currently owe to whom
- **Conversation modal per peer** — full turn-by-turn history
- **Event feed (SSE)** — real-time stream of inbox / decision / delivery /
  pending_expired / chain_expired / error events
- **Manual send box** — send a one-off message to any peer

---

## API endpoints

| Method | Path                | Purpose                                         |
|--------|---------------------|-------------------------------------------------|
| `POST` | `/buzon`            | Receive a message (ack only — async processing) |
| `GET`  | `/dashboard`        | SPA dashboard                                   |
| `GET`  | `/api/state`        | Inventory, goals, surplus, progress snapshot    |
| `GET`  | `/api/agents`       | Active peer list (proxied from Butler)          |
| `GET`  | `/api/events`       | Recent dashboard events (buffer)                |
| `GET`  | `/api/stream`       | Server-Sent Events stream of live events        |
| `GET`  | `/api/pending`      | All outstanding offers                          |
| `GET`  | `/api/conversation` | Full per-peer history (`?peer=<ip>` to filter)  |
| `POST` | `/api/send`         | Manually send a message to a peer               |
| `GET`  | `/state`            | Same as `/api/state` (legacy alias)             |

### `/buzon` payload

```json
{ "msg": "<JSON string or Spanish natural-language text>" }
```

The agent ack's with `{"status": "queued"}` immediately and processes in
the background. All decisions, counters, and resource deliveries flow back
out through separate channels (peer-to-peer `/buzon` calls + Butler
`paquete` deliveries).

### Recognised JSON inbound

```json
{
  "kind": "request" | "delivery" | "accept" | "reject" | "counter_offer",
  "resources": { "arroz": 2, "madera": 1 },
  "offered_resources": { "vino": 1 },
  "from_agent": "FCxxx"
}
```

### Recognised natural-language formats (fast-path regex)

| Format                                                  | Parsed as                       |
|---------------------------------------------------------|---------------------------------|
| `Necesito 3 queso, 5 aceite. Puedo ofrecer 4 tela.`      | `request`                       |
| `Te ofrezco 3 arroz a cambio de 2 queso.`                | `counter_offer`                 |
| `Puedo ofrecer 3 arroz a cambio de 2 queso.`             | `counter_offer`                 |
| `Te doy 3 arroz a cambio de 2 queso.`                    | `counter_offer`                 |
| `Te envío 2 piedras ahora.` / `Te mando 2 piedras.`      | `delivery`                      |
| `¿Puedes enviarme 2 piedra, como acordamos?`             | `request` (honour claim)        |
| `ok` / `vale` / `sí` / `acepto` / `trato` …              | `accept` if pending, else `clarification` |

The **honour claim** format is the exact sentence the agent itself emits to
collect on an already-agreed trade (`decision_engine._generate_trade_message`,
want-only branch). Parsing it deterministically here — rather than relying on
the LLM — guarantees the second leg of a barter routes to `process_request`'s
honour-pending fast path and the trade closes cleanly. A bare-affirmation
("ok") is only an `accept` when an offer to that peer is actually pending;
otherwise it is treated as `clarification`.

Anything else falls through to the Ollama `classify_intent` tool. If Ollama
is unreachable or returns `unknown`, the message is conservatively re-classified
as `clarification` so the peer is asked for clarification instead of being
silently ignored.

---

## State invariants enforced by code (not by LLM)

- **Goal-reserved resources** are never traded below the still-needed quantity.
- **Resource whitelist** is enforced in tool schemas via JSON Schema
  `propertyNames: { enum: [...] }`, so the LLM cannot invent resources.
- **Chain reservations** are subtracted from `surplus` and from
  `exchangeable` caps — no double-spending.
- **Inventory** never goes negative (atomic deduct under `asyncio.Lock`).
- **Pending offers** that fail to send at the HTTP layer are never recorded;
  the reject-loop guard sees only shapes that actually went out.
- **Reject responses from peers** flip the corresponding `peer_response` to
  `"rejected"` and release chain reservations tied to that pending offer.

---

## Troubleshooting

### `ConnectTimeout` to a peer's IP

Peer's machine is dropping packets (firewall) or that IP is stale in Butler:

```bash
ping <ip>           # L3 reachability
nc -zv <ip> 7720    # L4 reachability
```

- `ping` works, `nc` times out → firewall on the peer (Windows Defender, UFW)
- Both fail → wrong subnet / VMware NAT / AP isolation
- `nc` returns `Connection refused` → peer's agent isn't running

### `Ollama returned no tool call` warnings

The model occasionally fails to emit a tool call. The fast-path regex now
covers ~80% of standard messages so this is rarer; for the rest the agent
falls back to a clarification request. Switching to `qwen2.5:7b-instruct`
gives more consistent tool-calling than `llama3.1`.

### `ReadTimeout` on outbound sends

The peer's `/buzon` is slow to respond. Should be uncommon now that `/buzon`
ack's in <50 ms; if you still see it, the peer is running an old synchronous
version of `/buzon`. They should also be on the latest code.

### Same peer's IP keeps showing in Butler after they restarted

Butler persists registrations in `estado_butler.json`. Stop Butler, delete
the file, restart — peers will re-register cleanly.
