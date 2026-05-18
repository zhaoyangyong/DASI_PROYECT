"""
events.py  —  lightweight in-process event bus for the real-time dashboard.
"""
import asyncio
from collections import deque
from datetime import datetime

_log: deque = deque(maxlen=500)
_queues: list[asyncio.Queue] = []


def emit(kind: str, **data) -> None:
    """Record an event and push it to all active SSE subscribers."""
    event = {"kind": kind, "ts": datetime.now().strftime("%H:%M:%S"), **data}
    _log.append(event)
    dead = []
    for q in _queues:
        try:
            q.put_nowait(event)
        except Exception:
            dead.append(q)
    for q in dead:
        try:
            _queues.remove(q)
        except ValueError:
            pass


def recent() -> list:
    return list(_log)


async def stream():
    """Async generator that yields events as they arrive (for SSE)."""
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    _queues.append(q)
    try:
        while True:
            yield await q.get()
    finally:
        try:
            _queues.remove(q)
        except ValueError:
            pass
