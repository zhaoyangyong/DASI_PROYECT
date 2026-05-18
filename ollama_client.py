"""
ollama_client.py
----------------
Async HTTP client for the Ollama REST chat API using function/tool calling.

Each LLM task in this project is expressed as a `tool` — the model is asked
to call exactly one tool and its `arguments` are returned as a parsed dict.
Returns None on any failure so callers can fall back gracefully.
"""

import json

import httpx
from loguru import logger

from config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT


async def call_ollama_tool(prompt: str, tool: dict) -> dict | None:
    """
    Send `prompt` to Ollama's chat endpoint with `tool` declared, and return
    the parsed `arguments` dict of the resulting tool call.

    Returns None when Ollama times out, is unreachable, fails to produce a
    tool call, calls a different tool, or returns arguments that aren't a
    dict.
    """
    url = f"{OLLAMA_URL}/api/chat"
    expected_name = tool["function"]["name"]
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [tool],
        "stream": False,
    }

    logger.info(
        f"Calling Ollama (model='{OLLAMA_MODEL}', tool='{expected_name}')..."
    )

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
        data = response.json()
    except httpx.TimeoutException:
        logger.warning("Ollama request timed out — activating fallback.")
        return None
    except httpx.RequestError as exc:
        logger.warning(f"Ollama unreachable: {exc} — activating fallback.")
        return None
    except Exception as exc:
        logger.error(f"Unexpected Ollama error: {exc} — activating fallback.")
        return None

    message = data.get("message") or {}
    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        logger.warning(
            f"Ollama returned no tool call (expected '{expected_name}')."
        )
        return None

    fn = (tool_calls[0] or {}).get("function") or {}
    name = fn.get("name")
    args = fn.get("arguments")

    if name and name != expected_name:
        logger.warning(
            f"Ollama called unexpected tool '{name}' (expected '{expected_name}')."
        )
        return None

    # Ollama usually returns arguments pre-parsed as a dict, but some builds
    # still return a JSON string — handle both.
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Ollama tool arguments were a non-JSON string.")
            return None

    if not isinstance(args, dict):
        logger.warning("Ollama tool arguments were not an object.")
        return None

    logger.info(f"Ollama tool '{expected_name}' returned successfully.")
    return args
