import asyncio

import httpx
from loguru import logger

import butler
from config import AGENT_NAME, MY_PORT, HTTP_TIMEOUT


async def send_message_to_agent(agent_ip: str, message: str) -> dict | None:
    """
    Send a message directly to another agent's /buzon endpoint.

    Retries once after 1 second on transport or HTTP errors to absorb the
    common case of a transient network blip; gives up after the second
    failure to avoid backlogging the proactive loop.
    """
    if agent_ip in ("127.0.0.1", "localhost", "::1"):
        logger.warning(f"Refusing to send message to self ({agent_ip}) — skipping.")
        return None
    url = f"http://{agent_ip}:{MY_PORT}/buzon"
    payload = {"msg": message}
    logger.debug(f"Sending to {url}: {message[:80]}")

    last_error: Exception | None = None
    for attempt in (1, 2):
        try:
            async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                logger.success(f"Message sent to {agent_ip}")
                return response.json()
        except (httpx.RequestError, httpx.HTTPStatusError) as error:
            last_error = error
            if attempt == 1:
                logger.warning(
                    f"Send to {agent_ip} failed ({type(error).__name__}: "
                    f"{error!r}) — retrying in 1s."
                )
                await asyncio.sleep(1)

    logger.error(
        f"Failed to send message to {agent_ip} after retry "
        f"({type(last_error).__name__}: {last_error!r})"
    )
    return None


async def broadcast_message(message: str) -> None:
    """
    Concurrently send `message` to every active peer except ourselves.

    Uses `asyncio.gather` so one slow / unreachable peer does not block the
    rest. Exceptions inside individual sends are absorbed by
    `return_exceptions=True` — `send_message_to_agent` already returns None
    for failures, so the caller just gets a fire-and-forget broadcast.
    """
    agents_list = await butler.get_active_agents()
    if not agents_list:
        logger.warning("No active agents found for broadcasting.")
        return

    tasks = []
    for agent in agents_list:
        alias = agent.get("alias", "")
        ip = agent.get("ip")
        if alias == AGENT_NAME or not ip or ip in ("127.0.0.1", "localhost", "::1"):
            continue
        logger.info(f"Broadcasting to '{alias}' ({ip})")
        tasks.append(send_message_to_agent(ip, message))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
