import httpx
from loguru import logger

from config import SERVER_URL, AGENT_NAME, HTTP_TIMEOUT, BUTLER_TIMEOUT


async def register_agent() -> int | None:
    """Registers the agent with Butler and returns the HTTP status code."""
    async with httpx.AsyncClient(timeout=BUTLER_TIMEOUT) as client:
        try:
            response = await client.post(SERVER_URL + "alias/" + AGENT_NAME)
            response.raise_for_status()
            logger.success(f"Registered as '{AGENT_NAME}' | Status: {response.status_code}")
            return response.status_code
        except httpx.RequestError as error:
            logger.error(f"Failed to register agent: {error}")
            return None


async def get_agent_info() -> dict | None:
    """Fetches current inventory and objective from Butler."""
    async with httpx.AsyncClient(timeout=BUTLER_TIMEOUT) as client:
        try:
            response = await client.get(SERVER_URL + "info")
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as error:
            logger.error(f"Failed to fetch agent info: {error}")
            return None


async def get_active_agents() -> list | None:
    """Fetches active agents list (alias + IP) from Butler."""
    async with httpx.AsyncClient(timeout=BUTLER_TIMEOUT) as client:
        try:
            response = await client.get(SERVER_URL + "gente")
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as error:
            logger.error(f"Failed to fetch active agents: {error}")
            return None


async def send_resources(target_alias: str, resource_package: dict) -> dict | None:
    """Sends a resource package to another agent via Butler."""
    logger.info(f"Sending {resource_package} to '{target_alias}'")
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        try:
            response = await client.post(
                SERVER_URL + "paquete/" + target_alias,
                json=resource_package,
            )
            response.raise_for_status()
            result = response.json()
            logger.success(f"Package sent to '{target_alias}': {result}")
            return result
        except httpx.RequestError as error:
            logger.error(f"Failed to send resources to '{target_alias}': {error}")
            return None


async def get_alias_for_ip(ip: str) -> str | None:
    """Resolve an agent's IP address to its registered alias."""
    agents = await get_active_agents()
    if not agents:
        return None
    for agent in agents:
        if agent.get("ip") == ip:
            return agent.get("alias")
    return None
