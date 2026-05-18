import os

# Butler server
SERVER_URL: str = os.getenv("SERVER_URL", "http://192.168.1.153:7719/")

# Agent identity
AGENT_NAME: str = os.getenv("AGENT_NAME", "FC1111129")
MY_PORT: int = int(os.getenv("MY_PORT", "7720"))

# Ollama 
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_TIMEOUT: float = float(os.getenv("OLLAMA_TIMEOUT", "30.0"))

# General HTTP timeout for agent-to-agent calls (seconds)
HTTP_TIMEOUT: float = float(os.getenv("HTTP_TIMEOUT", "60.0"))

# Shorter timeout used only for Butler startup calls (register, info, peers)
BUTLER_TIMEOUT: float = float(os.getenv("BUTLER_TIMEOUT", "5.0"))

# Pending offer time-to-live (seconds). Pendings older than this are cancelled
# by the proactive loop so the reject-loop guard can propose fresh shapes.
PENDING_OFFER_TTL: float = float(os.getenv("PENDING_OFFER_TTL", "300.0"))

# Chain-trade intermediate reservation time-to-live (seconds). After this an
# unfulfilled chain plan is released so its intermediate flows back to free
# surplus instead of being earmarked forever.
CHAIN_RESERVE_TTL: float = float(os.getenv("CHAIN_RESERVE_TTL", "180.0"))

# Set to true to skip Butler registration and use mock data
LOCAL_TEST_MODE: bool = os.getenv("LOCAL_TEST_MODE", "false").lower() == "true"
