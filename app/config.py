"""
Configuration management for the AI Runtime.
Loads environment variables and application settings.
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load .env file from the ai-runtime directory
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    loaded = load_dotenv(dotenv_path=env_path, override=False)
    if loaded:
        import logging

        logger = logging.getLogger(__name__)
        logger.info(f"Loaded .env file from: {env_path.absolute()}")
else:
    import logging

    logger = logging.getLogger(__name__)
    logger.warning(f".env file not found at: {env_path.absolute()}")

# Kafka Configuration
KAFKA_BROKERS: List[str] = os.getenv("KAFKA_BROKERS", "localhost:9092").split(",")
KAFKA_CONSUMER_GROUP: str = os.getenv("KAFKA_CONSUMER_GROUP", "ai-runtime-consumer")

# LangSmith Configuration
LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "symvion-ai")
LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "true").lower() == "true"

# Application Configuration
APP_PORT: int = int(os.getenv("APP_PORT", "8000"))
APP_ENV: str = os.getenv("APP_ENV", "development")

# LLM Configuration
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
