import logging
import os
from crewai import LLM


# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Prevent OpenTelemetry errors from logging when behind a firewall
os.environ["OTEL_PYTHON_DISABLED_INSTRUMENTATIONS"] = (
    "azure_sdk,django,fastapi,flask,psycopg2,requests,urllib,urllib3"
)


def llm_call(llm: LLM, content: str = "Testing 1 2 3!", role: str = "user") -> str:
    """Tests the LLM with a simple prompt.

    Args:
        content (str): The test prompt content. Defaults to "Testing 1 2 3!".
        role (str): The role of the LLM during the interaction. Defaults to "system".

    Returns:
        str: The LLM's response as a string.
    """
    return str(llm.call(messages=[{"role": role, "content": content}]))
