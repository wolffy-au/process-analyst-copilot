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


class OllamaLLM(LLM):  # type: ignore[misc]
    """Represents a custom LLM for Ollama with context window adjustments.

    Attributes:
        agents_config (Any): Configuration for agents (not used).
        tasks_config (Any): Configuration for tasks (not used).
        num_ctx (int): The maximum context window size.
    """

    num_ctx = 2048

    def get_context_window_size(self) -> int:
        """Returns the adjusted context window size.

        Returns:
            int: The context window size.
        """
        if self.num_ctx > 2048:
            logging.warning(
                """
                Ollama is currently restricted to 2048 context windows by default.
                For larger contexts, use the work-around documented here:
                https://github.com/ollama/ollama/issues/8356#issuecomment-2579221678
                """
            )
        return int(self.num_ctx * 0.75)


def llm_call(llm: LLM, content: str = "Testing 1 2 3!", role: str = "system") -> str:
    """Tests the LLM with a simple prompt.

    Args:
        content (str): The test prompt content. Defaults to "Testing 1 2 3!".
        role (str): The role of the LLM during the interaction. Defaults to "system".

    Returns:
        str: The LLM's response as a string.
    """
    return str(llm.call([{"role": role, "content": content}]))
