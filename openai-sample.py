from typing import Any, Dict
from dotenv import load_dotenv, find_dotenv
from process_analyst_copilot import ClarifyTheAsk

load_dotenv(find_dotenv())

draft_process = ClarifyTheAsk()
# print(draft_process.test_llm())
draft_process.setup()
results: Dict[str, Any] = draft_process.kickoff(
    input_ask="The simplest way to make a cup of tea?"
)
print("See the outputs directory for outputs.")
