from typing import Any, Dict
from process_analyst_copilot import ClarifyTheAsk, OllamaLLM

if __name__ == "__main__":
    llm_model = OllamaLLM(
        model="ollama/llama3.1:8b",
        temperature=0.3,
        api_base="http://localhost:11434",
    )
    llm_model.num_ctx = 2048

    draft_process = ClarifyTheAsk(llm_model=llm_model)
    draft_process.setup()
    results: Dict[str, Any] = draft_process.kickoff(
        input_ask="How do I make a good cup of tea?"
    )
    print("See the outputs directory for outputs.")
