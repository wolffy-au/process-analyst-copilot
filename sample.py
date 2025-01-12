from process_analyst_copilot import ClarifyTheAsk

if __name__ == "__main__":
    draft_process = ClarifyTheAsk(
        llm_model="ollama/llama3.1:8b",
    )
    draft_process.setup()
    results = draft_process.kickoff(input_ask="How do I make a good cup of tea?")
    print("See the outputs directory for outputs.")
