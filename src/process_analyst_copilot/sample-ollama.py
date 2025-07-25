from dotenv import load_dotenv, find_dotenv
import os
from crewai import LLM
from process_analyst_copilot import ClarifyTheAsk

load_dotenv(find_dotenv())

llm_model = LLM(
    model="ollama/llama3.1:8b",
    temperature=0.1,
    num_ctx=131072,
)

draft_process = ClarifyTheAsk(llm_model=llm_model)

# FIXME: pydantic_core._pydantic_core.ValidationError: 1 validation error for Crew Value error, Please provide an
# OpenAI API key.
# Need to set Crew() embedder to avoid this error using memory=True on your Crew()
draft_process.embedder = dict(
    provider="ollama",
    config=dict(
        model="nomic-embed-text",
        base_url=os.environ["OLLAMA_API_BASE"],
    ),
)
# This is in addition to the above for PDFReader on non-OpenAI LLMs
# https://docs.crewai.com/tools/pdfsearchtool#custom-model-and-embeddings
draft_process.embedder_llm = dict(
    provider="ollama",  # or google, openai, anthropic, llama2, ...
    config=dict(
        model="llama3.1:8b",
        base_url=os.environ["OLLAMA_API_BASE"],
    ),
)

# print(draft_process.test_llm())
draft_process.setup()
results = draft_process.kickoff(input_ask="The simplest way to make a cup of tea?")
print("See the outputs directory for outputs.")
