from process_analyst_copilot import ClarifyTheAsk
from process_analyst_copilot.utils import OllamaLLM

# FIXME Ollama is currently restricted to 2048 context windows
# Workaround found here which is why this is editable
# https://github.com/ollama/ollama/issues/8356#issuecomment-2579221678
llm_model = OllamaLLM(
    model="ollama/llama3.1:8b",
    temperature=0.3,
    base_url="http://localhost:11434",
)
llm_model.num_ctx = 2048

draft_process = ClarifyTheAsk(llm_model=llm_model)

# FIXME: pydantic_core._pydantic_core.ValidationError: 1 validation error for Crew Value error, Please provide an
# OpenAI API key.
# Need to set Crew() embedder to avoid this error using memory=True on your Crew()
draft_process.embedder = dict(
    provider="ollama",
    config=dict(
        model="nomic-embed-text",
    ),
)
# This is in addition to the above for PDFReader on non-OpenAI LLMs
# https://docs.crewai.com/tools/pdfsearchtool#custom-model-and-embeddings
draft_process.embedder_llm = dict(
    provider="ollama",  # or google, openai, anthropic, llama2, ...
    config=dict(
        model="llama3.1:8b",
    ),
)

# print(draft_process.test_llm())
draft_process.setup()
results = draft_process.kickoff(input_ask="The simplest way to make a cup of tea?")
print("See the outputs directory for outputs.")
