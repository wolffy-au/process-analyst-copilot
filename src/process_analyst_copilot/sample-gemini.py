from crewai import LLM
from dotenv import load_dotenv, find_dotenv
from process_analyst_copilot import ClarifyTheAsk

load_dotenv(find_dotenv())

llm_model = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.3,
)

draft_process = ClarifyTheAsk(llm_model=llm_model)

# Need to set Crew() embedder to avoid this error using memory=True on your Crew()
# draft_process.embedder = dict(
#     provider="google",
#     config=dict(
#         model="gemini-embedding-001",
#     ),
# )
# This is in addition to the above for PDFReader on non-OpenAI LLMs
# https://docs.crewai.com/tools/pdfsearchtool#custom-model-and-embeddings
# draft_process.embedder_llm = dict(
#     provider="gemini",
#     config=dict(
#         model="gemini-2.0-flash",
#     ),
# )

# print(draft_process.test_llm())
draft_process.setup()
results = draft_process.kickoff(input_ask="The simplest way to make a cup of tea?")
print("See the outputs directory for outputs.")
