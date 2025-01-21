from behave import given, when, then  # type: ignore[reportAttributeAccessIssue]
from behave.runner import Context
from process_analyst_copilot import ClarifyTheAsk, OllamaLLM
from process_analyst_copilot.SemanticAssert import semantic_assert


@given('I have a request "{input_ask}"')  # type: ignore[misc]
def step_impl_given_request(context: Context, input_ask: str) -> None:
    context.input_ask = input_ask


@when("I document the process steps")  # type: ignore[misc]
def step_impl_when_document_process(context: Context) -> None:

    # OpenAI setup for pytest
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv())
    context.draft_process = ClarifyTheAsk()

    # # FIXME Ollama is currently restricted to 2048 context windows
    # # Workaround found here which is why this is editable
    # # https://github.com/ollama/ollama/issues/8356#issuecomment-2579221678
    # context.llm_model = OllamaLLM(
    #     model="ollama/llama3.1:8b",
    #     temperature=0.3,
    #     api_base="http://localhost:11434",
    # )
    # context.llm_model.num_ctx = 2048
    # # FIXME: pydantic_core._pydantic_core.ValidationError: 1 validation error for Crew Value error, Please provide an
    # # OpenAI API key.
    # # Need to set Crew() embedder to avoid this error using memory=True on your Crew()
    # context.draft_process.embedder = {
    #     "provider": "ollama",
    #     "config": {"model": "llama3.1:8b"},
    # }
    # context.draft_process = ClarifyTheAsk(llm_model=context.llm_model)

    context.draft_process.setup()
    context.draft_process.draft_process.output_file = None


@then("I should get the following steps")  # type: ignore[misc]
def step_impl_then_verify_steps(context: Context) -> None:
    expected_steps = """
        Step 1: Boil water
        Step 2: Add tea leaves to cup
        Assumption: Tea variety: black
        Assumption: 1 tea bag per cup
        Step 3: Pour hot water into cup
        Step 4: Steep tea for desired time
        Assumption: 2-5 minutes
        Step 5: Remove tea bag or leaves
        Step 6: Serve tea
    """

    context.result = context.draft_process.kickoff(input_ask=context.input_ask)

    assert semantic_assert(
        context.result, expected_steps
    ), f"Expected {expected_steps}, but got {context.result}"
