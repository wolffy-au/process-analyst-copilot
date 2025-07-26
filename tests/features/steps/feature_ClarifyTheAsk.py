from behave import given, when, then
from behave.runner import Context
from process_analyst_copilot import ClarifyTheAsk
from process_analyst_copilot.SemanticAssert import semantic_assert


@given('I have a request "{input_ask}"')  # type: ignore[misc]
def step_impl_given_request(context: Context, input_ask: str) -> None:
    context.input_ask = input_ask


@when("I document the process steps")  # type: ignore[misc]
def step_impl_when_document_process(context: Context) -> None:

    # OpenAI setup for pytest
    from dotenv import load_dotenv, find_dotenv
    import os
    from crewai import LLM

    load_dotenv(find_dotenv())
    context.draft_process = ClarifyTheAsk()

    context.llm_model = LLM(
        model="ollama/llama3.1:8b",
        temperature=0.1,
    )

    # FIXME: pydantic_core._pydantic_core.ValidationError: 1 validation error for Crew Value error, Please provide an
    # OpenAI API key.
    # Need to set Crew() embedder to avoid this error using memory=True on your Crew()
    context.draft_process.embedder = dict(
        provider="ollama",
        config=dict(
            model="nomic-embed-text",
            base_url=os.environ["OLLAMA_API_BASE"],
        ),
    )
    context.draft_process = ClarifyTheAsk(llm_model=context.llm_model)

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
