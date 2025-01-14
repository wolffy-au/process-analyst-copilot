from crewai.crews.crew_output import CrewOutput
import pytest
from crewai import Agent, Task, Crew
from process_analyst_copilot import ClarifyTheAsk, OllamaLLM
from process_analyst_copilot.SemanticAssert import semantic_assert


@pytest.fixture
def clarify_the_ask() -> ClarifyTheAsk:
    llm_model = OllamaLLM(
        model="ollama/llama3.1:8b",
        temperature=0.3,
        api_base="http://localhost:11434",
    )
    llm_model.num_ctx = 4096

    return ClarifyTheAsk(llm_model=llm_model)


def test_llm_response(clarify_the_ask: ClarifyTheAsk) -> None:
    # Given some input data
    input_data = "In one word what is the colour of the sky?"
    expected_output = "blue"

    # When calling your agent's method or task processing logic
    result: str = clarify_the_ask.test_llm(content=input_data)

    # Then assert that the result meets your expected output
    assert semantic_assert(expected_output, result)


def test_agent_response(clarify_the_ask: ClarifyTheAsk) -> None:
    clarify_the_ask.setup()

    # Given some input data
    input_data = "In one word what is the colour of the sky?"
    expected_output = "blue"

    # When calling your agent's method or task processing logic
    result: str = clarify_the_ask.business_process_analyst.execute_task(
        Task(
            description=input_data,
            expected_output="A single word",
            agent=clarify_the_ask.business_process_analyst,
        )
    )

    # Then assert that the result meets your expected output
    assert semantic_assert(
        expected_output, result
    ), f"Expected {expected_output}, but got {result}"


# Example test case for `draft_process`
def test_draft_process(clarify_the_ask: ClarifyTheAsk) -> None:
    clarify_the_ask.setup()

    # Given some input data
    input_ask = "Make a cup of tea"
    expected_output = """
        Step 1: Boil water
        Step 2: Add tea leaves or tea bag to cup
        Step 3: Pour hot water into the cup
        Step 4: Steep for desired time
        Step 5: Remove tea bag or leaves
        Step 6: Serve
    """

    clarify_the_ask.draft_process.output_file = None
    crew = Crew(
        agents=[clarify_the_ask.business_process_analyst],
        tasks=[clarify_the_ask.draft_process],
    )
    result: str = crew.kickoff(
        inputs={
            "input_ask": input_ask,
            # "draft_file": self.draft_file,
        }
    ).raw
    assert semantic_assert(
        expected_output, result
    ), f"Expected {expected_output}, but got {result}"


# # Example test case for `capture_assumptions`
# def test_capture_assumptions():
#     assumptions_file = "assumptions.md"
#     draft_file = "draft_process.md"

#     expected_output = [
#         "What type of tea should be used?",
#         "What is the desired steeping time?",
#         "What is the preferred serving temperature?",
#     ]

#     result = capture_assumptions(assumptions_file, draft_file)
#     assert result == expected_output, f"Expected {expected_output}, but got {result}"


# # Example test case for `clarify_details`
# def test_clarify_details():
#     assumptions_file = "assumptions.md"
#     draft_file = "draft_process.md"
#     assumptions = {"tea_type": "black tea", "steeping_time": "5 minutes"}
#     clarified_output = [
#         "Assumption: Tea type is black tea.",
#         "Assumption: Steeping time is 5 minutes.",
#         "Updated Step 1: Boil water.",
#         "Updated Step 2: Add black tea leaves to cup.",
#         "Updated Step 3: Pour hot water into the cup.",
#     ]

#     result = clarify_details(assumptions, draft_file)
#     assert result == clarified_output, f"Expected {clarified_output}, but got {result}"
