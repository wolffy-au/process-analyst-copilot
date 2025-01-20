import pytest
from dotenv import load_dotenv, find_dotenv
from crewai import Task, Crew
from crewai_tools import FileReadTool
from process_analyst_copilot import ClarifyTheAsk
from process_analyst_copilot.SemanticAssert import semantic_assert


@pytest.fixture
def clarify_the_ask() -> ClarifyTheAsk:

    # # OpenAI setup for pytest
    # from dotenv import load_dotenv, find_dotenv
    # load_dotenv(find_dotenv())
    # return ClarifyTheAsk()

    # Ollama setup for pytest
    from process_analyst_copilot import OllamaLLM

    llm_model = OllamaLLM(
        model="ollama/llama3.1:8b",
        temperature=0.3,
        api_base="http://localhost:11434",
    )
    # Ollama default context window
    # 2048 + 1 to trigger OllamaLLM context window warning
    llm_model.num_ctx = 2048 + 1
    return ClarifyTheAsk(llm_model=llm_model)


def test_llm_ctx_warn(clarify_the_ask: ClarifyTheAsk) -> None:
    result: int = clarify_the_ask.llm_model.get_context_window_size()
    expected_output: int = int(2049 * 0.75)

    # Then assert that the result meets your expected output
    assert expected_output == result, f"Expected {expected_output}, but got {result}"


def test_llm_default(clarify_the_ask: ClarifyTheAsk) -> None:
    cta = ClarifyTheAsk()

    expected_output = "gpt-4o-mini"
    result = cta.llm_model.model
    # Then assert that the result meets your expected output
    assert expected_output == result, f"Expected {expected_output}, but got {result}"


def test_llm_response(clarify_the_ask: ClarifyTheAsk) -> None:
    # Given some input data
    input_data = "In one word what is the colour of the sky?"
    expected_output = "blue"

    # When calling your agent's method or task processing logic
    result: str = clarify_the_ask.test_llm(content=input_data)

    # Then assert that the result meets your expected output
    assert semantic_assert(
        expected_output, result
    ), f"Expected {expected_output}, but got {result}"


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
    input_ask = "The simplest way to make a cup of tea?"
    expected_output = """
        1. Boil water
        2. Add tea leaves or tea bag to cup
        3. Pour hot water into the cup
        4. Steep for desired time
        5. Remove tea bag or leaves
        6. Serve
    """

    clarify_the_ask.draft_process.output_file = None
    crew = Crew(
        agents=[clarify_the_ask.business_process_analyst],
        tasks=[clarify_the_ask.draft_process],
    )
    result: str = crew.kickoff(
        inputs={
            "input_ask": input_ask,
        }
    ).raw
    assert semantic_assert(
        expected_output, result
    ), f"Expected {expected_output}, but got {result}"


# Example test case for `capture_assumptions`
def test_capture_assumptions(clarify_the_ask: ClarifyTheAsk) -> None:
    clarify_the_ask.setup()

    expected_output = """
    - Assumes a structured approach to making good tea.
    - Assumes the type of tea the user prefers.
    - Assumes standardised quantity of tea leaves.
    """

    draft_file = "doctest_1_draftprocess.md"
    clarify_the_ask.draft_file_tool = FileReadTool(file_path=draft_file)

    clarify_the_ask.capture_assumptions.output_file = None
    crew = Crew(
        agents=[clarify_the_ask.business_process_analyst],
        tasks=[clarify_the_ask.capture_assumptions],
    )
    result: str = crew.kickoff(
        inputs={
            "draft_file": draft_file,
        }
    ).raw
    assert semantic_assert(
        expected_output, result
    ), f"Expected {expected_output}, but got {result}"


# Example test case for `clarify_details`
def test_clarify_details(clarify_the_ask: ClarifyTheAsk) -> None:
    clarify_the_ask.setup()

    expected_output = """
    1. What is the ideal water temperature for brewing black tea?
    2. How does the steeping time differ for herbal teas compared to other types of tea?
    3. Do you have access to fresh water that has been purified or filtered? If not, what type of water do you
    typically use for brewing tea?
    4. Do you own a suitable strainer and teapot or cups with lids for removing tea leaves from the brewed tea?
    """

    draft_file = "doctest_1_draftprocess.md"
    clarify_the_ask.draft_file_tool = FileReadTool(file_path=draft_file)
    assumptions_file = "doctest_2_assumptions.md"
    clarify_the_ask.assumptions_file_tool = FileReadTool(file_path=assumptions_file)

    clarify_the_ask.clarify_details.output_file = None
    crew = Crew(
        agents=[clarify_the_ask.business_process_analyst],
        tasks=[clarify_the_ask.clarify_details],
    )
    result: str = crew.kickoff(
        inputs={
            "assumptions_file": assumptions_file,
        }
    ).raw
    assert semantic_assert(
        expected_output, result
    ), f"Expected {expected_output}, but got {result}"


# Example test case for `reviewed_process`
def test_reviewed_process(clarify_the_ask: ClarifyTheAsk) -> None:
    clarify_the_ask.setup()

    expected_output = """
    Step 4. **Add Tea Leaves/Bag**
        - Place one teaspoon of loose tea leaves per person into the teapot, plus an extra "for the pot."
        - Alternatively, place a tea bag in each cup if using individual servings.
        - **Assumption:** Type of Tea: User prefers black tea
        - **Assumption:** Standardized quantity: One teaspoon of loose tea leaves per person ensures consistent strength and flavor.
        - **Assumption:** Steeping: Steep Black tea 3-5 minutes adjusted time based on desired strength.
    """

    draft_file = "doctest_1_draftprocess.md"
    clarify_the_ask.draft_file_tool = FileReadTool(file_path=draft_file)
    assumptions_file = "doctest_2_assumptions.md"
    clarify_the_ask.assumptions_file_tool = FileReadTool(file_path=assumptions_file)

    clarify_the_ask.reviewed_process.output_file = None
    crew = Crew(
        agents=[clarify_the_ask.business_process_analyst],
        tasks=[clarify_the_ask.reviewed_process],
    )
    result: str = crew.kickoff(
        inputs={
            "draft_file": draft_file,
            "assumptions_file": assumptions_file,
        }
    ).raw
    assert semantic_assert(
        expected_output, result
    ), f"Expected {expected_output}, but got {result}"


# Example test case for `quality_assurance_review`
def test_quality_assurance_review(clarify_the_ask: ClarifyTheAsk) -> None:
    clarify_the_ask.setup()

    expected_output = """
    4. **Steep and Infuse**
        * Substep 4.1: Allow the tea to steep for the recommended time (usually 2-5 minutes, depending on personal preference).
        - **Assumption:** We assume that the user knows their preferred steeping time.
        * Substep 4.2: If using loose-leaf tea, consider adding a strainer over your cup to catch leaves.
    """

    input_ask = "The simplest way to make a cup of tea?"
    draft_file = "doctest_1_draftprocess.md"
    clarify_the_ask.draft_file_tool = FileReadTool(file_path=draft_file)
    assumptions_file = "doctest_2_assumptions.md"
    clarify_the_ask.assumptions_file_tool = FileReadTool(file_path=assumptions_file)
    reviewed_file = "doctest_4_reviewedprocess.md"
    clarify_the_ask.reviewed_file_tool = FileReadTool(file_path=reviewed_file)

    clarify_the_ask.reviewed_process.output_file = None
    crew = Crew(
        agents=[
            clarify_the_ask.business_process_analyst,
            clarify_the_ask.process_analyst_quality_assurance,
        ],
        tasks=[clarify_the_ask.quality_assurance_review],
    )
    result: str = crew.kickoff(
        inputs={
            "input_ask": input_ask,
            "reviewed_file": reviewed_file,
        }
    ).raw
    assert semantic_assert(
        expected_output, result
    ), f"Expected {expected_output}, but got {result}"
