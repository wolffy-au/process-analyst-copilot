import pytest
from dotenv import load_dotenv, find_dotenv
import os
from pytest import MonkeyPatch
from pathlib import Path
import yaml
from crewai import Task, Crew, LLM
from crewai_tools import FileReadTool
from process_analyst_copilot import ClarifyTheAsk
from process_analyst_copilot.SemanticAssert import semantic_assert


@pytest.fixture
def clarify_the_ask() -> ClarifyTheAsk:

    load_dotenv(find_dotenv())

    clarify_the_ask: ClarifyTheAsk = ClarifyTheAsk()
    return clarify_the_ask


def test_llm_ctx_warn(clarify_the_ask: ClarifyTheAsk) -> None:
    result: int = clarify_the_ask.llm_model.get_context_window_size()

    expected_output: int = 170000  # 6963

    # Then assert that the result meets your expected output
    assert expected_output == result, f"Expected {expected_output}, but got {result}"


def test_llm_default(clarify_the_ask: ClarifyTheAsk) -> None:
    cta = ClarifyTheAsk()

    expected_output = os.getenv("MODEL", "gpt-4o-mini")
    result = cta.llm_model.model
    # Then assert that the result meets your expected output
    assert expected_output == result, f"Expected {expected_output}, but got {result}"


def test_llm_response(clarify_the_ask: ClarifyTheAsk) -> None:
    # Given some input data
    input_data = "In one basic word, what is the colour of the sky on a clear day?"
    expected_output = "blue"

    # When calling your agent's method or task processing logic
    result: str = LLM(model=os.getenv("MODEL", "gpt-4o-mini")).call(messages=input_data)

    # Then assert that the result meets your expected output
    assert semantic_assert(
        expected_output, result
    ), f"Expected {expected_output}, but got {result}"


def test_bpa_agent_response(clarify_the_ask: ClarifyTheAsk) -> None:
    clarify_the_ask.setup_agents()

    # Given some input data
    input_data = "In two words only, after Collaborating with Stakeholders, what is the first step to improving an \
existing process? Requirements Gathering, Solution Design or Solution Testing?"
    expected_output = "Requirements Gathering"

    # When calling your agent's method or task processing logic
    result: str = clarify_the_ask.business_process_analyst.execute_task(
        Task(
            description=input_data,
            expected_output="Two words only, no preamble or summary.",
            max_retries=0,
        )
    )
    print(result)
    # Then assert that the result meets your expected output
    assert semantic_assert(
        expected_output, result
    ), f"Expected {expected_output}, but got {result}"


def test_pqa_agent_response(clarify_the_ask: ClarifyTheAsk) -> None:
    clarify_the_ask.setup_agents()

    # Given some input data
    input_data = (
        "In two words only, what is the primary goal of CQPA process improvement?"
    )
    expected_output = "Continuous Improvement"

    # When calling your agent's method or task processing logic
    result: str = clarify_the_ask.process_analyst_quality_assurance.execute_task(
        Task(
            description=input_data,
            expected_output="Two words only, no preamble or summary.",
            max_retries=0,
        )
    )
    print(result)
    # Then assert that the result meets your expected output
    assert semantic_assert(
        expected_output, result
    ), f"Expected {expected_output}, but got {result}"


def test_load_yaml_success(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    # Create a temporary YAML file
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    file_name = "test.yaml"
    file_path = config_dir / file_name
    file_path.write_text("key: value")

    # Mock the config_dir attribute
    clarify = ClarifyTheAsk()
    monkeypatch.setattr(clarify, "config_dir", config_dir)

    # Test the _load_yaml method
    result = clarify._load_yaml(file_name)
    assert result == {"key": "value"}


def test_load_yaml_file_not_found(monkeypatch: MonkeyPatch) -> None:
    # Mock the config_dir attribute
    clarify = ClarifyTheAsk()
    monkeypatch.setattr(clarify, "config_dir", Path("/non/existent/path"))

    # Test the _load_yaml method
    with pytest.raises(FileNotFoundError):
        clarify._load_yaml("non_existent.yaml")


def test_load_yaml_parsing_error(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    # Create a temporary invalid YAML file
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    file_name = "invalid.yaml"
    file_path = config_dir / file_name
    file_path.write_text("key: : value")

    # Mock the config_dir attribute
    clarify = ClarifyTheAsk()
    monkeypatch.setattr(clarify, "config_dir", config_dir)

    # Test the _load_yaml method
    with pytest.raises(yaml.YAMLError):
        clarify._load_yaml(file_name)


# Example test case for `draft_process`
def test_draft_process(clarify_the_ask: ClarifyTheAsk) -> None:
    clarify_the_ask.setup_agents()
    clarify_the_ask.setup_tasks()

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


# FIXME
# Example test case for `draft_process_structured`
def test_draft_process_structured(clarify_the_ask: ClarifyTheAsk) -> None:
    clarify_the_ask.setup_agents()
    clarify_the_ask.setup_draft_process_structured()

    expected_output = """
    """

    draft_file = "doctest_1_draftprocess.md"
    clarify_the_ask.draft_file_tool = FileReadTool(file_path=draft_file)

    clarify_the_ask.draft_process_structured.output_file = None
    crew = Crew(
        agents=[clarify_the_ask.business_process_analyst],
        tasks=[clarify_the_ask.draft_process_structured],
    )
    result: str = crew.kickoff().raw
    assert expected_output == result, f"Expected {expected_output}, but got {result}"


# FIXME
# Example test case for `capture_assumptions`
def test_capture_assumptions(clarify_the_ask: ClarifyTheAsk) -> None:
    clarify_the_ask.setup_agents()
    clarify_the_ask.setup_tasks()

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
    clarify_the_ask.setup_agents()
    clarify_the_ask.setup_tasks()

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
    clarify_the_ask.setup_agents()
    clarify_the_ask.setup_tasks()

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
    questions_file = "doctest_3_questions.md"
    clarify_the_ask.questions_file_tool = FileReadTool(file_path=questions_file)

    clarify_the_ask.reviewed_process.output_file = None
    crew = Crew(
        agents=[clarify_the_ask.business_process_analyst],
        tasks=[clarify_the_ask.reviewed_process],
    )
    result: str = crew.kickoff(inputs={}).raw
    assert semantic_assert(
        expected_output, result
    ), f"Expected {expected_output}, but got {result}"


# Example test case for `quality_assurance_review`
def test_quality_assurance_review(clarify_the_ask: ClarifyTheAsk) -> None:
    clarify_the_ask.setup_agents()
    clarify_the_ask.setup_tasks()

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
