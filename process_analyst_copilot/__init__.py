from typing import Any, Dict, Optional
import logging
import json
import os
from pathlib import Path
import yaml
from pydantic import BaseModel
from crewai import Agent, Task, Crew, LLM
from crewai_tools import FileReadTool

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# FIXME: pydantic_core._pydantic_core.ValidationError: 1 validation error for Crew Value error, Please provide an
# OpenAI API key.
# Need to set false OPENAI_API_KEY to a non-empty string to avoid this error using memory=True on your Crew()
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "N/A"


class StepTask(BaseModel):
    """Represents a task within a process step.

    Attributes:
        id (int): The unique identifier for the task.
        task (str): The description of the task.
        materials (list[str]): A list of materials required for the task.
        optional (list[str]): A list of optional items for the task.
    """

    id: int
    task: str
    materials: list[str] = []
    optional: list[str] = []


class ProcessStep(BaseModel):
    """Represents a step within a process.

    Attributes:
        id (int): The unique identifier for the process step.
        name (str): The name of the process step.
        description (str): A description of the process step.
        tasks (list[StepTask]): A list of tasks within the process step.
    """

    id: int
    name: str
    description: str
    tasks: list[StepTask]


class GeneralProcess(BaseModel):
    """Represents a general process containing multiple steps.

    Attributes:
        steps (list[ProcessStep]): A list of process steps.
    """

    steps: list[ProcessStep]

    @staticmethod
    def load(json_file_path: str) -> "GeneralProcess":
        """Loads a GeneralProcess instance from a JSON file.

        Args:
            json_file_path (str): The path to the JSON file.

        Returns:
            GeneralProcess: An instance of GeneralProcess.
        """
        try:
            with open(json_file_path, "r") as file:
                data = json.load(file)
            return GeneralProcess(**data)
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {json_file_path}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON file: {json_file_path} - {e}")
            raise


class OllamaLLM(LLM):  # type: ignore[misc]
    """Represents a custom LLM for Ollama with context window adjustments.

    Attributes:
        agents_config (Any): Configuration for agents (not used).
        tasks_config (Any): Configuration for tasks (not used).
        num_ctx (int): The maximum context window size.
    """

    num_ctx = 2048

    def get_context_window_size(self) -> int:
        """Returns the adjusted context window size.

        Returns:
            int: The context window size.
        """
        if self.num_ctx > 2048:
            logging.warning(
                """
                Ollama is currently restricted to 2048 context windows by default.
                For larger contexts, use the work-around documented here:
                https://github.com/ollama/ollama/issues/8356#issuecomment-2579221678
                """
            )
        return int(self.num_ctx * 0.75)


class ClarifyTheAsk:
    """Manages the clarification process for business tasks using LLM and Crew.

    Attributes:
        llm_model (LLM): The LLM model instance.
        draft_file (str): Path to the draft process output file.
        assumptions_file (str): Path to the assumptions output file.
        questions_file (str): Path to the questions output file.
        reviewed_file (str): Path to the reviewed process output file.
    """

    CONFIG_FILES = {"agents": "agents.yaml", "tasks": "tasks.yaml"}
    embedder: dict[Any, Any] | None = None

    def __init__(
        self,
        llm_model: Optional[LLM] = None,
        config_dir: str = "config",
        output_dir: str = "outputs",
        draft_file: str = "1-draftprocess.md",
        assumptions_file: str = "2-assumptions.md",
        questions_file: str = "3-questions.md",
        reviewed_file: str = "4-reviewedprocess.md",
    ) -> None:
        """Initializes the ClarifyTheAsk instance.

        Args:
            llm_model (Optional[LLM, Any]): The LLM model instance.
            draft_file (str): Path to the draft process output file.
            assumptions_file (str): Path to the assumptions output file.
            questions_file (str): Path to the questions output file.
            reviewed_file (str): Path to the reviewed process output file.
        """
        self.llm_model: LLM = llm_model or LLM(model="gpt-4o-mini")

        # Paths
        self.config_dir = Path(config_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.draft_file = self.output_dir / draft_file
        self.assumptions_file = self.output_dir / assumptions_file
        self.questions_file = self.output_dir / questions_file
        self.reviewed_file = self.output_dir / reviewed_file

        # Load configurations from YAML files
        self.agents_config: Dict[str, Any] = self._load_yaml(
            self.CONFIG_FILES["agents"]
        )
        self.tasks_config: Dict[str, Any] = self._load_yaml(self.CONFIG_FILES["tasks"])

    def _load_yaml(self, file_name: str) -> Dict[str, Any]:
        file_path = self.config_dir / file_name
        try:
            with open(file_path, "r") as file:
                return yaml.safe_load(file)  # type: ignore[no-any-return]
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {file_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML file: {file_path} - {e}")
            raise

    def test_llm(self, content: str = "Testing 1 2 3!", role: str = "system") -> str:
        """Tests the LLM with a simple prompt.

        Args:
            content (str): The test prompt content. Defaults to "Testing 1 2 3!".
            role (str): The role of the LLM during the interaction. Defaults to "system".

        Returns:
            str: The LLM's response as a string.
        """
        return str(self.llm_model.call([{"role": role, "content": content}]))

    def test_crew(self, n_iterations: int = 3) -> None:
        """Tests the Crew instance with the given number of iterations.

        Args:
            n_iterations (int): The number of iterations to run. Defaults to 3.
        """
        if hasattr(self, "crew"):
            self.crew.test(
                n_iterations=n_iterations, openai_model_name=self.llm_model.model
            )

    def setup(self) -> None:
        """Sets up agents and tasks for the clarification process."""
        # Agent: Busness Process Analyst
        self.business_process_analyst = Agent(
            config=self.agents_config["business_process_analyst"],
            max_iter=2,  # Default: 20 iterations
            llm=self.llm_model,
        )  # type: ignore[reportCallIssue]

        # Agent: Certified Process Quality Assurance
        self.cpqa_bok_tool: FileReadTool = FileReadTool(
            file_path=Path(
                # Sample reference doc
                Path(self.config_dir)
                / "references"
                / "cpqa-cert-insert.pdf"
            ).as_posix()
        )
        self.process_analyst_quality_assurance = Agent(
            config=self.agents_config["process_analyst_quality_assurance"],
            max_iter=2,  # Default: 20 iterations
            llm=self.llm_model,
            tools=[self.cpqa_bok_tool],
        )  # type: ignore

        # Task 1.1: Draft process generation
        self.draft_process = Task(
            config=self.tasks_config["draft_process"],
            agent=self.business_process_analyst,
            output_file=self.draft_file.as_posix(),
        )  # type: ignore[reportCallIssue]

        self.draft_file_tool = FileReadTool(file_path=self.draft_file.as_posix())
        # Task 2.1: Capture assumptions
        self.capture_assumptions = Task(
            config=self.tasks_config["capture_assumptions"],
            agent=self.business_process_analyst,
            output_file=self.assumptions_file.as_posix(),
            tools=[self.draft_file_tool],
        )  # type: ignore[reportCallIssue]

        # Task 3.1: Clarify details
        self.assumptions_file_tool = FileReadTool(
            file_path=self.assumptions_file.as_posix()
        )
        self.clarify_details = Task(
            config=self.tasks_config["clarify_details"],
            agent=self.business_process_analyst,
            output_file=self.questions_file.as_posix(),
            tools=[self.draft_file_tool, self.assumptions_file_tool],
        )  # type: ignore[reportCallIssue]

        # TODO include human clarification
        # TODO identify constraints
        # TODO identify contradictions
        # TODO identify solution components impacted

        # Task 5.1: Review process
        self.questions_file_tool = FileReadTool(
            file_path=self.questions_file.as_posix()
        )

        self.reviewed_process = Task(
            config=self.tasks_config["reviewed_process"],
            agent=self.business_process_analyst,
            output_file=self.reviewed_file.as_posix(),
            tools=[
                self.draft_file_tool,
                self.assumptions_file_tool,
                self.questions_file_tool,
            ],
        )  # type: ignore[reportCallIssue]

        # Task 5.2 Quality assure process
        self.reviewed_file_tool: FileReadTool = FileReadTool(
            file_path=self.reviewed_file.as_posix()
        )
        self.quality_assurance_review = Task(
            config=self.tasks_config["quality_assurance_review"],
            agent=self.process_analyst_quality_assurance,
            tools=[
                self.reviewed_file_tool,
            ],
        )  # type: ignore

        self.crew = Crew(
            agents=[
                self.business_process_analyst,
                self.process_analyst_quality_assurance,
            ],
            tasks=[
                self.draft_process,
                self.capture_assumptions,
                self.clarify_details,
                self.reviewed_process,
                self.quality_assurance_review,
            ],
            memory=True,
            embedder=self.embedder,
            # cache=False,
            verbose=True,
        )

    def kickoff(self, input_ask: str) -> Dict[str, Any]:
        """Kicks off the clarification process with the given input.

        Args:
            input_ask (str): The initial input for the process.

        Returns:
            Dict[str, Any]: The results of the clarification process.
        """
        if not hasattr(self, "crew"):
            raise RuntimeError("Setup method must be called before kickoff.")

        results: Dict[str, Any] = self.crew.kickoff(
            inputs={
                "input_ask": input_ask,
                "draft_file": self.draft_file.name,
                "assumptions_file": self.assumptions_file.name,
                "questions_file": self.questions_file.name,
                "reviewed_file": self.reviewed_file.name,
            }
        ).to_dict()
        return results
