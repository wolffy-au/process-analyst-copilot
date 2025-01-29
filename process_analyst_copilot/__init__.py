from typing import Any, Dict, Optional
import logging
import os
from pathlib import Path
import yaml
from crewai import Agent, Task, Crew, LLM, Flow
from crewai.flow.flow import listen, start
from crewai.crews.crew_output import CrewOutput
from crewai_tools import FileReadTool, PDFSearchTool
from process_analyst_copilot.utils import OllamaLLM


# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Prevent OpenTelemetry errors from logging when behind a firewall
os.environ["OTEL_PYTHON_DISABLED_INSTRUMENTATIONS"] = (
    "azure_sdk,django,fastapi,flask,psycopg2,requests,urllib,urllib3"
)


class ProcessAnalystCopilotBase(Flow):
    CONFIG_FILES = {"agents": "agents.yaml", "tasks": "tasks.yaml"}
    embedder_llm: dict[Any, Any] | None = None
    embedder: dict[Any, Any] | None = None

    def __init__(
        self,
        llm_model: Optional[LLM] = None,
        config_dir: str = "config",
        output_dir: str = "outputs",
        draft_file: str = "1-draftprocess.md",
    ) -> None:
        super().__init__()
        """Base process initialization."""
        self.llm_model: LLM = llm_model or LLM(model="gpt-4o-mini")

        # Paths
        self.config_dir = Path(config_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.draft_file = self.output_dir / draft_file

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

    def setup_agents(self) -> None:
        """This method should be implemented in child classes to set up agents."""
        raise NotImplementedError("setup_agents() must be implemented in subclasses.")

    def setup_tasks(self) -> None:
        """This method should be implemented in child classes to set up tasks."""
        raise NotImplementedError("setup_tasks() must be implemented in subclasses.")

    def setup_crew(self) -> None:
        """This method should be implemented in child classes to set up the Crew."""
        raise NotImplementedError("setup_crew() must be implemented in subclasses.")

    def setup(self) -> None:
        """Runs setup methods in subclasses."""
        self.setup_agents()
        self.setup_tasks()
        self.setup_crew()


class ClarifyTheAsk(ProcessAnalystCopilotBase):
    """Manages the clarification process for business tasks using LLM and Crew.

    Attributes:
        llm_model (LLM): The LLM model instance.
        draft_file (str): Path to the draft process output file.
        assumptions_file (str): Path to the assumptions output file.
        questions_file (str): Path to the questions output file.
        reviewed_file (str): Path to the reviewed process output file.
    """

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
        super().__init__(llm_model, config_dir, output_dir, draft_file)
        # Paths
        self.assumptions_file = self.output_dir / assumptions_file
        self.questions_file = self.output_dir / questions_file
        self.reviewed_file = self.output_dir / reviewed_file

    def setup_agents(self) -> None:
        """Sets up Busness Process Analyst agent."""
        # Agent: Busness Process Analyst
        self.business_process_analyst = Agent(
            config=self.agents_config["business_process_analyst"],
            max_iter=5,  # Default: 20 iterations
            llm=self.llm_model,
        )  # type: ignore[reportCallIssue]

        self.draft_file_tool = FileReadTool(file_path=self.draft_file.as_posix())
        self.assumptions_file_tool = FileReadTool(
            file_path=self.assumptions_file.as_posix()
        )
        self.questions_file_tool = FileReadTool(
            file_path=self.questions_file.as_posix()
        )
        self.reviewed_file_tool: FileReadTool = FileReadTool(
            file_path=self.reviewed_file.as_posix()
        )

        """Sets up Certified  Quality Process Assurance agent."""
        config = None
        if self.embedder is not None and self.embedder_llm is not None:
            config = dict(
                llm=self.embedder_llm,
                embedder=self.embedder,
            )
        elif self.embedder is not None and self.embedder_llm is None:
            config = dict(
                embedder=self.embedder,
            )

        # Agent: Certified  Quality Process Assurance
        self.cqpa_bok_tool: PDFSearchTool = PDFSearchTool(
            pdf=Path(
                # Sample reference doc
                Path(self.config_dir)
                / "references"
                / "certified-quality-process-analyst-handbook.pdf"
            ).as_posix(),
            config=config,
        )
        self.process_analyst_quality_assurance = Agent(
            config=self.agents_config["process_analyst_quality_assurance"],
            max_iter=5,  # Default: 20 iterations
            llm=self.llm_model,
            tools=[self.cqpa_bok_tool],
        )  # type: ignore

    def setup_tasks(self) -> None:
        # Task 1.1: Draft process generation
        self.draft_process = Task(
            config=self.tasks_config["draft_process"],
            agent=self.business_process_analyst,
            output_file=self.draft_file.as_posix(),
        )  # type: ignore[reportCallIssue]

        # Task 2.1: Capture assumptions
        self.capture_assumptions = Task(
            config=self.tasks_config["capture_assumptions"],
            agent=self.business_process_analyst,
            output_file=self.assumptions_file.as_posix(),
            tools=[self.draft_file_tool],
        )  # type: ignore[reportCallIssue]

        # Task 3.1: Clarify details
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
        self.quality_assurance_review = Task(
            config=self.tasks_config["quality_assurance_review"],
            agent=self.process_analyst_quality_assurance,
            tools=[
                self.reviewed_file_tool,
            ],
        )  # type: ignore

    def setup_crew(self) -> None:
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
            # memory=True,
            # embedder=self.embedder,
            # cache=False,
            verbose=True,
        )

    @start()
    def run_crew(self) -> str:
        """Kicks off the clarification process with the given input.

        Args:
            input_ask (str): The initial input for the process.

        Returns:
            Dict[str, Any]: The results of the clarification process.
        """
        if not hasattr(self, "crew"):
            self.setup()

        inputs = {
            "input_ask": self.state.get("input_ask"),
            "draft_file": self.draft_file.name,
            "assumptions_file": self.assumptions_file.name,
            "questions_file": self.questions_file.name,
            "reviewed_file": self.reviewed_file.name,
        }

        results: str = self.crew.kickoff(inputs).raw
        return results

    # def test_crew(self, n_iterations: int = 3) -> None:
    #     """Tests the Crew instance with the given number of iterations.

    #     Args:
    #         n_iterations (int): The number of iterations to run. Defaults to 3.
    #     """
    #     if hasattr(self, "crew"):
    #         self.crew.test(
    #             n_iterations=n_iterations, openai_model_name=self.llm_model.model
    #         )
