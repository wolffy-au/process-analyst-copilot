from typing import Any, Dict, Optional
import logging
import os
from pathlib import Path
import yaml
from crewai import Agent, Task, Crew, LLM
from crewai_tools import FileReadTool, PDFSearchTool

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
    embedder_llm: dict[Any, Any] | None = None
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

    def setup_bpa_agent(self) -> None:
        """Sets up Busness Process Analyst agent."""
        # Agent: Busness Process Analyst
        self.business_process_analyst = Agent(
            config=self.agents_config["business_process_analyst"],
            max_iter=2,  # Default: 20 iterations
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

    def setup_pqa_agent(self) -> None:
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
            max_iter=2,  # Default: 20 iterations
            llm=self.llm_model,
            tools=[self.cqpa_bok_tool],
        )  # type: ignore

    def setup_draft_process(self) -> None:
        # Task 1.1: Draft process generation
        self.draft_process = Task(
            config=self.tasks_config["draft_process"],
            agent=self.business_process_analyst,
            output_file=self.draft_file.as_posix(),
        )  # type: ignore[reportCallIssue]

    def setup_capture_assumptions(self) -> None:
        # Task 2.1: Capture assumptions
        self.capture_assumptions = Task(
            config=self.tasks_config["capture_assumptions"],
            agent=self.business_process_analyst,
            output_file=self.assumptions_file.as_posix(),
            tools=[self.draft_file_tool],
        )  # type: ignore[reportCallIssue]

    def setup_clarify_details(self) -> None:
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

    def setup_reviewed_process(self) -> None:
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

    def setup_quality_assurance_review(self) -> None:
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
            memory=True,
            embedder=self.embedder,
            # cache=False,
            verbose=True,
        )

    def setup(self) -> None:
        """Sets up the agents, tasks, and crew for the clarification process."""
        self.setup_bpa_agent()
        self.setup_pqa_agent()
        self.setup_draft_process()
        self.setup_capture_assumptions()
        self.setup_clarify_details()
        self.setup_reviewed_process()
        self.setup_quality_assurance_review()
        self.setup_crew()

    def kickoff(self, input_ask: str) -> str:
        """Kicks off the clarification process with the given input.

        Args:
            input_ask (str): The initial input for the process.

        Returns:
            Dict[str, Any]: The results of the clarification process.
        """
        if not hasattr(self, "crew"):
            raise RuntimeError("Setup method must be called before kickoff.")

        results: str = self.crew.kickoff(
            inputs={
                "input_ask": input_ask,
                "draft_file": self.draft_file.name,
                "assumptions_file": self.assumptions_file.name,
                "questions_file": self.questions_file.name,
                "reviewed_file": self.reviewed_file.name,
            }
        ).raw
        return results

    def test_crew(self, n_iterations: int = 3) -> None:
        """Tests the Crew instance with the given number of iterations.


class ResearchProcess(ProcessAnalystCopilotBase):
    def __init__(
        self,
        llm_model: Optional[LLM] = None,
        config_dir: str = "config",
        output_dir: str = "outputs",
        draft_file: str = "1-draftprocess.md",
    ) -> None:
        super().__init__(llm_model, config_dir, output_dir, draft_file)
        # Tools
        self.serper_tool = SerperDevTool(n_results=3)
        self.scrape_tool = ScrapeWebsiteTool()
        self.draft_file_tool = FileReadTool(file_path=self.draft_file.as_posix())

    def setup_agents(self) -> None:
        # Agent: Web Intelligence Agent
        self.web_intelligence_agent = Agent(
            config=self.agents_config["web_intelligence_agent"],
            max_iter=5,  # Default: 20 iterations
            llm=self.llm_model,
        )  # type: ignore[reportCallIssue]

        # Agent: Busness Process Analyst
        self.business_process_analyst = Agent(
            config=self.agents_config["business_process_analyst"],
            max_iter=5,  # Default: 20 iterations
            llm=self.llm_model,
        )  # type: ignore[reportCallIssue]

    def setup_tasks(self) -> None:
        # Task: Research Activity
        search_results_file = self.output_dir / "search_results.json"
        self.research_activity_task = Task(
            config=self.tasks_config["research_activity"],
            agent=self.web_intelligence_agent,
            output_json=SearchResponse,
            output_file=search_results_file.as_posix(),
            tools=[self.serper_tool],
        )  # type: ignore[reportCallIssue]

        # Task: Research Process
        search_results_tool = FileReadTool(file_path=search_results_file.as_posix())
        self.research_process_task = Task(
            config=self.tasks_config["research_process"],
            agent=self.web_intelligence_agent,
            # output_file=self.draft_file.as_posix(),
            tools=[search_results_tool, self.scrape_tool],
        )  # type: ignore[reportCallIssue]

    def setup_crew(self) -> None:
        self.crew = Crew(
            agents=[self.web_intelligence_agent, self.business_process_analyst],
            tasks=[
                # self.research_activity_task,
                self.research_process_task,
            ],
            # memory=True,
            # embedder=self.embedder,
            # cache=False,
            verbose=True,
        )

    @start()
    def gather_inputs(self) -> dict[Any, Any]:
        if not hasattr(self, "crew"):
            self.setup()

        inputs: dict[str, Any] = {
            "input_ask": "The simplest way to make a cup of tea?",
            "draft_process": (
                "1. Boil water, "
                "2. Add tea leaves or tea bag to cup, "
                "3. Pour hot water into the cup, "
                "4. Steep for desired time, "
                "5. Remove tea bag or leaves, "
                "6. Serve"
            ),
            # "draft_file": "outputs/1-draftprocess.md",
        }
        return inputs

    @listen(gather_inputs)
    def research_activity(self, inputs: dict[str, Any]) -> CrewOutput:
        results = self.crew.kickoff(inputs)
        print(results.raw)
        return results

    # @research_process(research_activity)
    # def store_leads_score(self, scores):
    #     # Here we would store the scores in the database
    #     return scores

    # @listen(score_leads)
    # def filter_leads(self, scores):
    #     return [score for score in scores if score["lead_score"].score > 70]

    # @listen(filter_leads)
    # def write_email(self, leads):
    #     scored_leads = [lead.to_dict() for lead in leads]
    #     emails = email_writing_crew.kickoff_for_each(scored_leads)
    #     return emails

    # @listen(write_email)
    # def send_email(self, emails):
    #     # Here we would send the emails to the leads
    #     return emails


if __name__ == "__main__":
    # FIXME Ollama is currently restricted to 2048 context windows
    # Workaround found here which is why this is editable
    # https://github.com/ollama/ollama/issues/8356#issuecomment-2579221678
    llm_model = OllamaLLM(
        model="ollama/custom",
        temperature=0.3,
        base_url="http://localhost:11434",
    )
    llm_model.num_ctx = 131072

    # draft_process = ResearchProcess(llm_model=llm_model)
    # draft_process.plot()
    # draft_process.kickoff()
