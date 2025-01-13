import json
from typing import Any, Dict
import yaml
from pydantic import BaseModel
from crewai import Agent, Task, Crew, LLM
from crewai_tools import FileReadTool


class Task(BaseModel):
    id: int
    task: str
    materials: list[str] = []
    optional: list[str] = []


class Step(BaseModel):
    id: int
    name: str
    description: str
    tasks: list[Task]


class Process(BaseModel):
    steps: list[Step]

    @staticmethod
    def load(json_file_path: str):
        with open(json_file_path, "r") as file:
            data = json.load(file)
        return Process(**data)


class OllamaLLM(LLM):
    agents_config = None
    tasks_config = None
    num_ctx = 2048

    def get_context_window_size(self) -> int:
        # Override the method with your custom implementation
        return int(self.num_ctx * 0.75)


class ClarifyTheAsk:
    def __init__(
        self,
        llm_model: str = "ollama/llama3.1:8b",
        num_ctx: int = 4096,
        draft_file: str = "./outputs/1-draftprocess.md",
        assumptions_file: str = "./outputs/2-assumptions.md",
        questions_file: str = "./outputs/3-questions.md",
        reviewed_file: str = "./outputs/4-reviewedprocess.md",
    ) -> None:
        self.llm_model = OllamaLLM(
            model=llm_model,
            temperature=0.3,
            api_base="http://localhost:11434",
        )
        self.llm_model.num_ctx = num_ctx

        self.draft_file: str = draft_file
        self.draft_file_tool: FileReadTool = FileReadTool(file_path=self.draft_file)
        self.assumptions_file: str = assumptions_file
        self.assumptions_file_tool: FileReadTool = FileReadTool(file_path=self.assumptions_file)
        self.questions_file: str = questions_file
        self.questions_file_tool: FileReadTool = FileReadTool(file_path=self.questions_file)
        self.reviewed_file: str = reviewed_file

        # Define file paths for YAML configurations
        files: Dict[str, str] = {
            "agents": "config/agents.yaml",
            "tasks": "config/tasks.yaml",
        }

        # Load configurations from YAML files
        configs = {}
        for config_type, file_path in files.items():
            with open(file_path, "r") as file:
                configs[config_type] = yaml.safe_load(file)

        # Assign loaded configurations to specific variables
        self.agents_config = configs["agents"]
        self.tasks_config = configs["tasks"]

    def test_llm(self) -> None:
        print(
            self.llm_model.call(
                [{"role": "system", "content": "What is your context window size?"}]
            )
        )

    def test_crew(self, n_iterations: int = 3) -> None:
        if self.crew is not None:
            self.crew.test(
                n_iterations=n_iterations, openai_model_name=self.llm_model.model
            )

    def setup(self) -> None:
        self.business_process_analyst = Agent(
            config=self.agents_config["business_process_analyst"],
            max_iter=2,  # Default: 20 iterations
            llm=self.llm_model,
        )  # type: ignore

        self.draft_process = Task(
            config=self.tasks_config["draft_process"],
            agent=self.business_process_analyst,
            output_file=self.draft_file,
        )  # type: ignore

        self.capture_assumptions = Task(
            config=self.tasks_config["capture_assumptions"],
            agent=self.business_process_analyst,
            output_file=self.assumptions_file,
            tools=[self.draft_file_tool],
        )  # type: ignore

        self.clarify_details = Task(
            config=self.tasks_config["clarify_details"],
            agent=self.business_process_analyst,
            output_file=self.questions_file,
            tools=[
                self.assumptions_file_tool,
                self.draft_file_tool,
            ],
        )  # type: ignore

        self.reviewed_process = Task(
            config=self.tasks_config["reviewed_process"],
            agent=self.business_process_analyst,
            output_file=self.reviewed_file,
            tools=[
                self.draft_file_tool,
                self.assumptions_file_tool,
                self.questions_file_tool,
            ],
        )  # type: ignore

        self.crew = Crew(
            agents=[self.business_process_analyst],
            tasks=[
                self.draft_process,
                self.capture_assumptions,
                self.clarify_details,
                self.reviewed_process,
            ],
            verbose=True,
            cache=False,
        )

    def kickoff(
        self,
        input_ask: str,
    ) -> Dict[str, Any]:
        self.input_ask = input_ask
        results: Dict[str, Any] = self.crew.kickoff(
            inputs={
                "input_ask": self.input_ask,
                "draft_file": self.draft_file,
                "assumptions_file": self.assumptions_file,
                "questions_file": self.questions_file,
            }
        ).to_dict()
        return results


