from typing import Any, Dict
import yaml
from crewai import Agent, Task, Crew, LLM
from crewai_tools import FileReadTool

num_ctx = 16384


class OllamaLLM(LLM):
    agents_config = None
    tasks_config = None

    def get_context_window_size(self) -> int:
        # Override the method with your custom implementation
        return int(num_ctx * 0.75)


class ClarifyTheAsk:
    def __init__(
        self,
        llm_model: str = "ollama/Xphi4",
        draft_file: str = "./outputs/1-draftprocess.md",
        assumptions_file: str = "./outputs/2-assumptions.md",
        questions_file: str = "./outputs/3-questions.md",
        reviewed_file: str = "./outputs/4-reviewedprocess.md",
    ) -> None:
        self.draft_file: str = draft_file
        self.assumptions_file: str = assumptions_file
        self.questions_file: str = questions_file
        self.reviewed_file: str = reviewed_file
        self.llm_model = OllamaLLM(
            model=llm_model,
            temperature=0.3,
            api_base="http://localhost:11434",
        )

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

    def testLLM(self) -> None:
        print(
            self.llm_model.call(
                [{"role": "system", "content": "What is your context window size?"}]
            )
        )

    def testCrew(self, n_iterations: int = 3) -> None:
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
            tools=[FileReadTool(file_path=self.draft_file)],
        )  # type: ignore

        self.clarify_details = Task(
            config=self.tasks_config["clarify_details"],
            agent=self.business_process_analyst,
            output_file=self.questions_file,
            tools=[
                FileReadTool(file_path=self.assumptions_file),
                FileReadTool(file_path=self.draft_file),
            ],
        )  # type: ignore

        self.reviewed_process = Task(
            config=self.tasks_config["reviewed_process"],
            agent=self.business_process_analyst,
            output_file=self.reviewed_file,
            tools=[
                FileReadTool(file_path=self.draft_file),
                FileReadTool(file_path=self.assumptions_file),
                FileReadTool(file_path=self.questions_file),
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


if __name__ == "__main__":
    draft_process = ClarifyTheAsk(
        llm_model="ollama/Xllama3.1",
        # llm_model="ollama/Xllama3.2",
        # llm_model="ollama/Xdolphin3",
        # llm_model="ollama/Xphi4",
        # llm_model="ollama/Xolmo2",
    )
    draft_process.setup()
    # draft_process.testLLM()
    # draft_process.testCrew(n_iterations=1)
    draft_process.kickoff(input_ask="How do I make a good cup of tea?")
