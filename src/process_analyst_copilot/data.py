from typing import Type, TypeVar, ClassVar, Optional, Any
import json
import logging
import os
from pydantic import BaseModel, model_validator

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.WARNING),
    format="%(asctime)s - %(levelname)s - %(message)s",
)


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

    @model_validator(mode="before")
    def validate_step(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate the process step to ensure that it has at least one task."""
        if not values.get("tasks"):
            raise ValueError(f"{cls.name} must have at least one task.")
        return values


class GeneralProcess(BaseModel):
    """Represents a general process containing multiple steps.

    Attributes:
        name (str): The name of the process step.
        description (str): A description of the process step.
        steps (list[ProcessStep]): A list of process steps.
    """

    name: str
    description: str
    steps: list[ProcessStep]
    T: ClassVar = TypeVar("T", bound="GeneralProcess")  # type: ignore

    @staticmethod
    def load(cls: Type[T], json_file_path: str) -> T:
        """Loads a GeneralProcess instance from a JSON file.

        Args:
            json_file_path (str): The path to the JSON file.

        Returns:
            GeneralProcess: An instance of GeneralProcess.
        """
        try:
            with open(json_file_path, "r") as file:
                data = json.load(file)
            return cls(**data)
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {json_file_path}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON file: {json_file_path} - {e}")
            raise


class Assumption(BaseModel):
    """Represents an assumption made during the process analysis.

    Attributes:
        title (str): The title of the assumption.
        description (str): The description of the assumption.
        assumed_value (str): The assumed value for the assumption.
        context (str): Additional context or explanation for the assumption.

    """

    title: str
    description: str
    assumed_value: str
    context: str


class ProcessStepAssumptions(ProcessStep):
    """Represents a process step with additional assumptions.

    Attributes:
        assumptions (Optional[list[Assumption]]): A list of assumptions made for the process step.
    """

    assumptions: Optional[list[Assumption]] = []


class GeneralProcessAssumptions(GeneralProcess):
    """Represents a general process with additional assumptions.

    Attributes:
        steps (list[ProcessStepAssumptions]): A list of assumptions made for the general process.
    """

    steps: list[ProcessStep | ProcessStepAssumptions]
    T: ClassVar = TypeVar("T", bound="GeneralProcessAssumptions")  # type: ignore
