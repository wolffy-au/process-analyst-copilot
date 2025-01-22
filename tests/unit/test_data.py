import pytest
from typing import Any
import json
import os
from pathlib import Path
from process_analyst_copilot.data import GeneralProcess, GeneralProcessAssumptions


def test_general_process_load() -> None:
    test_json: dict[str, Any] = {
        "name": "Test Process",
        "description": "A test process.",
        "steps": [
            {
                "id": 1,
                "name": "Step 1",
                "description": "First step",
                "tasks": [
                    {
                        "id": 1,
                        "task": "Task 1",
                    },
                ],
            }
        ],
    }
    with open("tests/unit/test.json", "w") as file:
        json.dump(test_json, file)

    process: GeneralProcess = GeneralProcess.load(
        GeneralProcess, "tests/unit/test.json"
    )

    # remove file
    os.remove("tests/unit/test.json")
    assert process.name == "Test Process"


def test_general_process_assumptions_load() -> None:
    test_json: dict[str, Any] = {
        "name": "Test Process with Assumptions",
        "description": "A test process.",
        "steps": [
            {
                "id": 1,
                "name": "Step 1",
                "description": "First step",
                "tasks": [
                    {
                        "id": 1,
                        "task": "Task 1",
                    },
                ],
                "assumptions": [],
            }
        ],
    }
    with open("tests/unit/test.json", "w") as file:
        json.dump(test_json, file)

    process: GeneralProcessAssumptions = GeneralProcessAssumptions.load(
        GeneralProcessAssumptions, "tests/unit/test.json"
    )
    # remove file
    os.remove("tests/unit/test.json")
    assert process.name == "Test Process with Assumptions"


def test_load_json_file_not_found() -> None:
    # Test the load method
    with pytest.raises(FileNotFoundError):
        _ = GeneralProcess.load(GeneralProcess, "/non/existent/path")


def test_load_json_parsing_error(tmp_path: Path) -> None:
    # Create a temporary invalid YAML file
    file_name = "invalid.json"
    file_path = tmp_path / file_name
    file_path.write_text("key: : value")

    # Test the _load_json method
    with pytest.raises(json.JSONDecodeError):
        _ = GeneralProcess.load(GeneralProcess, file_path.as_posix())
