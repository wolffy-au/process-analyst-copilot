# Process Analyst Copilot

## Overview

The Process Analyst Copilot is an application designed to assist process analysts in automating various tasks such as drafting processes, capturing assumptions, and clarifying details. It leverages a language model to provide intelligent responses and streamline the workflow.

## Features

- **Draft Process**: Automatically draft business processes based on input queries.
- **Capture Assumptions**: Identify and document assumptions during the analysis.
- **Clarify Details**: Provide detailed clarifications on specific aspects of the business process.
- **Quality Assurance**: Review and improve specific aspects of the business process.
- **Integration with YAML Configurations**: Load and utilize configurations from YAML files for agents and tasks.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/wolffy-au/process-analyst-copilot.git
    cd process-analyst-copilot
    ```

2. Install dependencies using Poetry:
    ```sh
    poetry install
    ```

3. Set up the environment variables:
    ```sh
    cp .env.example .env
    ```

4. Update the `.env` file with the necessary configurations.

## Configuration

The application uses YAML files for configuring agents and tasks. These files are located in the [config](./config) directory:

- [agents.yaml](./config/agents.yaml): Configuration for agents.
- [tasks.yaml](./config/tasks.yaml): Configuration for tasks.

## Usage

**Copy one of the `sample.py` files to continue:**
- openai-sample.py
- ollama-sample.py

1. Run the application:
    ```sh
    poetry run python sample.py
    ```

2. The main entry point is the `kickoff` method in the `ClarifyTheAsk` class, which initiates the process based on the provided input query.

3. Check the `outputs` directory for the results.

## Example

Here is an example of how to use the application:

```python
from process_analyst_copilot import ClarifyTheAsk, OllamaLLM

llm_model = OllamaLLM(
    model="ollama/llama3.1:8b",
    temperature=0.3,
    api_base="http://localhost:11434",
)
llm_model.num_ctx = 2048

draft_process = ClarifyTheAsk(llm_model=llm_model)
draft_process.setup()
result = draft_process.kickoff(input_ask="How do I make a good cup of tea?")
print("See the outputs directory for outputs.")
```

## Testing

To run the tests, use the following command:
    poetry run pytest

Note: The `semantic_assert` function has a `verbose` parameter that can be set to `True` to log the similarity score during testing.

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please read the CONTRIBUTING.md file for guidelines on how to contribute to this project.

## Contact

For any questions or issues, please open an issue on GitHub or contact the maintainers.
