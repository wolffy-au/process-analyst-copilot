# Business Analyst Copilot

## Overview

The Business Analyst Copilot is an application designed to assist business analysts in automating various tasks such as drafting processes, capturing assumptions, and clarifying details. It leverages a language model to provide intelligent responses and streamline the workflow.

## Features

- **Draft Process**: Automatically draft business processes based on input queries.
- **Capture Assumptions**: Identify and document assumptions during the analysis.
- **Clarify Details**: Provide detailed clarifications on specific aspects of the business process.
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

4. Update the [.env](http://_vscodecontentref_/1) file with the necessary configurations.

## Configuration

The application uses YAML files for configuring agents and tasks. These files are located in the [config](http://_vscodecontentref_/2) directory:

- [agents.yaml](http://_vscodecontentref_/3): Configuration for agents.
- [tasks.yaml](http://_vscodecontentref_/4): Configuration for tasks.

## Usage

1. Run the application:
    ```sh
    poetry run python sample.py
    ```

2. The main entry point is the [kickoff](http://_vscodecontentref_/5) method, which initiates the process based on the provided input query.

## Example

Here is an example of how to use the application:

```python
from process_analyst_copilot import ClarifyTheAsk

if __name__ == "__main__":
    draft_process = ClarifyTheAsk(
        llm_model="ollama/llama3.1:8b",
    )
    draft_process.setup()
    results = draft_process.kickoff(input_ask="How do I make a good cup of tea?")
    print("See the outputs directory for outputs.")
```

## Testing

To run the tests, use the following command:
    poetry run pytest

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please read the CONTRIBUTING.md file for guidelines on how to contribute to this project.

## Contact

For any questions or issues, please open an issue on GitHub or contact the maintainers.
