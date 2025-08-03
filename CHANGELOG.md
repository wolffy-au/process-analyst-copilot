## v0.5.0 (2025-08-03)

### Feat

- switched to environment variables to control CrewAI models
- update LLM integration and enhance documentation; refactor agent setup and cleanup scripts
- update local build and git update scripts; bump dependencies and clean up test imports
- update local build script and developer notes for final changes and git push commands

### Refactor

- made base class

## v0.4.0 (2025-01-27)

### Feat

- Update dependencies and improve local build script; fix CQPA references in configuration files
- Add .gitignore files for config and db directories; refactor embedder configuration in ClarifyTheAsk class
- Enhance setup of PQA agent with dynamic configuration
- Introduce OllamaLLM class and utility functions; update VSCode settings for improved linting
- Update poetry.lock and pyproject.toml to include flake8-mypy and adjust markers for compatibility

### Fix

- adjust context size and embedder model in ollama-sample.py

### Refactor

- Update Ollama code in test_ClarifyTheAsk.py for clarity and consistency

## v0.3.0 (2025-01-26)

### Feat

- Add gitupdate script and enhance local build process; update questions in documentation

### Refactor

- Update process consolidation to include questions_file for improved clarity
- Rename setup_agents to setup_bpa_agent and add setup_pqa_agent for improved clarity and functionality
- Remove context parameters from ClarifyTheAsk methods and add unit tests for YAML loading

## v0.2.0 (2025-01-21)

### Feat

- Process quality assurance role and task
- Process quality assurance role and task

### Fix

- Added examples to tasks, refactored ClarifyTheAsk class, improved tests.
- Minor fixes to semantic assert
- Updated review process
- Minor updates to Semantic Assert and tests

### Refactor

- Updated Behave feature tests, refactored setup() for atomic testing and other minor fixes

## v0.1.1 (2025-01-18)

### Fix

- implemented pathlib for platform independent file paths
- Minor change to sample file initial prompt

## v0.1.0 (2025-01-18)

## v0.0.0 (2025-01-16)
