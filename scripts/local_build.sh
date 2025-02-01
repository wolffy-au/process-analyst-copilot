#!/bin/sh

python -m pip install --upgrade pip
pip install --upgrade setuptools
pip install --upgrade virtualenv
pip install --upgrade poetry
pip install --upgrade poetry-dynamic-versioning
pip install --upgrade types-requests
pip install --upgrade types-PyYAML

poetry env use python
poetry env info

poetry dynamic-versioning enable

pyenv update
pyenv install --list | grep 3.11

poetry show --outdated
pip check

poetry lock
poetry update --lock
poetry install # --sync --without=dev

poetry run mypy process_analyst_copilot tests
poetry run flake8 #--output-file=build/flake8/flake8.txt
poetry run pytest --cov=process_analyst_copilot --cov-report=term-missing --cov-report=html:build/coverage-reports
poetry run behave

poetry run pdoc --output-dir docs process_analyst_copilot
# Update repo with final changes
poetry run cz bump --changelog --yes
poetry build

# poetry run twine check dist/*
