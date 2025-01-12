#!/bin/sh

# python -m pip install --upgrade pip
# python -m pip install --upgrade virtualenv
# python -m venv .venv
# source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install --upgrade virtualenv
pip install --upgrade poetry
pip install --upgrade poetry-dynamic-versioning
pip install --upgrade types-requests
pip install --upgrade types-PyYAML

poetry env use python
poetry env info

poetry dynamic-versioning enable
poetry install --sync # --without=dev

poetry run mypy guessing_game tests
poetry run flake8 #--output-file=build/flake8/flake8.txt
poetry run pytest --cov=guessing_game --cov-report=term-missing --cov-report=html:build/coverage-reports
poetry run behave
poetry run pdoc --output-dir build/pdoc guessing_game

poetry run cz bump --changelog --yes
poetry build

# poetry run twine check dist/*

deactivate