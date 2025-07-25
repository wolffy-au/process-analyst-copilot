#!/bin/sh

git fetch --all --prune
git checkout main
git pull --rebase
git checkout develop
git pull --rebase

git push --delete origin <tag>; git tag -d <tag>

python -m pip install --upgrade pip
pip list --outdated
pip list --outdated --format=columns | awk 'NR>2 {print $1}' | xargs -n1 pip install -U
pip check
pip check | awk '{print $1}' | xargs -n1 pip install -U

poetry env use python
poetry env info
poetry self add poetry-dynamic-versioning
poetry dynamic-versioning enable

poetry lock
poetry update --lock

poetry show --outdated
poetry sync -E dev
poetry run pip check

poetry run mypy # --strict --show-error-codes --no-color
poetry run flake8 # --show-source --statistics
poetry run pytest # --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml
poetry run behave # --no-capture --no-capture-stderr --no-color

poetry run pdoc --output-dir docs --template-dir docs/template strategic_realisation_assistant

poetry build