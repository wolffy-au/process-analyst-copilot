#!/bin/sh

rm -rf `find -type d -name __pycache__`
rm -rf .coverage cov.json poetry.lock .mypy_cache/ .pytest_cache/ build/ dist/ docs/ test_sqlite_models_*.db
