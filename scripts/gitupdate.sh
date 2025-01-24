#!/bin/sh

git fetch --all --prune
git checkout main
git pull --rebase
git checkout develop
git pull --rebase
git checkout feature-structuredprocess
git pull --rebase
# git rebase develop
# git push --force-with-lease