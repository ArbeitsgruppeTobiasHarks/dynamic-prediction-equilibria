#!/bin/sh

# Install this using:
# ln -s ../../pre-commit.sh .git/hooks/pre-commit


for package in dynflows dpe_experiments ; do
  if [ ! -x "$(command -v uv)" ]; then
    echo "You need to install uv in order to commit and format files"
    exit 1
  fi

  (
    cd "$package"
    PY_FILES=$(git diff --cached --name-only --diff-filter=ACMR -- "./**/*.py" | sed 's| |\\ |g' | sed "s|^$package/||")
    if [ -z "$PY_FILES" ]; then
      echo "No Python files to check in $package"
      exit 0
    fi
    set -x
    uv run black --check $PY_FILES || exit 1
    uv run isort --profile black --check $PY_FILES || exit 1
  ) || exit 1
done

exit 0
