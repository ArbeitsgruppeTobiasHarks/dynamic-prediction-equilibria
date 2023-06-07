#!/bin/sh

# Install this using:
# ln -s pre-commit.sh .git/hooks/pre-commit

PY_FILES=$(git diff --cached --name-only --diff-filter=ACMR "*.py" | sed 's| |\\ |g')

if [ ! -z "$PY_FILES" ]; then
  if [ ! -x "$(command -v poetry)" ]; then
    echo "You need to install poetry in order to commit and format files"
    exit 1
  fi

  # Format all selected files
  echo "$PY_FILES" | xargs poetry run black
  echo "$PY_FILES" | xargs poetry run isort --profile black

  # Add back the modified/prettified files to staging
  echo "$PY_FILES" | xargs git add
fi

exit 0
