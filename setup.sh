#!/usr/bin/env bash

set -euo pipefail

set +e
PYTHON_VERSION=$(python -c "import sys; print('.'.join([str(x) for x in sys.version_info[:3]])); sys.exit(0 if sys.version_info[0] == 3 and sys.version_info[1] <= 9 else 1)")
PYTHON_VERSION_VALID=$?
set -e

# Prompt user to confirm they want to continue
read -p "This script will setup a new virtual env and install required dependencies. Continue? (y/n): " choice
case "$choice" in
  y|Y )
    ;;
  n|N )
    exit 1
    ;;
  * )
    echo "Invalid choice: $choice"
    exit 1
    ;;
esac

if [[ $PYTHON_VERSION_VALID -eq 0 ]]; then
    echo "Using Python $PYTHON_VERSION"
else
    read -p "This project may not work with your Python version (${PYTHON_VERSION}). Continue anyways? (y/n): " choice
    case "$choice" in
      y|Y )
        ;;
      n|N )
        exit 1
        ;;
      * )
        echo "Invalid choice: $choice"
        exit 1
        ;;
    esac
fi

python -m venv env
echo "Virtualenv created at in env/"

source env/bin/activate
echo "Activated virtualenv"

echo "Installing dependencies..."
pip install -r requirements.txt
echo -e "All set! \U1F42C"

