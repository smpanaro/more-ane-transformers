#!/usr/bin/env bash

set -euo pipefail

set +e
PYTHON_VERSION=$(python3 -c "import sys; print('.'.join([str(x) for x in sys.version_info[:3]])); sys.exit(0 if sys.version_info[0] == 3 and sys.version_info[1] <= 11 else 1)")
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
    read -p "This project will probably not work with your Python version (${PYTHON_VERSION}). Try anyways? (y/n): " choice
    case "$choice" in
      y|Y )
        ;;
      n|N )
        echo "You can edit this file (line 6 and 42) to replace 'python3' references with 'python3.10' or lower and re-run."
        exit 1
        ;;
      * )
        echo "Invalid choice: $choice"
        exit 1
        ;;
    esac
fi

python3 -m venv env
echo "Virtualenv created at in env/"

source env/bin/activate
echo "Activated virtualenv"

echo "Installing dependencies..."
pip install -r requirements.txt
pip install ane-transformers==0.1.3
printf "All set! \360\237\220\254\n"
