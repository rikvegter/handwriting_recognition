#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
echo $SCRIPT_DIR
python3 "${SCRIPT_DIR}/scripts/classify_style.py" $1 "${SCRIPT_DIR}/classifier" "${PWD}/results"