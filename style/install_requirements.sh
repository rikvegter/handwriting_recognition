#!/bin/bash
# Install python3.8-venv
echo "In case virtualenv creation fails make sure python3.8-venv is installed."
echo "E.g. on ubuntu:"
echo "sudo apt-get install python3.8-venv"
# Create or clear the virtual environment
python3.8 -m venv ./venv --clear

. ./venv/bin/activate
pip install requirements.txt