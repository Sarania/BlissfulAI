#!/bin/bash

# Activate the virtual environment
source ./venv/bin/activate

# Run the Python script
python download_model.py

# Deactivate the virtual environment
deactivate
read -p "Press any key to continue... " -n1 -s