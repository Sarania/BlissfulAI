#!/bin/bash

# Check if the venv directory exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Updating pip..."
python -m pip install --upgrade pip

echo "Installing setup requirements..."
pip install -r setup_requirements.txt

echo "Running easy_setup.py..."
python easy_setup.py
./install_torch.sh
rm install_torch.sh

echo "Installing BlissfulAI requirements..."
pip install -r requirements.txt

echo "Done."
read -p "Press any key to continue... " -n1 -s