#!/bin/bash
if [ -d "venv" ]; then
    # Activate the virtual environment
    source ./venv/bin/activate
    # Run the Python script with all passed arguments
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python BlissfulAI.py "$@"
    # Deactivate the virtual environment
    deactivate
else
    echo "Venv directory does not exist. Beginning setup process..."
    echo "Creating virtual environment..."
    python -m venv venv
    echo "Activating virtual environment..."
    source ./venv/bin/activate
    echo "Updating pip..."
    python -m pip install --upgrade pip

    echo "Running easy_setup.py..."
    if ! python easy_setup.py; then
        echo "Installation failed or was cancelled!"
        echo "Deactivating and removing venv..."
        deactivate
        rm -rf ./venv
        # Exit the script if easy_setup.py fails
        exit 1
    fi

    chmod +x ./install_torch.sh
    ./install_torch.sh
    rm install_torch.sh

    echo "Installing BlissfulAI requirements..."
    pip install -r linux_requirements.txt

    echo "Setup complete! Launching BlissfulAI!"
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python BlissfulAI.py "$@"
fi
