#!/bin/bash
if [ -d "venv" ]; then
    # Activate the virtual environment
    source ./venv/bin/activate
    # Run the Python script with all passed arguments
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
    echo "Installing setup requirements..."
    pip install -r setup_requirements.txt

    echo "Running easy_setup.py..."
    if ! python easy_setup.py; then
        echo "It seems like there's a problem with your setup, possibly related to tkinter."
        echo "Please ensure tkinter is installed for your system."
        echo "On Ubuntu/Debian: sudo apt-get install python3-tk"
        echo "On Fedora: sudo dnf install python3-tkinter"
        echo "On Arch Linux: sudo pacman -S tk"
        echo "Then run the script again."
        echo "Removing venv..."
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
    python BlissfulAI.py "$@"
fi
