@echo off
if exist "venv\" (
    REM Activate the virtual environment
    call .\venv\Scripts\activate.bat
    REM Run the Python script with all passed arguments
    python BlissfulAI.py %*
    REM Deactivate the virtual environment
    call .\venv\Scripts\deactivate.bat
) else (
    echo venv directory does not exist. Beginning setup process...
    echo Creating virtual environment...
    python -m venv venv
    echo Activating virtual environment...
    CALL venv\Scripts\activate.bat
    echo Updating pip...
    python -m pip install --upgrade pip
    echo Running easy_setup.py...
    python easy_setup.py
    CALL install_torch.bat
    del install_torch.bat
    echo Installing BlissfulAI requirements...
    pip install -r windows_requirements.txt
    echo Setup complete! Launching BlissfulAI!
    python BlissfulAI.py %*
)
