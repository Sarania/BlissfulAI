@echo off
REM Check if the venv directory exists
IF NOT EXIST "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
CALL venv\Scripts\activate.bat

echo Updating pip...
python -m pip install --upgrade pip

echo Installing setup requirements...
pip install -r setup_requirements.txt

echo Running easy_setup.py...
python easy_setup.py
CALL install_torch.bat
del install_torch.bat

echo Installing BlissfulAI requirements...
pip install -r requirements.txt

echo Done.

