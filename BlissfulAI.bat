@echo off
REM Activate the virtual environment
call .\venv\Scripts\activate.bat
REM Run the Python script with all passed arguments
python BlissfulAI.py %*
REM Deactivate the virtual environment
call .\venv\Scripts\deactivate.bat
