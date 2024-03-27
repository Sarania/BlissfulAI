@echo off
REM Activate the virtual environment
call .\venv\Scripts\activate.bat
python download_model.py
REM Deactivate the virtual environment
call .\venv\Scripts\deactivate.bat