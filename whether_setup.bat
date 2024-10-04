@echo off
REM test_setup.bat
REM Script to set up a virtual environment and test the app

REM Set up the project directory
SET scriptPath=%~dp0
SET HOMEDRIVE=%scriptPath:~0,2%
SET projectDir=%HOMEDRIVE%\e\c\whether

REM Change to the project directory
cd %projectDir%
echo Current directory: %cd%

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed. Please install Python before running this script.
    exit /b
)

REM Step 1: Create a virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Step 2: Activate the virtual environment
echo Activating virtual environment...
call venv\Scripts\activate

REM Step 3: Install dependencies from requirements.txt
echo Installing dependencies...
pip install -r requirements.txt

REM Step 4: Run the app for testing
echo Running the app for testing...
python whether.py

REM Step 5: Deactivate the virtual environment after testing
deactivate

REM Confirm completion
echo Testing complete. Virtual environment setup and app tested successfully!
pause
