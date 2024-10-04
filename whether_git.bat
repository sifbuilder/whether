@echo off
REM unified.bat
REM Script to either deploy a new repo or update an existing one, depending on the state

REM Check if -d or --doc is passed as the first argument to show documentation
if "%~1"=="-d" (
    goto :DOC
) else if "%~1"=="--doc" (
    goto :DOC
)

REM High-level default param for project name (whether)
set projectName=%~2
if "%projectName%"=="" (
    set projectName=whether
)

REM Continue with normal execution

REM Get the drive from the script path
SET scriptPath=%~dp0
SET HOMEDRIVE=%scriptPath:~0,2%

REM Define the folder structure: drive\e\c\project
SET entryDir=%HOMEDRIVE%\e\c\%projectName%

REM Ensure the project directory exists
if not exist "%entryDir%" (
    echo Project directory %entryDir% does not exist. Creating it...
    mkdir %entryDir%
)

REM Change directory to the project
cd %entryDir%
echo Project directory: %entryDir%

REM Ensure the GitHub token, username, and email are available
if "%GITHUB_TOKEN%"=="" (
    echo GitHub token is not set. Please run the git_config.bat script to set the token.
    exit /b
)

if "%GITHUB_USER%"=="" (
    echo GitHub username is not set. Please run the git_config.bat script to set the GitHub username.
    exit /b
)

if "%GITHUB_EMAIL%"=="" (
    echo GitHub email is not set. Please run the git_config.bat script to set the GitHub email.
    exit /b
)

REM Get the current date
for /f "tokens=2 delims==" %%i in ('wmic os get localdatetime /value') do set dt=%%i
set dt=%dt:~0,8%

REM Check if the Git repository already exists
if exist ".git" (
    echo Repository already exists. Entering update mode...
    goto :UPDATE
) else (
    echo No repository found. Entering deployment mode...
    goto :DEPLOY
)

:DEPLOY
REM Step 1: Initialize a new Git repository
git init
echo Git repository initialized.

REM Step 2: Stage the Python script, README, and requirements.txt for commit
git add whether.py README.md requirements.txt

REM Step 3: Commit the staged files with a message containing the date
git commit -m "Initial commit - %dt%"

REM Step 4: Use GitHub's API to create a new repository using the token from the environment
curl -u "%GITHUB_USER%:%GITHUB_TOKEN%" https://api.github.com/user/repos -d "{\"name\":\"%projectName%-expert\"}"

REM Step 5: Set the remote URL (without including the token)
git remote add origin https://github.com/%GITHUB_USER%/%projectName%-expert.git

REM Step 6: Push the local repository to GitHub
git push -u origin master

REM Confirm successful project setup
@echo Project setup and pushed to GitHub successfully!
pause
goto :EOF

:UPDATE
REM Step 1: Check if remote origin exists, add it if missing
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo Remote origin not found. Adding remote origin...
    git remote add origin https://github.com/%GITHUB_USER%/%projectName%-expert.git
) else (
    echo Remote origin exists.
)

REM Step 2: Stage any changes for commit
git add -A

REM Step 3: Commit the changes with a message containing the date
git commit -m "Update commit - %dt%"

REM Step 4: Push the updates to the remote repository
git push origin master

REM Confirm successful update
@echo Project updated and pushed to GitHub successfully!
pause
goto :EOF

:DOC
echo *************************************
echo Unified Deployment and Update Script - Documentation
echo *************************************
echo.
echo This script either deploys the project to GitHub for the first time or updates the repository.
echo The mode is determined by the presence of a Git repository (.git folder):
echo - If no repository exists, the script will enter deployment mode to initialize the repo and push the initial commit.
echo - If a repository is already present, the script will enter update mode to commit and push new changes.
echo.
echo Usage:
echo     - Run the script normally to either deploy or update the project.
echo     - You can provide the project name as a second argument. If not provided, the default is 'whether'.
echo     - Ensure that the following environment variables are set:
echo         - GITHUB_TOKEN (GitHub Personal Access Token)
echo         - GITHUB_USER (GitHub username)
echo         - GITHUB_EMAIL (GitHub email)
echo *************************************
exit /b
