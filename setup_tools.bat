@echo off
setlocal
cd /d "%~dp0"
title Setup: download third-party tools
REM ----------------------------------------------------------------------------------------------
REM  Downloads the third-party tools into tools\ and installs the Python packages (requirements.txt).
REM  It lists what it will download and asks before starting.   Options (add after the file name):
REM     --check            show what is installed / missing, change nothing
REM     --dry-run          show what would be downloaded, change nothing
REM     --yes              do not ask for confirmation
REM     --only brush,colmap   install just these (deps brush colmap colmap-cuda da3-src da3-venv da3-model dav2)
REM     --skip deps        install everything except these
REM  Python used: the first "python" on PATH (tested: 3.13.7). Set PYTHON below to use another one.
REM ----------------------------------------------------------------------------------------------
set "PYTHON=python"
"%PYTHON%" -u "setup_tools.py" %*
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" (echo Setup finished with exit code %RC%. Read the messages above.) else (echo Setup finished.)
pause
endlocal
