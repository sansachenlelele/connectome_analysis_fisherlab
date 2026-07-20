@echo off
REM Double-click this file to open the Opto-Camera Control GUI.
REM It runs the app using the dedicated Python 3.10 venv (with PySpin + nidaqmx
REM + PySide6). %~dp0 = the folder this .bat lives in, so it works no matter
REM where the repo is moved.

cd /d "%~dp0"

if not exist ".venv310\Scripts\pythonw.exe" (
    echo ERROR: Python 3.10 venv not found at .venv310
    echo See README.md for setup steps.
    pause
    exit /b 1
)

start "" ".venv310\Scripts\pythonw.exe" "main.py"
