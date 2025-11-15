@echo off
echo Installing Simple Bot Trading System
echo ====================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Python is installed, proceeding with installation...

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo pip is not available
    pause
    exit /b 1
)

REM Create and activate virtual environment
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install required packages
echo Installing required packages...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Installation completed!
echo.
echo To run the bot:
echo 1. Make sure MetaTrader 5 terminal is running and logged in
echo 2. Copy your config.py file to this directory
echo 3. Run: python simple_bot.py
echo.
echo IMPORTANT: Before running the bot, you must have:
echo - MetaTrader 5 terminal installed and running
echo - Valid MT5 account credentials in config.py
echo - "Allow automated trading" enabled in MT5 settings
echo.

pause