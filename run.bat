@echo off
echo 🚀 Starting TrustLens AI...
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found. Using system Python...
)

REM Check if Streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo ❌ Streamlit not found. Please run setup.py first:
    echo    python setup.py
    pause
    exit /b 1
)

REM Start the application
echo Starting TrustLens application...
echo Open your browser to: http://localhost:8501
echo.
streamlit run app.py

pause
