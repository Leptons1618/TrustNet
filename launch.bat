@echo off
echo 🔍 TrustLens - Explainable AI
echo =============================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Run setup if requested
if "%1"=="setup" (
    echo 🔧 Running setup...
    python setup_trustlens.py
    if errorlevel 1 (
        echo ❌ Setup failed
        pause
        exit /b 1
    )
    echo ✅ Setup completed successfully!
    echo.
)

REM Check if Streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Streamlit not found. Running setup...
    python setup_trustlens.py
    if errorlevel 1 (
        echo ❌ Setup failed
        pause
        exit /b 1
    )
)

REM Launch the application
echo 🚀 Launching TrustLens...
echo.
echo 🌐 The application will open in your default browser
echo 📱 URL: http://localhost:8501
echo.
echo ⚠️ To stop the application, press Ctrl+C in this window
echo.

streamlit run app.py

pause
