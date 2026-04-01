@echo off
setlocal enabledelayedexpansion
title PneumoAI Launcher
echo ========================================
echo        PneumoAI - Pneumonia Detection
echo ========================================
echo.

:: Use the folder where this script resides
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Check if portable Python already exists
if exist "%SCRIPT_DIR%python_portable\python.exe" (
    echo Using portable Python from local folder...
    set "PYTHON_EXE=%SCRIPT_DIR%python_portable\python.exe"
    set "PIP_EXE=%SCRIPT_DIR%python_portable\Scripts\pip.exe"
    goto :install_deps
)

:: Download Python embeddable package (64-bit)
echo Downloading portable Python (this may take a minute)...
set "PYTHON_URL=https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip"
set "PYTHON_ZIP=%SCRIPT_DIR%python_portable.zip"

:: Use PowerShell to download
powershell -Command "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%PYTHON_ZIP%'"
if errorlevel 1 (
    echo Download failed. Please check your internet connection.
    pause
    exit /b
)

:: Extract
powershell -Command "Expand-Archive -Path '%PYTHON_ZIP%' -DestinationPath '%SCRIPT_DIR%python_portable' -Force"
del "%PYTHON_ZIP%" >nul 2>&1

:: Enable pip by editing the python._pth file (to allow site-packages)
echo Adding pip support...
(
    echo python310.zip
    echo .
    echo # Uncomment to run site-packages
    echo import site
) > "%SCRIPT_DIR%python_portable\python._pth"

:: Bootstrap pip
set "PYTHON_EXE=%SCRIPT_DIR%python_portable\python.exe"
"%PYTHON_EXE%" -c "import ensurepip; ensurepip.bootstrap()"

:: Upgrade pip and install setuptools, wheel
"%PYTHON_EXE%" -m pip install --upgrade pip setuptools wheel

set "PIP_EXE=%SCRIPT_DIR%python_portable\Scripts\pip.exe"

:install_deps
:: Install required packages
echo Installing required packages (this may take a few minutes)...
"%PIP_EXE%" install -r requirements.txt

:: Run Streamlit
echo Starting PneumoAI...
echo The app will open in your browser shortly.
"%PYTHON_EXE%" -m streamlit run app.py --server.headless true

echo.
echo Press any key to close this window.
pause >nul
