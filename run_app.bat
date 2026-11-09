@echo off
rem Start the investor chat (Windows). Double-click, or run from a terminal in the repository folder.
rem First run: finds Python 3.12+, creates .venv and installs the requirements, a few minutes.
rem If an earlier install failed, it is repeated: the marker .venv\installed.ok is written only on success.
setlocal EnableDelayedExpansion
cd /d "%~dp0"

rem 1. a .venv from an older Python is rebuilt
if not exist .venv\Scripts\python.exe goto :check_installed
.venv\Scripts\python.exe -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
if not errorlevel 1 goto :check_installed
echo The existing .venv uses Python older than 3.12. Rebuilding it...
rmdir /s /q .venv

:check_installed
if exist .venv\installed.ok goto :run

rem 2. find Python 3.12 or newer: the py launcher first, then python on PATH
set "PY="
for %%V in (3.14 3.13 3.12) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if defined PY goto :have_py
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
if not errorlevel 1 set "PY=python"
if defined PY goto :have_py
echo.
echo Python 3.12 or newer was not found. The pinned packages - numpy 2.5, pandas 3, scikit-learn 1.9 -
echo need it. Install Python 3.13 from https://www.python.org/downloads/ and tick "Add python.exe to PATH",
echo then run this file again.
pause
exit /b 1

:have_py
if exist .venv\Scripts\python.exe goto :install
echo Creating .venv with: !PY!
!PY! -m venv .venv
if errorlevel 1 goto :fail

:install
echo Installing the requirements, first run only, a few minutes...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-app.txt
if errorlevel 1 goto :fail
echo ok> .venv\installed.ok

:run
.venv\Scripts\python.exe -m streamlit run src\app\streamlit_app.py
exit /b 0

:fail
echo.
echo The installation failed - see the messages above. Check the internet connection and run this
echo file again: it will retry. If it keeps failing, delete the .venv folder and try once more.
pause
exit /b 1
