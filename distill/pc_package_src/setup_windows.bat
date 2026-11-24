@echo off
cd /d "%~dp0"
echo === WSET distill: one-time setup ===

set PY=py -3
where py >nul 2>nul || set PY=python
%PY% --version
if errorlevel 1 goto nopython
%PY% -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"
if errorlevel 1 goto oldpython

if not exist .venv %PY% -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo.
echo Installing PyTorch with CUDA 12.8 (about 3 GB, needed for the RTX 5060 Ti)...
pip install torch --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto fail

echo.
echo Installing the other packages...
pip install -r requirements.txt
if errorlevel 1 goto fail

echo.
python check_gpu.py
echo.
echo Setup finished. Next: double-click run_notebook.bat
pause
exit /b 0

:nopython
echo Python was not found. Install Python 3.11 or 3.12 from python.org (tick "Add python.exe to PATH"), then run this file again.
pause
exit /b 1

:oldpython
echo Python 3.10 or newer is needed. Install Python 3.11 or 3.12 from python.org, then run this file again.
pause
exit /b 1

:fail
echo Something failed during installation. Copy the messages above and send them over.
pause
exit /b 1
