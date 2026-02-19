@echo off
REM PDS post-install script for Windows.
REM Installs PDS from bundled source and runs "pds -m" to create Integrator/Filter icons.
REM Logs to <install-prefix>\pds_post_install.log for diagnosis if install fails.

setlocal enabledelayedexpansion

set INSTALL_DIR=%PREFIX%
if not "%~1"=="" set INSTALL_DIR=%~1

set LOG=%INSTALL_DIR%\pds_post_install.log
set PYTHON=%INSTALL_DIR%\python.exe
set SCRIPTS=%INSTALL_DIR%\Scripts

REM Start log (append)
echo === PDS post-install %date% %time% === >> "%LOG%"
echo PREFIX=%INSTALL_DIR% >> "%LOG%"
echo PYTHON=%PYTHON% >> "%LOG%"

REM Validate install prefix and Python
if not exist "%INSTALL_DIR%" (
  echo ERROR: Install prefix not set or not a directory: PREFIX=%INSTALL_DIR% >> "%LOG%"
  echo ERROR: Install prefix not set or not a directory. Check that the installer is run normally. Log: %LOG%
  exit /b 1
)
if not exist "%PYTHON%" (
  echo ERROR: Python not found: %PYTHON% >> "%LOG%"
  echo ERROR: Python not found. Log: %LOG%
  exit /b 1
)

echo PDS post-install: prefix=%INSTALL_DIR%

REM Unzip bundled source if we have the zip (constructor copies files only, so we ship a zip)
if exist "%INSTALL_DIR%\share\pds-src.zip" (
  echo Extracting bundled PDS source...
  echo Extracting bundled PDS source... >> "%LOG%"
  "%PYTHON%" -c "import zipfile; zipfile.ZipFile(r'%INSTALL_DIR%\share\pds-src.zip').extractall(r'%INSTALL_DIR%\share')" >> "%LOG%" 2>&1
)

set PDS_SRC=%INSTALL_DIR%\share\pds-src
if not exist "%PDS_SRC%\pyproject.toml" (
  echo ERROR: Bundled PDS source not found at %PDS_SRC% ^(missing pyproject.toml^). Rebuild the installer. >> "%LOG%"
  echo ERROR: Bundled PDS source not found. Rebuild the installer. Log: %LOG%
  exit /b 1
)

echo Installing PDS from bundled source...
echo Installing PDS from bundled source (%PDS_SRC%)... >> "%LOG%"
"%PYTHON%" -m pip install --force-reinstall "%PDS_SRC%" >> "%LOG%" 2>&1
if errorlevel 1 (
  echo ERROR: pip install failed. >> "%LOG%"
  echo ERROR: pip install failed. Log: %LOG%
  exit /b 1
)

REM Create icons for Integrator and Filter (pds -m)
echo Creating PDS icons (pds -m)...
echo Creating PDS icons (pds -m)... >> "%LOG%"
"%SCRIPTS%\pds.exe" -m >> "%LOG%" 2>&1

echo PDS post-install complete. Log: %LOG%
echo PDS post-install complete. Log: %LOG% >> "%LOG%"
endlocal
