@echo off
REM Build PDS installer with conda constructor (Windows .exe).
REM Run from the repo root. Requires conda with constructor and Python.

setlocal enabledelayedexpansion

cd /d "%~dp0"

set CONDA_ENV=%CONDA_ENV%
if "%CONDA_ENV%"=="" set CONDA_ENV=pdsENV

set OUTPUT_DIR=%OUTPUT_DIR%
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=dist

REM Bundle PDS source into a zip (constructor's extra_files only copies files, not dirs)
echo Bundling PDS source into constructor\pds-src.zip ...
if exist constructor\pds-src-bundle rmdir /s /q constructor\pds-src-bundle
if exist constructor\pds-src.zip del /q constructor\pds-src.zip

mkdir constructor\pds-src-bundle\pds-src
xcopy /e /i /q pds constructor\pds-src-bundle\pds-src\pds\
copy /y pyproject.toml README.md LICENSE constructor\pds-src-bundle\pds-src\

python -c "import zipfile; from pathlib import Path; src = Path('constructor/pds-src-bundle/pds-src'); zf = zipfile.ZipFile('constructor/pds-src.zip', 'w', zipfile.ZIP_DEFLATED); [zf.write(f, f.relative_to(src.parent).as_posix()) for f in src.rglob('*') if f.is_file()]; zf.close()"

rmdir /s /q constructor\pds-src-bundle

echo Using conda environment: %CONDA_ENV%
echo Building from constructor\ -^> %OUTPUT_DIR%\ ...
conda run -n "%CONDA_ENV%" constructor constructor/ --output-dir="%OUTPUT_DIR%"

echo.
echo Done. Installer(s) in %OUTPUT_DIR%\
endlocal
