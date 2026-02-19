#!/bin/bash
# PDS post-install script for macOS and Linux.
# Installs PDS from bundled source and runs "pds -m" to create Integrator/Filter icons.
# Logs to <install-prefix>/pds_post_install.log for diagnosis if install fails.

# PREFIX is exported by the constructor installer; fallback to $1 or CONDA_PREFIX
PREFIX="${PREFIX:-${1:-$CONDA_PREFIX}}"
if [[ -n "$PREFIX" && -d "$PREFIX" ]]; then
  LOG="$PREFIX/pds_post_install.log"
else
  LOG="${TMPDIR:-/var/tmp}/pds_post_install.log"
fi
mkdir -p "$(dirname "$LOG")"
touch "$LOG"
exec > >(tee -a "$LOG") 2>&1
echo "=== PDS post-install $(date) ==="

if [[ -z "$PREFIX" || ! -d "$PREFIX" ]]; then
  echo "ERROR: Install prefix not set or not a directory: PREFIX='$PREFIX'"
  echo "Check that the installer is run normally (PREFIX is set by constructor)."
  echo "Log: $LOG"
  exit 1
fi

PYTHON="$PREFIX/bin/python"
BIN="$PREFIX/bin"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found or not executable: $PYTHON"
  exit 1
fi

echo "PREFIX=$PREFIX"
echo "PYTHON=$PYTHON"

# Unzip bundled source if we have the zip (constructor copies files only, so we ship a zip)
if [[ -f "$PREFIX/share/pds-src.zip" ]]; then
  echo "Extracting bundled PDS source..."
  "$PYTHON" -c "import zipfile; zipfile.ZipFile('$PREFIX/share/pds-src.zip').extractall('$PREFIX/share')"
fi
PDS_SRC="$PREFIX/share/pds-src"
if [[ ! -f "$PDS_SRC/pyproject.toml" ]]; then
  echo "ERROR: Bundled PDS source not found at $PDS_SRC (missing pyproject.toml). Rebuild the installer."
  echo "Log: $LOG"
  exit 1
fi
echo "Installing PDS from bundled source ($PDS_SRC)..."
set +e
"$PYTHON" -m pip install "$PDS_SRC"
PIP_ERR=$?
set -e
if [[ $PIP_ERR -ne 0 ]]; then
  echo "ERROR: pip install failed (exit $PIP_ERR). Log: $LOG"
  exit 1
fi

# Create icons for Integrator and Filter (pds -m); skip if root (.pkg runs as root, cannot create user shortcuts)
if [[ "$(id -u)" = "0" ]]; then
  echo "Skipping pds -m (running as root). Run manually: $BIN/pds -m"
else
  echo "Creating PDS icons (pds -m)..."
  set +e
  "$BIN/pds" -m
  set -e
fi

echo "PDS post-install complete. Log: $LOG"
