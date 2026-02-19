#!/usr/bin/env bash
# Build PDS installer with conda constructor (macOS .pkg or Windows .exe).
set -e
cd "$(dirname "$0")"

CONDA_ENV="${CONDA_ENV:-pdsENV}"
OUTPUT_DIR="${OUTPUT_DIR:-dist}"

# Bundle PDS source into a zip (constructor's extra_files only copies files, not dirs)
echo "Bundling PDS source into constructor/pds-src.zip ..."
rm -rf constructor/pds-src-bundle constructor/pds-src.zip
mkdir -p constructor/pds-src-bundle/pds-src
cp -r pds pyproject.toml README.md LICENSE constructor/pds-src-bundle/pds-src/
(cd constructor/pds-src-bundle && zip -rq ../pds-src.zip pds-src)
rm -rf constructor/pds-src-bundle

echo "Using conda environment: $CONDA_ENV"
echo "Building from constructor/ -> $OUTPUT_DIR/ ..."
conda run -n "$CONDA_ENV" constructor constructor/ --output-dir="$OUTPUT_DIR"

echo ""
echo "Done. Installer(s) in $OUTPUT_DIR/"
