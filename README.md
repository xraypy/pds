# PDS - Surface Scattering Integration

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive Python toolkit for X-ray scattering data analysis, providing both GUI and CLI interfaces for SPEC file conversion, data processing, and integration analysis.

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [📖 Usage](#-usage)
- [🧪 Examples](#-examples)
- [🔍 Advanced Options](#-advanced-options)
- [🛠️ Development](#️-development)
- [🐛 Troubleshooting](#-troubleshooting)
- [📄 License](#-license)

## 🚀 Quick Start

### Installation

1. **Prerequisites**: Ensure [Anaconda/Miniconda](https://docs.conda.io/en/latest/miniconda.html) is installed

2. **Create Environment**:
   ```bash
   conda create -n pdsENV python=3.13
   conda activate pdsENV
   conda config --add channels conda-forge
   conda install -c conda-forge h5py matplotlib numpy pillow pytest scipy tables wxpython
   pip install pyshortcuts
   ```

3. **Install PDS**:
   ```bash
   pip install -e .
   ```

4. **Verify Installation**:
   ```bash
   pds --help
   ```

## 📖 Usage

### Command Line Interface

After installation, PDS provides a unified `pds` command with multiple modes:

```bash
# Show all available commands
pds --help

# Show detailed convert options
pds --convert help
```

### GUI Applications

#### Integrator GUI
```bash
pds --integrator
# or short form:
pds -i
```

#### Filter GUI
```bash
pds --filter
# or short form:
pds -f
```

### SPEC File Conversion

PDS can convert SPEC files to HDF5 format with full metadata and image data preservation:

#### Single File Conversion
```bash
# Convert single SPEC file
pds --convert data.spec

# Convert with custom output name
pds --convert data.spec output.mh5

# Overwrite existing file
pds --convert data.spec -w

# Specify custom image directory
pds --convert data.spec -i /path/to/images
```

#### Batch Directory Processing
```bash
# Convert all SPEC files in directory once and exit
pds --convert /path/to/spec/files --once

# Monitor directory for new/modified files (continuous)
pds --convert /path/to/spec/files

# Monitor with custom interval
pds --convert /path/to/spec/files -i 30

# Process existing files first, then monitor
pds --convert /path/to/spec/files --process-existing
```

### Testing

```bash
# Run all tests
pds --test
# or short form:
pds -t
```

### Desktop Shortcuts

```bash
# Create desktop shortcuts for GUIs
pds --make_icon
# or short form:
pds -m
```

## 🧪 Examples

### Typical Workflow

1. **Setup Environment**:
   ```bash
   conda activate pdsENV
   cd /path/to/your/experiment
   ```

2. **Convert Data**:
   ```bash
   # For completed experiments
   pds --convert . --once
   
   # For live experiments
   pds --convert . --process-existing
   ```

3. **Launch Analysis GUI**:
   ```bash
   pds --integrator
   ```

### Real-World Usage

Based on our testing with real scientific data:

```bash
# Convert a single experimental run
pds --convert uo2-29a_1d_O2_1.spec
# ✅ Processed 43 scans with image data → 89 MB HDF5 file

# Batch convert entire experimental directory
pds --convert /path/to/experiment/ --once
# ✅ Processed 243+ scans → 490+ MB total data

# Monitor live experiment data
pds --convert /beamline/data --process-existing -i 10
# ✅ Process existing data, then monitor every 10 seconds
```

### Conversion Results
Typical conversion performance:
- **43 scans** → 89 MB HDF5 (0.17 minutes)
- **70 scans** → 134 MB HDF5 (0.24 minutes)
- **86 scans** → 164 MB HDF5 (0.31 minutes)

## 🔍 Advanced Options

### Convert Command Options

#### Single File Options:
- `-w` - Overwrite existing output file
- `-a` - Append to existing output file (default)
- `-q` - Quiet mode (suppress verbose output)
- `-i <image_dir>` - Custom image directory path

#### Directory Monitoring Options:
- `-i <seconds>` - Monitoring interval (default: 20 seconds)
- `-q` - Quiet mode
- `--process-existing` - Process existing files before monitoring
- `--once` - Process existing files once and exit

## 🛠️ Development

### Running Tests
```bash
# Run all tests
python -m pytest

# Run specific test categories
python -m pytest -m "not slow"          # Skip slow tests
python -m pytest -m "unit"              # Unit tests only
python -m pytest -m "integration"       # Integration tests
python -m pytest -m "cli"               # CLI tests
```

### Code Quality
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linting
ruff check .

# Run formatting
ruff format .

# Pre-commit hooks
pre-commit install
```

## 🐛 Troubleshooting

### Common Issues

**Environment Setup Issues**:
```bash
# Ensure correct Python version
python --version  # Should show 3.13.x

# Verify conda environment
conda list | grep h5py  # Should show h5py ≥3.14.0
```

**Conversion Issues**:
```bash
# Check file permissions
ls -la your_spec_file.spec

# Verify SPEC file format
head -20 your_spec_file.spec | grep "#S"

# Use verbose mode for debugging
pds --convert your_file.spec -v
```

**GUI Issues**:
```bash
# Test wxPython installation
python -c "import wx; print('wxPython OK')"

# Try command line mode if GUI fails
pds --convert your_data --once
```

### Getting Help

```bash
# General help
pds --help

# Detailed convert help
pds --convert help

# Test installation
pds --test
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

For additional support, please open an issue on the [GitHub repository](https://github.com/xraypy/pds).
