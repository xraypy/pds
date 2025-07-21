# Tests for PDS

This directory contains pytest-based tests for the PDS codebase and is structured as a Python package.

## Structure

- `__init__.py` - Package initialization, shared test utilities, and test runner
- `test_active_area.py` - Tests for the active area calculation module
- `test_background.py` - Tests for the background determination module
- `test_converters.py` - Tests for the data conversion utility functions
- `test_ctr_data.py` - Tests for the CTR data processing module
- `test_file_locker.py` - Tests for the cross-platform file locking module  
- `test_filtertools.py` - Tests for the HDF5 filtering and date/list operations module
- `test_gonio_psic.py` - Tests for the PSIC 6-circle diffractometer geometry calculations
- `test_image_data.py` - Tests for the image data handling module
- `test_lattice.py` - Tests for the crystallographic lattice calculations module
- `test_mathutil.py` - Tests for mathematical utility functions and numpy/scipy wrappers
- `test_polygon.py` - Tests for generalized polygon computations and geometric operations

## Running Tests

To run all tests using the PDS CLI:
```bash
python pds.py -t
```

To run all tests using the test package directly:
```bash
python -c "from tests import run_all_tests; run_all_tests()"
```

To run all tests using pytest directly:
```bash
pytest tests/
```

To run specific test files:
```bash
pytest tests/test_active_area.py
pytest tests/test_background.py
pytest tests/test_converters.py
pytest tests/test_ctr_data.py
pytest tests/test_file_locker.py
pytest tests/test_filtertools.py
pytest tests/test_gonio_psic.py
pytest tests/test_image_data.py
pytest tests/test_lattice.py
pytest tests/test_mathutil.py
pytest tests/test_polygon.py
```

To run with verbose output:
```bash
pytest tests/ -v
```

## General Test Information

The test suite provides comprehensive coverage of the PDS scientific computing modules. Tests validate core functionality, edge cases, and ensure Python 2/3 compatibility across all modules.
