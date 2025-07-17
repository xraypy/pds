# Tests for PDS

This directory contains pytest-based tests for the PDS codebase.

## Structure

- `test_background.py` - Tests for the background determination module
- `test_file_locker.py` - Tests for the cross-platform file locking module  
- `test_image_data.py` - Tests for the image data handling module
- `test_filtertools.py` - Tests for the HDF5 filtering and date/list operations module

## Running Tests

To run all tests:
```bash
pytest tests/
```

To run specific test files:
```bash
pytest tests/test_background.py
pytest tests/test_file_locker.py
pytest tests/test_image_data.py
pytest tests/test_filtertools.py
```

To run with verbose output:
```bash
pytest tests/ -v
```

## Test Categories

- **Compatibility Tests**: Ensure Python 2/3 compatibility (especially in `test_image_data.py`)
- **Functional Tests**: Test core functionality across all modules
- **Edge Case Tests**: Test boundary conditions and error handling
- **Integration Tests**: Test module interactions
- **Concurrency Tests**: Test thread-safe operations (in `test_file_locker.py`)
- **Cross-platform Tests**: Test OS-specific functionality (file locking, path handling)
- **Performance Tests**: Basic performance benchmarks

## Special Test Features

### HDF5 Compatibility (`test_image_data.py`)
- Tests for Python 2→3 migration string/bytes compatibility
- Unicode handling in PyTables group names
- Comprehensive data type conversion testing

### File Locking (`test_file_locker.py`)
- Concurrent access testing with thread synchronization
- Cross-platform username/hostname detection
- Context manager and manual lock/release patterns

### HDF5 Filtering (`test_filtertools.py`)
- Tests for Python 2→3 migration with original behavior preservation
- Date parsing and comparison functions (is_before, is_after)
- List operations (union, intersection) with None value handling
- HDF5 scan filtering with string/bytes compatibility
- Point-level and scan-level filtering validation
- Numeric type compatibility (numpy integers, floats)
- Original Python 2 eval() expression compatibility
- Timeout and error condition handling
