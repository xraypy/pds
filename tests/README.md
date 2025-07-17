# Tests for PDS

This directory contains pytest-based tests for the PDS codebase.

## Structure

- `test_background.py` - Tests for the background determination module
- `test_image_data.py` - Tests for the image data handling module

## Running Tests

To run all tests:
```bash
pytest tests/
```

To run specific test files:
```bash
pytest tests/test_background.py
pytest tests/test_image_data.py
```

To run with verbose output:
```bash
pytest tests/ -v
```

## Test Categories

- **Compatibility Tests**: Ensure Python 2/3 compatibility
- **Functional Tests**: Test core functionality
- **Edge Case Tests**: Test boundary conditions
- **Integration Tests**: Test module interactions
- **Performance Tests**: Basic performance benchmarks
