# Tests for PDS

This directory contains pytest-based tests for the PDS codebase.

## Structure

- `test_background.py` - Tests for the background determination module

## Running Tests

To run all tests:
```bash
pytest tests/
```

To run specific test files:
```bash
pytest tests/test_background.py
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
