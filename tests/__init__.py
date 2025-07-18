import sys
from pathlib import Path


def run_all_tests() -> None:
    """Run all tests in the tests directory using pytest."""
    try:
        import pytest

        # Get the tests directory
        tests_dir = Path(__file__).parent

        print(f"Running all tests in {tests_dir}...")

        # Run pytest with the tests directory, verbose output
        exit_code = pytest.main([str(tests_dir), "-v"])
        sys.exit(exit_code)

    except ImportError:
        print("pytest is not installed. Please install it with: pip install pytest")
        sys.exit(1)
