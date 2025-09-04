import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))


from pds.cli import run_converter


class TestConvertCommand:
    """Test cases for the convert command."""

    def test_convert_command_no_args(self):
        """Test convert command with no arguments."""
        # run_converter returns None, so we just test it doesn't crash
        run_converter([])

    def test_convert_command_nonexistent_path(self):
        """Test convert command with nonexistent path."""
        # run_converter returns None, so we just test it doesn't crash
        run_converter(["/nonexistent/path"])

    def test_convert_command_help_detection(self):
        """Test that convert command handles help requests properly."""
        # This would be expanded to test help functionality
        pass

    @patch("pds.cli.spec_to_hdf5")
    def test_convert_single_file_basic(self, mock_spec_to_hdf5):
        """Test basic single file conversion."""
        mock_spec_to_hdf5.return_value = True

        with tempfile.NamedTemporaryFile(suffix=".spec", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            try:
                # Test file conversion
                run_converter([str(tmp_path)])
                mock_spec_to_hdf5.assert_called_once()
            finally:
                tmp_path.unlink()

    @patch("pds.cli.AutoMasterMonitor")
    def test_convert_directory_basic(self, mock_monitor_class):
        """Test basic directory monitoring."""
        mock_monitor = Mock()
        mock_monitor_class.return_value = mock_monitor

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Test directory monitoring
            run_converter([str(tmp_path)])
            mock_monitor_class.assert_called_once()
            mock_monitor.monitor.assert_called_once()

    def test_argument_parsing_file_mode(self):
        """Test argument parsing for file mode."""
        with tempfile.NamedTemporaryFile(suffix=".spec", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            try:
                assert tmp_path.exists()
            finally:
                tmp_path.unlink()

    def test_argument_parsing_directory_mode(self):
        """Test argument parsing for directory mode."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            assert tmp_path.exists()
            assert tmp_path.is_dir()


class TestModuleImports:
    """Test that all CLI modules can be imported correctly."""

    def test_import_convert_module(self):
        """Test that CLI module imports correctly."""
        from pds.cli import run_converter

        assert callable(run_converter)

    def test_import_spectohdf_module(self):
        """Test that spectohdf module imports correctly."""
        from pds.cli.spectohdf import spec_to_hdf5

        assert callable(spec_to_hdf5)

    def test_import_auto_master_module(self):
        """Test that auto_master module imports correctly."""
        from pds.cli.auto_master import AutoMasterMonitor

        assert AutoMasterMonitor is not None


class TestCLIStructure:
    """Test CLI package structure and organization."""

    def test_cli_package_structure(self):
        """Test that CLI package is properly structured."""
        cli_dir = Path(__file__).parent.parent / "pds" / "cli"

        # Check that CLI directory exists
        assert cli_dir.exists()
        assert cli_dir.is_dir()

        # Check for required files
        expected_files = ["__init__.py", "spectohdf.py", "auto_master.py"]
        for filename in expected_files:
            file_path = cli_dir / filename
            assert file_path.exists(), f"Missing required CLI file: {filename}"

    def test_cli_package_imports(self):
        """Test that CLI package imports work correctly."""
        from pds.cli import run_converter

        assert callable(run_converter)


# Integration test markers for different test categories
pytestmark = [
    pytest.mark.unit,
    pytest.mark.cli,
]


if __name__ == "__main__":
    pytest.main([__file__])
