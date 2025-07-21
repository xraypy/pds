import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from pds.utils.master_to_project import CORRECTION_PARAMETERS, DETECTOR_PARAMETERS, INTEGRATION_PARAMETERS, master_to_project, read_pixel_map


class TestReadPixelMap:
    """Test pixel map file reading function."""

    def test_valid_pixel_map_file(self, tmp_path):
        """Test reading a valid pixel map file."""
        # Create a test pixel map file
        pixel_map_content = "10,20 30,40\n50,60 70,80\n90,100\n"
        pixel_map_file = tmp_path / "test_pixel_map.txt"
        pixel_map_file.write_text(pixel_map_content)

        bad_pixels, good_pixels = read_pixel_map(str(pixel_map_file))

        expected_bad = [[10, 20], [50, 60], [90, 100]]
        expected_good = [[30, 40], [70, 80]]

        assert bad_pixels == expected_bad
        assert good_pixels == expected_good

    def test_pixel_map_with_only_bad_pixels(self, tmp_path):
        """Test pixel map file with only bad pixels (no good pixels)."""
        pixel_map_content = "10,20\n50,60\n90,100\n"
        pixel_map_file = tmp_path / "test_pixel_map.txt"
        pixel_map_file.write_text(pixel_map_content)

        bad_pixels, good_pixels = read_pixel_map(str(pixel_map_file))

        expected_bad = [[10, 20], [50, 60], [90, 100]]
        expected_good = []

        assert bad_pixels == expected_bad
        assert good_pixels == expected_good

    def test_empty_pixel_map_file(self, tmp_path):
        """Test reading an empty pixel map file."""
        pixel_map_file = tmp_path / "empty_pixel_map.txt"
        pixel_map_file.write_text("")

        bad_pixels, good_pixels = read_pixel_map(str(pixel_map_file))

        assert bad_pixels == []
        assert good_pixels == []

    def test_nonexistent_pixel_map_file(self, capsys):
        """Test reading a nonexistent pixel map file."""
        result = read_pixel_map("nonexistent_file.txt")

        captured = capsys.readouterr()
        assert "Error reading file: nonexistent_file.txt" in captured.out
        assert result == []

    def test_malformed_pixel_map_file(self, tmp_path, capsys):
        """Test reading a malformed pixel map file."""
        pixel_map_content = "invalid,content,here\n10,20,30,40,50\n"
        pixel_map_file = tmp_path / "malformed_pixel_map.txt"
        pixel_map_file.write_text(pixel_map_content)

        # This should raise an exception when trying to convert non-integer values
        result = read_pixel_map(str(pixel_map_file))

        captured = capsys.readouterr()
        assert "Error reading file:" in captured.out
        assert result == []

    def test_pixel_map_with_whitespace(self, tmp_path):
        """Test pixel map file with various whitespace."""
        pixel_map_content = "  10,20   30,40  \n  50,60  70,80\n  90,100  \n"
        pixel_map_file = tmp_path / "whitespace_pixel_map.txt"
        pixel_map_file.write_text(pixel_map_content)

        bad_pixels, good_pixels = read_pixel_map(str(pixel_map_file))

        expected_bad = [[10, 20], [50, 60], [90, 100]]
        expected_good = [[30, 40], [70, 80]]

        assert bad_pixels == expected_bad
        assert good_pixels == expected_good


class TestMasterToProject:
    """Test master to project conversion function."""

    def create_mock_master_file(self, tmp_path, spec_name="spec_file_001", scan_num="1"):
        """Create a mock master HDF5 file for testing."""
        master_file = tmp_path / "test_master.h5"

        with h5py.File(master_file, "w") as f:
            # Create spec file group
            spec_group = f.create_group(spec_name)

            # Create scan group
            scan_group = spec_group.create_group(scan_num)
            scan_group.attrs["s_type"] = "ascan"
            scan_group.attrs["cmd"] = "ascan phi 0 10 10 1"
            scan_group.attrs["init_epoch"] = 1000000000
            scan_group.attrs["energy"] = 12.5

            # Create point labels
            point_labs = [b"phi", b"chi", b"H", b"K", b"L", b"Epoch", b"detector"]
            scan_group.create_dataset("point_labs", data=point_labs)

            # Create parameter labels
            param_labs = [
                b"g_aa",
                b"g_bb",
                b"g_cc",
                b"g_al",
                b"g_be",
                b"g_ga",
                b"g_aa_s",
                b"g_bb_s",
                b"g_cc_s",
                b"g_al_s",
                b"g_be_s",
                b"g_ga_s",
                b"g_LAMBDA",
                b"g_h0",
                b"g_k0",
                b"g_l0",
                b"g_h1",
                b"g_k1",
                b"g_l1",
                b"chi",
                b"TwoTheta",
                b"theta",
                b"Psi",
                b"Nu",
                b"phi",
                b"g_haz",
                b"g_kaz",
                b"g_laz",
            ]
            scan_group.create_dataset("param_labs", data=param_labs)

            # Create point data (2 points)
            point_data = np.array(
                [
                    [0.0, 10.0, 1.0, 0.0, 0.5, 100, 1000],  # Point 1
                    [5.0, 10.0, 1.0, 0.0, 0.5, 200, 2000],  # Point 2
                ]
            )
            scan_group.create_dataset("point_data", data=point_data)

            # Create parameter data
            param_data = np.array(
                [
                    5.0,
                    5.0,
                    5.0,
                    90.0,
                    90.0,
                    90.0,  # Real lattice
                    0.2,
                    0.2,
                    0.2,
                    90.0,
                    90.0,
                    90.0,  # Reciprocal lattice
                    1.54,  # Lambda
                    1.0,
                    0.0,
                    0.0,  # H0, K0, L0
                    0.0,
                    1.0,
                    0.0,  # H1, K1, L1
                    0.0,
                    10.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,  # Angles
                    0.0,
                    0.0,
                    1.0,  # Haz vector
                ]
            )
            scan_group.create_dataset("param_data", data=param_data)

            # Optionally add image data
            if hasattr(self, "add_image_data") and self.add_image_data:
                image_data = np.random.randint(0, 1000, size=(2, 100, 100))
                scan_group.create_dataset("image_data", data=image_data)

        return str(master_file)

    def test_master_to_project_basic(self, tmp_path):
        """Test basic master to project conversion."""
        master_file = self.create_mock_master_file(tmp_path)
        project_file = tmp_path / "test_project.h5"

        desired_scans = {"spec_file_001": {"1": {}}}

        master_to_project(master_file, desired_scans, str(project_file), append=False, gui=False)

        # Verify project file was created
        assert project_file.exists()

        # Check the contents
        with h5py.File(project_file, "r") as f:
            # Should have 2 points (000001 and 000002)
            assert "000001" in f
            assert "000002" in f

            # Check point attributes
            point1 = f["000001"]
            assert "name" in point1.attrs
            assert "type" in point1.attrs
            assert "geom" in point1.attrs
            assert point1.attrs["type"] == "ascan"
            assert point1.attrs["geom"] == "psic"

            # Check datasets exist
            assert "angle_labels" in point1
            assert "angle_values" in point1
            assert "lattice_labels" in point1
            assert "lattice_values" in point1
            assert "Q" in point1
            assert "det_0" in point1

    def test_master_to_project_with_attributes(self, tmp_path):
        """Test master to project conversion with custom attributes."""
        master_file = self.create_mock_master_file(tmp_path)
        project_file = tmp_path / "test_project.h5"

        desired_scans = {"spec_file_001": {"1": {"geom": "custom_geom", "bgrflag": 2, "scale": 2.0e6}}}

        master_to_project(master_file, desired_scans, str(project_file), append=False, gui=False)

        with h5py.File(project_file, "r") as f:
            point1 = f["000001"]
            assert point1.attrs["geom"] == "custom_geom"

            # Check detector integration parameters
            det0 = point1["det_0"]
            int_values = [v.decode() if isinstance(v, bytes) else v for v in det0["int_values.1"]]
            int_labels = [label.decode() if isinstance(label, bytes) else label for label in det0["int_labels"]]
            bgrflag_idx = int_labels.index("bgrflag")
            assert int_values[bgrflag_idx] == "2"

    def test_master_to_project_append_mode(self, tmp_path):
        """Test master to project conversion in append mode."""
        master_file = self.create_mock_master_file(tmp_path)
        project_file = tmp_path / "test_project.h5"

        desired_scans = {"spec_file_001": {"1": {}}}

        # First conversion
        master_to_project(master_file, desired_scans, str(project_file), append=False, gui=False)

        # Create a second master file with different data
        master_file2 = self.create_mock_master_file(tmp_path, "spec_file_002", "1")
        desired_scans2 = {"spec_file_002": {"1": {}}}

        # Append second conversion
        master_to_project(master_file2, desired_scans2, str(project_file), append=True, gui=False)

        with h5py.File(project_file, "r") as f:
            # Should have 4 points now (2 from each file)
            assert "000001" in f
            assert "000002" in f
            assert "000003" in f
            assert "000004" in f

    def test_master_to_project_overwrite_mode(self, tmp_path):
        """Test master to project conversion in overwrite mode."""
        master_file = self.create_mock_master_file(tmp_path)
        project_file = tmp_path / "test_project.h5"

        desired_scans = {"spec_file_001": {"1": {}}}

        # First conversion
        master_to_project(master_file, desired_scans, str(project_file), append=False, gui=False)

        # Second conversion with overwrite
        master_file2 = self.create_mock_master_file(tmp_path, "spec_file_002", "1")
        desired_scans2 = {"spec_file_002": {"1": {}}}

        master_to_project(master_file2, desired_scans2, str(project_file), append=False, gui=False)

        with h5py.File(project_file, "r") as f:
            # Should only have 2 points (from second file)
            assert "000001" in f
            assert "000002" in f
            assert "000003" not in f

            # Check that the data is from the second file
            point1 = f["000001"]
            point_name = point1.attrs["name"]
            if isinstance(point_name, bytes):
                point_name = point_name.decode()
            assert "spec_file_002:S1:P1" in point_name

    @patch("wx.ProgressDialog")
    @patch("wx.GetApp")
    def test_master_to_project_gui_mode(self, mock_get_app, mock_progress_dialog, tmp_path):
        """Test master to project conversion with GUI progress dialog."""
        # Setup mocks
        mock_app = MagicMock()
        mock_get_app.return_value = mock_app
        mock_progress = MagicMock()
        mock_progress.Update.return_value = (True, None)  # Continue processing
        mock_progress_dialog.return_value = mock_progress

        master_file = self.create_mock_master_file(tmp_path)
        project_file = tmp_path / "test_project.h5"

        desired_scans = {"spec_file_001": {"1": {}}}

        # Mock wx import inside the function
        mock_wx_module = MagicMock()
        mock_wx_module.ProgressDialog = mock_progress_dialog
        mock_wx_module.GetApp = mock_get_app
        mock_wx_module.PD_CAN_ABORT = 1
        mock_wx_module.PD_APP_MODAL = 2
        mock_wx_module.PD_AUTO_HIDE = 4
        mock_wx_module.PD_ELAPSED_TIME = 8
        mock_wx_module.PD_REMAINING_TIME = 16

        with patch.dict("sys.modules", {"wx": mock_wx_module}):
            master_to_project(master_file, desired_scans, str(project_file), append=False, gui=True)

        # Verify GUI components were called
        mock_progress_dialog.assert_called_once()
        assert mock_progress.Update.call_count >= 2  # At least 2 points processed
        mock_progress.Destroy.assert_called_once()

    def test_master_to_project_with_pixel_map(self, tmp_path):
        """Test master to project conversion with pixel map file."""
        # Create pixel map file
        pixel_map_content = "10,20 30,40\n50,60 70,80\n"
        pixel_map_file = tmp_path / "test_pixel_map.txt"
        pixel_map_file.write_text(pixel_map_content)

        master_file = self.create_mock_master_file(tmp_path)
        project_file = tmp_path / "test_project.h5"

        desired_scans = {"spec_file_001": {"1": {"bad_pixel_map": str(pixel_map_file)}}}

        master_to_project(master_file, desired_scans, str(project_file), append=False, gui=False)

        with h5py.File(project_file, "r") as f:
            point1 = f["000001"]
            det0 = point1["det_0"]

            # Check that pixel map was processed
            corr_values = [v.decode() if isinstance(v, bytes) else v for v in det0["corr_values.1"]]
            corr_labels = [label.decode() if isinstance(label, bytes) else label for label in det0["corr_labels"]]
            pixel_map_idx = corr_labels.index("bad_pixel_map")

            # Should contain the pixel map data as a string
            assert "[[10, 20], [50, 60]]" in corr_values[pixel_map_idx]

    def test_master_to_project_duplicate_points(self, tmp_path):
        """Test master to project conversion with duplicate point handling."""
        master_file = self.create_mock_master_file(tmp_path)
        project_file = tmp_path / "test_project.h5"

        desired_scans = {"spec_file_001": {"1": {}}}

        # First conversion
        master_to_project(master_file, desired_scans, str(project_file), append=False, gui=False)

        # Try to append the same data - should detect duplicates
        with patch("builtins.print") as mock_print:
            mock_progress = MagicMock()
            mock_progress.Update.return_value = (True, None)

            mock_wx_module = MagicMock()
            mock_wx_module.ProgressDialog.return_value = mock_progress
            mock_wx_module.GetApp.return_value = MagicMock()
            mock_wx_module.PD_CAN_ABORT = 1
            mock_wx_module.PD_APP_MODAL = 2
            mock_wx_module.PD_AUTO_HIDE = 4
            mock_wx_module.PD_ELAPSED_TIME = 8
            mock_wx_module.PD_REMAINING_TIME = 16

            with patch.dict("sys.modules", {"wx": mock_wx_module}):
                master_to_project(master_file, desired_scans, str(project_file), append=True, gui=True)

            # Should print messages about existing points
            print_calls = [str(call) for call in mock_print.call_args_list]
            # Check if any print call mentions "already in"
            already_in_calls = [call for call in print_calls if "already in" in call]
            assert len(already_in_calls) > 0

    def test_master_to_project_missing_master_file(self, tmp_path):
        """Test master to project conversion with missing master file."""
        project_file = tmp_path / "test_project.h5"

        desired_scans = {"spec_file_001": {"1": {}}}

        with pytest.raises(OSError):
            master_to_project("nonexistent_master.h5", desired_scans, str(project_file), append=False, gui=False)

    def test_master_to_project_with_image_data(self, tmp_path):
        """Test master to project conversion with image data."""
        self.add_image_data = True
        master_file = self.create_mock_master_file(tmp_path)
        project_file = tmp_path / "test_project.h5"

        desired_scans = {"spec_file_001": {"1": {}}}

        master_to_project(master_file, desired_scans, str(project_file), append=False, gui=False)

        with h5py.File(project_file, "r") as f:
            point1 = f["000001"]
            det0 = point1["det_0"]

            # Check that image data was included
            assert "image_data" in det0
            image_data = det0["image_data"][:]
            assert image_data.shape == (100, 100)

    def test_master_to_project_error_handling(self, tmp_path):
        """Test master to project conversion error handling."""
        master_file = self.create_mock_master_file(tmp_path)
        project_file = tmp_path / "test_project.h5"

        # Create malformed desired_scans that will cause an error
        desired_scans = {"nonexistent_spec": {"1": {}}}

        with pytest.raises(KeyError):
            master_to_project(master_file, desired_scans, str(project_file), append=False, gui=False)


class TestConstants:
    """Test module constants."""

    def test_integration_parameters(self):
        """Test INTEGRATION_PARAMETERS constant."""
        assert isinstance(INTEGRATION_PARAMETERS, dict)
        assert "geom" in INTEGRATION_PARAMETERS
        assert INTEGRATION_PARAMETERS["geom"] == "psic"
        assert "bgrflag" in INTEGRATION_PARAMETERS
        assert "roi" in INTEGRATION_PARAMETERS

    def test_correction_parameters(self):
        """Test CORRECTION_PARAMETERS constant."""
        assert isinstance(CORRECTION_PARAMETERS, dict)
        assert "bad_pixel_map" in CORRECTION_PARAMETERS
        assert "scale" in CORRECTION_PARAMETERS
        assert CORRECTION_PARAMETERS["scale"] == 1.0e6

    def test_detector_parameters(self):
        """Test DETECTOR_PARAMETERS constant."""
        assert isinstance(DETECTOR_PARAMETERS, dict)
        assert "name" in DETECTOR_PARAMETERS
        assert DETECTOR_PARAMETERS["name"] == "pilatus"
        assert "beam_slits" in DETECTOR_PARAMETERS
        assert "det_slits" in DETECTOR_PARAMETERS


if __name__ == "__main__":
    pytest.main([__file__])
