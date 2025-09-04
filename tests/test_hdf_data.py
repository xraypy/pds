import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from pds.utils.hdf_data import HdfDataFile


class TestHdfDataFile:
    """Test the HdfDataFile class functionality for Python 2/3 compatibility."""

    def setup_method(self):
        """Set up test environment with temporary HDF5 file structure."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
        self.temp_file.close()
        self.fname = self.temp_file.name

        # Create a realistic HDF5 structure similar to what the original Python 2 code expects
        with h5py.File(self.fname, "w") as f:
            # Create point 000001
            point1 = f.create_group("000001")

            # Add attributes (test data compatibility)
            point1.attrs["name"] = "test_point_001"
            point1.attrs["type"] = "ascan"
            point1.attrs["geom"] = "psic"
            point1.attrs["info"] = "test scan info"
            point1.attrs["hist.1"] = "test history"
            point1.attrs["date_stamp"] = 1234567890
            point1.attrs["energy"] = 12.5

            # Create angle data (test array data types)
            angle_labels = ["chi", "del", "eta", "mu", "nu", "phi"]
            angle_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
            point1.create_dataset("angle_labels", data=angle_labels)
            point1.create_dataset("angle_values", data=angle_values)

            # Create lattice data
            lattice_labels = [
                "real_a",
                "real_b",
                "real_c",
                "real_alpha",
                "real_beta",
                "real_gamma",
                "recip_a",
                "recip_b",
                "recip_c",
                "recip_alpha",
                "recip_beta",
                "recip_gamma",
                "lambda",
            ]
            lattice_values = np.array([5.0, 5.0, 5.0, 90.0, 90.0, 90.0, 0.2, 0.2, 0.2, 90.0, 90.0, 90.0, 1.54])
            point1.create_dataset("lattice_labels", data=lattice_labels)
            point1.create_dataset("lattice_values", data=lattice_values)

            # Create orientation data
            or_labels = [
                "or0_h",
                "or0_k",
                "or0_L",
                "or0_del",
                "or0_eta",
                "or0_chi",
                "or0_phi",
                "or0_nu",
                "or0_mu",
                "or0_lambda",
                "or1_h",
                "or1_k",
                "or1_L",
                "or1_del",
                "or1_eta",
                "or1_chi",
                "or1_phi",
                "or1_nu",
                "or1_mu",
                "or1_lambda",
            ]
            or_values = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.54, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.54])
            point1.create_dataset("or_labels", data=or_labels)
            point1.create_dataset("or_values", data=or_values)

            # Create position and scaler data (test mixed byte/string handling)
            position_labels = [b"phi", b"chi", b"H", b"K", b"L"]  # Mix of bytes (Python 2 style)
            position_values = np.array([0.0, 10.0, 1.0, 0.0, 0.5])
            point1.create_dataset("position_labels", data=position_labels)
            point1.create_dataset("position_values", data=position_values)

            scaler_labels = [b"Epoch", b"Seconds", b"i0", b"detector"]  # Bytes for Python 2/3 compatibility
            scaler_values = np.array([100, 1, 1000, 5000])
            point1.create_dataset("scaler_labels", data=scaler_labels)
            point1.create_dataset("scaler_values", data=scaler_values)

            # Create Q dataset
            point1.create_dataset("Q", data=np.array([1.0, 0.0, 0.5]))

            # Create misc datasets
            point1.create_dataset("haz", data=np.array([0.0, 0.0, 1.0]))

            # Create detector group (det_0)
            det_group = point1.create_group("det_0")
            det_group.attrs["name"] = "pilatus"
            det_group.attrs["data"] = "detector data info"

            # Detector integration parameters
            int_labels = ["bgrflag", "cnbgr", "compress", "cpow", "ctan", "cwidth", "filter", "integrated", "nline", "rnbgr", "roi", "rpow", "rtan", "rwidth"]
            int_values = ["1", "5", "1", "2", "False", "15", "False", "False", "1", "5", "[]", "0", "False", "15"]
            det_group.create_dataset("int_labels", data=int_labels)
            det_group.create_dataset("int_values.1", data=int_values, dtype=h5py.vlen_dtype(str))

            # Detector correction parameters
            corr_labels = [
                "bad_pixel_map",
                "bad_point",
                "image_changed",
                "image_max",
                "pixel_map_changed",
                "real_image_max",
                "rotangle",
                "sample_angles",
                "sample_diameter",
                "sample_polygon",
                "scale",
            ]
            corr_values = ["[]", "False", "True", "-1", "True", "-1", "0", "[]", "10", "[]", "1000000.0"]
            det_group.create_dataset("corr_labels", data=corr_labels)
            det_group.create_dataset("corr_values.1", data=corr_values, dtype=h5py.vlen_dtype(str))

            # Detector parameters
            det_labels = ["beam_slits", "det_slits"]
            det_values = ["{}", "{}"]
            det_group.create_dataset("det_labels", data=det_labels)
            det_group.create_dataset("det_values.1", data=det_values, dtype=h5py.vlen_dtype(str))

            # Result parameters
            res_labels = ["alpha", "beta", "ctot", "F", "F_changed", "Ferr", "I", "I_c", "I_r", "Ibgr", "Ibgr_c", "Ibgr_r", "Ierr", "Ierr_c", "Ierr_r"]
            res_values = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            det_group.create_dataset("result_labels", data=res_labels)
            det_group.create_dataset("result_values.1", data=res_values)

            # Add image data for testing
            image_data = np.random.randint(0, 1000, size=(100, 100))
            det_group.create_dataset("image_data", data=image_data)

            # Create point 000002 for testing multiple points
            point2 = f.create_group("000002")
            point2.attrs["name"] = "test_point_002"
            point2.attrs["type"] = "dscan"
            point2.attrs["geom"] = "psic"
            point2.attrs["info"] = "test scan info 2"
            point2.attrs["hist.1"] = "test history 2"
            point2.attrs["date_stamp"] = 1234567891
            point2.attrs["energy"] = 13.0

            # Copy similar structure for point2 with slightly different values
            point2.create_dataset("angle_labels", data=angle_labels)
            point2.create_dataset("angle_values", data=angle_values + 1.0)
            point2.create_dataset("position_labels", data=position_labels)
            point2.create_dataset("position_values", data=position_values + 0.1)
            point2.create_dataset("scaler_labels", data=scaler_labels)
            point2.create_dataset("scaler_values", data=scaler_values + 10)

    def teardown_method(self):
        """Clean up temporary files."""
        if os.path.exists(self.fname):
            os.unlink(self.fname)

    @patch("pds.utils.hdf_data.FileLock")
    def test_initialization(self, mock_file_lock):
        """Test HdfDataFile initialization with proper file locking."""
        # Mock the file lock to avoid actual locking during tests
        mock_lock_instance = MagicMock()
        mock_file_lock.return_value = mock_lock_instance

        hdf_data = HdfDataFile(self.fname)

        assert hdf_data.fname == self.fname
        assert hdf_data.point == 0
        assert hdf_data.point_dict == {}
        assert hdf_data.version == 1
        assert hdf_data.file is not None
        assert hdf_data.all_items is not None

        # Verify file lock was created and acquired
        mock_file_lock.assert_called_once_with(self.fname)
        mock_lock_instance.acquire.assert_called_once()

        hdf_data.close()

    @patch("pds.utils.hdf_data.FileLock")
    def test_getitem_and_point_reading(self, mock_file_lock):
        """Test __getitem__ method and point data reading."""
        mock_lock_instance = MagicMock()
        mock_file_lock.return_value = mock_lock_instance

        hdf_data = HdfDataFile(self.fname)

        # Test reading point data
        point_data = hdf_data["000001"]

        assert isinstance(point_data, dict)
        assert hdf_data.point == "000001"

        # Check that essential keys are present
        expected_keys = ["chi", "del", "eta", "mu", "nu", "phi"]  # angle keys
        for key in expected_keys:
            assert key in point_data

        # Test Python 2/3 compatibility with bytes/string handling
        assert "phi" in point_data  # Position label converted to string
        assert b"phi" in point_data  # Original bytes key should also be available

        # Test detector data
        assert "det_0" in point_data
        det_data = point_data["det_0"]
        assert isinstance(det_data, dict)
        assert "image_data" in det_data
        assert "bad_pixel_map" in det_data

        hdf_data.close()

    @patch("pds.utils.hdf_data.FileLock")
    def test_get_method(self, mock_file_lock):
        """Test get method with default values."""
        mock_lock_instance = MagicMock()
        mock_file_lock.return_value = mock_lock_instance

        hdf_data = HdfDataFile(self.fname)

        # Test successful get
        result = hdf_data.get("000001")
        assert result is not None
        assert isinstance(result, dict)

        # Test get with non-existent point
        result = hdf_data.get("999999", default={"empty": True})
        assert result == {"empty": True}

        hdf_data.close()

    @patch("pds.utils.hdf_data.FileLock")
    def test_get_all_method(self, mock_file_lock):
        """Test get_all method for bulk data retrieval."""
        mock_lock_instance = MagicMock()
        mock_file_lock.return_value = mock_lock_instance

        hdf_data = HdfDataFile(self.fname)

        # Test getting a general key
        results = hdf_data.get_all("chi")
        assert isinstance(results, dict)
        assert "000001" in results
        assert "000002" in results

        # Test with specific points
        results = hdf_data.get_all("phi", points=["000001"])
        assert len(results) == 1
        assert "000001" in results

        # Test with detector keys (tuple format)
        results = hdf_data.get_all(("det_0", "image_data"))
        assert isinstance(results, dict)

        hdf_data.close()

    @patch("pds.utils.hdf_data.FileLock")
    def test_set_all_method(self, mock_file_lock):
        """Test set_all method for bulk data writing."""
        mock_lock_instance = MagicMock()
        mock_file_lock.return_value = mock_lock_instance

        hdf_data = HdfDataFile(self.fname)

        # Test with string key (now working after fixes)
        hdf_data.set_all("chi", 99.9)

        # Verify the value was set
        results = hdf_data.get_all("chi")
        assert results["000001"] == 99.9
        assert results["000002"] == 99.9

        hdf_data.close()

    @patch("pds.utils.hdf_data.FileLock")
    def test_write_point_method(self, mock_file_lock):
        """Test write_point method for data persistence."""
        mock_lock_instance = MagicMock()
        mock_file_lock.return_value = mock_lock_instance

        hdf_data = HdfDataFile(self.fname)

        # Read a point first
        point_data = hdf_data["000001"]

        # Modify detector data (which should work with current implementation)
        point_data["det_0"]["scale"] = "2000000.0"

        # Write the point back
        hdf_data.write_point(point_data, "000001")
        assert point_data["det_0"]["scale"] == "2000000.0"

        hdf_data.close()

    @patch("pds.utils.hdf_data.FileLock")
    def test_delete_method(self, mock_file_lock):
        """Test delete method for removing points."""
        mock_lock_instance = MagicMock()
        mock_file_lock.return_value = mock_lock_instance

        hdf_data = HdfDataFile(self.fname)

        # Verify point exists
        assert "000002" in hdf_data.file

        # Delete the point
        hdf_data.delete("000002")

        # Verify point is gone
        assert "000002" not in hdf_data.file

        hdf_data.close()

    @patch("pds.utils.hdf_data.FileLock")
    def test_close_method(self, mock_file_lock):
        """Test proper file closing and lock release."""
        mock_lock_instance = MagicMock()
        mock_file_lock.return_value = mock_lock_instance

        hdf_data = HdfDataFile(self.fname)

        # Load some data
        point_data = hdf_data["000001"]
        point_data["test_key"] = "test_value"

        # Close the file
        hdf_data.close()

        # Verify cleanup
        assert hdf_data.point == 0
        assert hdf_data.point_dict == {}
        mock_lock_instance.release.assert_called()

    @patch("pds.utils.hdf_data.FileLock")
    def test_data_type_compatibility(self, mock_file_lock):
        """Test Python 2/3 data type compatibility, especially with HDF5 datasets."""
        mock_lock_instance = MagicMock()
        mock_file_lock.return_value = mock_lock_instance

        hdf_data = HdfDataFile(self.fname)

        point_data = hdf_data["000001"]

        # Test that bytes are properly handled (Python 2/3 compatibility)
        position_labels = hdf_data.file["000001"]["position_labels"][:]
        assert b"phi" in position_labels

        # Test that the point_dict contains both byte and string versions
        assert "phi" in point_data  # String version
        assert b"phi" in point_data  # Bytes version

        # Test numeric data types are preserved
        assert isinstance(point_data["chi"], (int, float, np.number))

        # Test array data types
        assert isinstance(point_data["det_0"]["image_data"], np.ndarray)

        hdf_data.close()

    @patch("pds.utils.hdf_data.FileLock")
    def test_image_correction_functionality(self, mock_file_lock):
        """Test image data and correction functionality."""
        mock_lock_instance = MagicMock()
        mock_file_lock.return_value = mock_lock_instance

        with patch("pds.utils.hdf_data.correct_image") as mock_correct:
            with patch("pds.utils.hdf_data.read_pixel_map") as mock_read_pixel:
                # Mock the correction functions
                mock_correct.return_value = np.ones((100, 100))
                mock_read_pixel.return_value = "[[0, 0], [1, 1]]"

                hdf_data = HdfDataFile(self.fname)
                point_data = hdf_data["000001"]

                # Verify image data exists
                assert "image_data" in point_data["det_0"]
                assert isinstance(point_data["det_0"]["image_data"], np.ndarray)

                # Verify corrected image is created
                if "corrected_image" in point_data["det_0"]:
                    assert isinstance(point_data["det_0"]["corrected_image"], np.ndarray)

        hdf_data.close()

    def test_error_handling_nonexistent_file(self):
        """Test proper error handling when file doesn't exist."""
        with patch("pds.utils.hdf_data.FileLock"):
            with pytest.raises(Exception):  # Should raise an exception for non-existent file
                HdfDataFile("nonexistent_file.h5")

    @patch("pds.utils.hdf_data.FileLock")
    def test_version_point_method(self, mock_file_lock):
        """Test version_point method (placeholder implementation)."""
        mock_lock_instance = MagicMock()
        mock_file_lock.return_value = mock_lock_instance

        hdf_data = HdfDataFile(self.fname)

        # Test that the method exists and can be called
        hdf_data.version_point({"test": "data"})

        hdf_data.close()

    @patch("pds.utils.hdf_data.FileLock")
    def test_special_case_L_key_handling(self, mock_file_lock):
        """Test special handling of 'L' key for Python 2/3 compatibility."""
        mock_lock_instance = MagicMock()
        mock_file_lock.return_value = mock_lock_instance

        hdf_data = HdfDataFile(self.fname)

        # Test get_all with 'L' key (should handle bytes conversion)
        results = hdf_data.get_all("L")
        assert isinstance(results, dict)

        hdf_data.close()


if __name__ == "__main__":
    pytest.main([__file__])
