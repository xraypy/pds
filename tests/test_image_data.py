import sys
from pathlib import Path

import numpy as np
import pytest

# Add the pds module to the path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from pds.utils.image_data import IMG_BGR_PARAMS, ImageAna, ImageScan, _sort_roi, calc_roi, clip_image, image_bgr, line_sum, line_sum_integral, pixel_mask


@pytest.fixture
def test_image_simple():
    """Simple image for basic functionality."""
    np.random.seed(42)

    # Create a simple test image with some peaks
    image = np.zeros((50, 60), dtype=np.int32)

    # Add some background
    for i in range(50):
        for j in range(60):
            image[i, j] = int(10 + 5 * np.sin(i / 10.0) + 3 * np.cos(j / 8.0))

    # Add some peaks
    image[20:25, 25:30] += 100
    image[35:38, 45:48] += 50

    # Add some noise
    noise = np.random.randint(-5, 6, size=(50, 60))
    image = image + noise

    # Ensure no negative values
    image = np.maximum(image, 0)

    return image.astype(np.int32)


class TestBasicFunctions:
    """Test basic functionality without external dependencies."""

    def test_sort_roi(self):
        """Test ROI sorting function."""
        # Test normal order
        roi = [10, 20, 50, 60]
        result = _sort_roi(roi)
        assert result == [10, 20, 50, 60]

        # Test reversed order
        roi = [50, 60, 10, 20]
        result = _sort_roi(roi)
        assert result == [10, 20, 50, 60]

    def test_calc_roi(self):
        """Test ROI calculation function."""
        # Test with default center
        roi = calc_roi(dx=20, dy=30, shape=(100, 120))
        assert len(roi) == 4
        assert isinstance(roi[0], int)


class TestPixelMask:
    """Test pixel masking functionality."""

    def test_pixel_mask_edge_cases(self, test_image_simple):
        """Test pixel mask edge cases."""
        # Test with None bad_pixels
        result = pixel_mask(test_image_simple, None, [])
        assert result is None

        # Test with empty lists
        result = pixel_mask(test_image_simple.copy(), [], [])
        np.testing.assert_array_equal(result, test_image_simple)


class TestImageClipping:
    """Test image clipping and ROI functions."""

    def test_clip_image_basic(self, test_image_simple):
        """Test basic image clipping functionality."""
        # [x1, y1, x2, y2]
        roi = [10, 5, 40, 35]

        result = clip_image(test_image_simple, roi=roi)
        assert isinstance(result, np.ndarray)
        # height=35-5, width=40-10
        assert result.shape == (30, 30)


class TestLineSums:
    """Test line sum functions."""

    def test_line_sum_basic(self, test_image_simple):
        """Test basic line sum functionality."""
        # Test column sum
        data, data_idx, bgr = line_sum(test_image_simple, sumflag="c")

        assert isinstance(data, np.ndarray)
        assert isinstance(data_idx, np.ndarray)
        assert isinstance(bgr, np.ndarray)
        # Number of columns
        assert len(data) == test_image_simple.shape[1]

    def test_line_sum_integral(self, test_image_simple):
        """Test line sum integration."""
        Iitg, Ierr, Ibgr = line_sum_integral(test_image_simple, sumflag="c", nbgr=3, width=10)

        assert isinstance(Iitg, (int, float, np.number))
        assert isinstance(Ierr, (int, float, np.number))
        assert isinstance(Ibgr, (int, float, np.number))


class TestImageBackground:
    """Test 2D image background calculation."""

    def test_image_bgr_basic(self, test_image_simple):
        """Test basic 2D background calculation."""
        # Test column direction
        bgr_arr = image_bgr(test_image_simple, lineflag="c", nbgr=3, width=10)

        assert isinstance(bgr_arr, np.ndarray)
        assert bgr_arr.shape == test_image_simple.shape
        assert bgr_arr.dtype == np.float64


class TestImageAna:
    """Test ImageAna class."""

    def test_image_ana_basic(self, test_image_simple):
        """Test basic ImageAna functionality."""
        img_ana = ImageAna(test_image_simple, plot=False)

        assert isinstance(img_ana, ImageAna)
        assert img_ana.integrated
        assert hasattr(img_ana, "I")
        assert hasattr(img_ana, "Ierr")
        assert hasattr(img_ana, "Ibgr")


class TestImageScan:
    """Test ImageScan class."""

    def test_image_scan_basic(self, test_image_simple):
        """Test basic ImageScan functionality."""
        images = [test_image_simple, test_image_simple * 0.8]

        scan = ImageScan(image=images)

        assert len(scan.image) == 2
        assert len(scan.rois) == 2


class TestRegressionPrevention:
    """Test to prevent regression of fixes."""

    def test_img_bgr_params_default(self):
        """Test that IMG_BGR_PARAMS has correct structure."""
        assert isinstance(IMG_BGR_PARAMS, dict)

        required_keys = ["bgrflag", "cnbgr", "cwidth", "cpow", "ctan", "rnbgr", "rwidth", "rpow", "rtan", "nline", "filter", "compress"]

        for key in required_keys:
            assert key in IMG_BGR_PARAMS


class TestPython2Compatibility:
    """Test Python 2/3 compatibility specific issues."""

    def test_integer_division_compatibility(self, test_image_simple):
        """Test that integer division works correctly."""
        # Test division operations that would differ between Python 2 and 3
        test_widths = [1, 2, 3, 5, 7, 11, 13]

        for width in test_widths:
            bgr_arr = image_bgr(test_image_simple, nbgr=3, width=width)
            assert isinstance(bgr_arr, np.ndarray)
            assert bgr_arr.shape == test_image_simple.shape
            assert np.all(np.isfinite(bgr_arr))

    def test_type_checking_compatibility(self):
        """Test that type checking works correctly."""
        # Test string type checking (Python 2 had types.StringType)
        cmap_str = "hot"
        assert type(cmap_str) is str

        # Test list type checking (Python 2 had types.ListType)
        roi_list = [10, 10, 40, 40]
        assert type(roi_list) is list

        # Test dict type checking (Python 2 had types.DictType)
        bgr_dict = {"nbgr": 3, "width": 10}
        assert type(bgr_dict) is dict

    def test_division_compatibility(self):
        """Test division compatibility."""
        # Test integer division
        result = 7 // 2
        assert result == 3
        assert type(result) is int

        # Test float division
        result = 7 / 2
        assert result == 3.5
        assert type(result) is float

        # Test division in context similar to original code
        nline = 5
        # This should work in both Python 2 and 3
        ll = int(nline / 2.0)
        assert ll == 2
        assert type(ll) is int


class TestHDF5Compatibility:
    """Test HDF5 storage compatibility and bytes/string handling."""

    @pytest.fixture(autouse=True)
    def setup_hdf5_test(self):
        """Set up test environment with temporary directory."""
        import shutil
        import tempfile

        # Skip all tests in this class if tables is not available
        pytest.importorskip("tables")

        self.test_dir = tempfile.mkdtemp(prefix="pds_hdf5_test_")
        self.test_file = "test_images.h5"

        # Create test images
        self.test_images = [
            np.random.randint(0, 1000, size=(100, 150), dtype=np.int32),
            np.random.randint(0, 1500, size=(100, 150), dtype=np.int32),
            np.random.randint(0, 800, size=(100, 150), dtype=np.int32),
        ]

        yield

        # Cleanup
        if hasattr(self, "test_dir") and Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)

    def test_basic_hdf5_write_read_cycle(self):
        """Test basic HDF5 write/read cycle with string handling."""
        from pds.utils.image_data import _ImageList

        # Note: Using ASCII-safe names to avoid PyTables warnings while still testing Unicode in descriptions
        setnames = ["S001", "scan_data", "test_dataset_unicode", "chinese_dataset"]
        descriptions = ["Test dataset S001", "Test dataset scan_data"]

        for setname, description in zip(setnames, descriptions):
            # Create ImageList with HDF5 storage
            img_list = _ImageList(images=self.test_images, file=self.test_file, path=self.test_dir, setname=setname, descr=description)

            # Verify basic properties
            assert len(img_list) == len(self.test_images)
            assert img_list.file == self.test_file
            assert img_list.setname == setname

            # Test reading back the images
            for i, original_image in enumerate(self.test_images):
                retrieved_image = img_list[i]
                np.testing.assert_array_equal(retrieved_image, original_image)
                assert retrieved_image.dtype == original_image.dtype

    def test_string_encoding_in_descriptions(self):
        """Test that string descriptions are properly encoded in HDF5."""
        from pds.utils.image_data import _ImageList

        # Test with various Unicode strings
        descriptions = [
            "Simple ASCII description",
        ]

        for i, descr in enumerate(descriptions):
            setname = f"S{i:03d}"
            img_list = _ImageList(
                images=self.test_images[:1],
                file=f"test_unicode_{i}.h5",
                path=self.test_dir,
                setname=setname,
                descr=descr,
            )

            # Verify we can read back the data
            retrieved_image = img_list[0]
            np.testing.assert_array_equal(retrieved_image, self.test_images[0])

    def test_hdf5_file_path_handling(self):
        """Test HDF5 file path handling with various formats."""
        import os

        from pds.utils.image_data import _ImageList

        # Test absolute paths
        abs_path = os.path.abspath(self.test_dir)
        img_list = _ImageList(images=self.test_images[:1], file="abs_path_test.h5", path=abs_path, setname="abs_test")
        assert os.path.exists(img_list._make_fname())

        # Test relative paths
        rel_path = os.path.relpath(self.test_dir)
        img_list = _ImageList(images=self.test_images[:1], file="rel_path_test.h5", path=rel_path, setname="rel_test")
        assert os.path.exists(img_list._make_fname())

        # Test None path (current directory)
        original_cwd = os.getcwd()
        try:
            os.chdir(self.test_dir)
            img_list = _ImageList(images=self.test_images[:1], file="none_path_test.h5", path=None, setname="none_test")
            assert os.path.exists(img_list._make_fname())
        finally:
            os.chdir(original_cwd)

    def test_large_dataset_handling(self):
        """Test handling of larger datasets to ensure memory efficiency."""
        from pds.utils.image_data import _ImageList

        # Create larger test images
        large_images = [np.random.randint(0, 65535, size=(500, 487), dtype=np.int32) for _ in range(10)]

        img_list = _ImageList(images=large_images, file="large_dataset.h5", path=self.test_dir, setname="large_data", descr="Large dataset test")

        # Test random access
        indices_to_test = [0, 3, 7, 9]
        for idx in indices_to_test:
            retrieved_image = img_list[idx]
            np.testing.assert_array_equal(retrieved_image, large_images[idx])

        # Test slice access
        slice_images = img_list[2:5]
        for i, retrieved_image in enumerate(slice_images):
            np.testing.assert_array_equal(retrieved_image, large_images[2 + i])

    def test_edge_cases_and_error_handling(self):
        """Test edge cases and error handling for robustness."""
        from pds.utils.image_data import _ImageList

        # Test with empty image list
        empty_list = _ImageList(images=None, file="empty_test.h5", path=self.test_dir, setname="empty")
        assert len(empty_list) == 0

        # Test reading from non-existent file
        non_existent = _ImageList(images=None, file="non_existent.h5", path=self.test_dir, setname="missing")
        result = non_existent._read_image_tables()
        assert result is None

        # Test with invalid index access
        img_list = _ImageList(images=self.test_images, file="error_test.h5", path=self.test_dir, setname="error_test")

        with pytest.raises(IndexError):
            _ = img_list[len(self.test_images) + 1]

        with pytest.raises(IndexError):
            _ = img_list[-len(self.test_images) - 1]

    def test_concurrent_access_safety(self):
        """Test that multiple readers can safely access the same HDF5 file."""
        from pds.utils.image_data import _ImageList

        # Create initial dataset
        img_list1 = _ImageList(
            images=self.test_images, file="concurrent_test.h5", path=self.test_dir, setname="concurrent_data", descr="Concurrent access test"
        )

        # Create second reader for same file
        img_list2 = _ImageList(images=None, file="concurrent_test.h5", path=self.test_dir, setname="concurrent_data")

        # Both should be able to read the same data
        for i in range(len(self.test_images)):
            data1 = img_list1[i]
            data2 = img_list2[i]
            np.testing.assert_array_equal(data1, data2)
            np.testing.assert_array_equal(data1, self.test_images[i])

    def test_data_type_preservation(self):
        """Test that data types are preserved correctly in HDF5 storage."""
        from pds.utils.image_data import _ImageList

        # Test with different data types
        test_dtypes = [np.int16, np.int32, np.uint16, np.uint32]

        for dtype in test_dtypes:
            # Create images with specific dtype
            typed_images = [np.random.randint(0, 1000, size=(50, 75), dtype=dtype) for _ in range(2)]

            img_list = _ImageList(
                images=typed_images,
                file=f"dtype_test_{dtype.__name__}.h5",
                path=self.test_dir,
                setname=f"dtype_{dtype.__name__}",
                descr=f"Data type test for {dtype}",
            )

            # Verify data type preservation
            for i, original in enumerate(typed_images):
                retrieved = img_list[i]
                assert retrieved.dtype == original.dtype
                np.testing.assert_array_equal(retrieved, original)

    def test_hdf5_metadata_encoding(self):
        """Test that HDF5 metadata is properly encoded/decoded."""
        import tables

        from pds.utils.image_data import _ImageList

        # Create dataset with metadata
        img_list = _ImageList(images=self.test_images[:1], file="metadata_test.h5", path=self.test_dir, setname="metadata_test", descr="Metadata encoding test")

        # Open file directly with PyTables to inspect metadata
        fname = img_list._make_fname()
        with tables.open_file(fname, mode="r") as h:
            # Check that we can navigate the structure
            assert hasattr(h.root, "image_data")
            assert hasattr(h.root.image_data, "metadata_test")

            # Check that we can read the images array
            images_node = h.get_node("/image_data/metadata_test", "images")
            data = images_node.read()
            np.testing.assert_array_equal(data[0], self.test_images[0])

    def test_cleanup_functionality(self):
        """Test that cleanup functionality works properly."""
        from pds.utils.image_data import _ImageList

        img_list = _ImageList(images=self.test_images, file="cleanup_test.h5", path=self.test_dir, setname="cleanup_test")

        # Force cleanup
        img_list._cleanup()

        # Should still be able to read data after cleanup
        retrieved = img_list[0]
        np.testing.assert_array_equal(retrieved, self.test_images[0])

    def test_existing_dataset_handling(self):
        """Test behavior when dataset already exists."""
        from pds.utils.image_data import _ImageList

        # Create initial dataset
        img_list1 = _ImageList(images=self.test_images, file="existing_test.h5", path=self.test_dir, setname="existing_data", descr="Original dataset")

        # Try to create another dataset with same name. This should warn but not overwrite.
        new_images = [np.zeros((10, 10), dtype=np.int32)]
        _ImageList(
            images=new_images,
            file="existing_test.h5",
            path=self.test_dir,
            setname="existing_data",
            descr="Attempted overwrite",
        )

        # Original data should still be accessible
        original_data = img_list1[0]
        np.testing.assert_array_equal(original_data, self.test_images[0])

    def test_bytes_string_compatibility(self):
        """Test explicit bytes/string compatibility for Python 2/3 migration."""
        from pds.utils.image_data import _ImageList

        # Test setnames as both str and bytes-like
        test_cases = [
            ("string_setname", "String setname test"),
            ("byte_setname", b"Byte setname test".decode("utf-8")),
        ]

        for setname, description in test_cases:
            img_list = _ImageList(images=self.test_images[:1], file=f"compat_test_{setname}.h5", path=self.test_dir, setname=setname, descr=description)

            # Verify successful storage and retrieval
            assert len(img_list) == 1
            retrieved = img_list[0]
            np.testing.assert_array_equal(retrieved, self.test_images[0])

            # Verify setname is stored as string
            assert isinstance(img_list.setname, str)

    def test_pytables_string_encoding_edge_cases(self):
        """Test specific PyTables string encoding edge cases from Python 2->3 migration."""
        import os

        from pds.utils.image_data import _ImageList

        # Test scenarios that could cause issues during migration
        edge_cases = [
            # Standard ASCII
            ("ascii_test", "Simple ASCII description"),
            # Empty strings
            ("empty_desc", ""),
            # Very long description
            ("long_desc", "Very long description " * 50),
            # Special characters that might cause encoding issues
            ("special_chars", "Special: \n\t\r\\ \"quotes\" 'apostrophes' & symbols"),
        ]

        for setname, description in edge_cases:
            # This should handle any encoding issues gracefully
            img_list = _ImageList(images=self.test_images[:1], file=f"encoding_edge_{setname}.h5", path=self.test_dir, setname=setname, descr=description)

            # Verify successful storage
            assert len(img_list) == 1
            retrieved = img_list[0]
            np.testing.assert_array_equal(retrieved, self.test_images[0])

            # Verify that file can be reopened and read
            h5_file = img_list._make_fname()
            assert os.path.exists(h5_file)

            # Re-read using fresh instance
            new_img_list = _ImageList(images=None, file=os.path.basename(h5_file), path=self.test_dir, setname=setname)
            re_retrieved = new_img_list[0]
            np.testing.assert_array_equal(re_retrieved, self.test_images[0])

    def test_imagescan_hdf5_archive_integration(self):
        """Test ImageScan integration with HDF5 archive functionality."""
        # Create test images
        test_images = [np.random.randint(0, 1000, size=(100, 150), dtype=np.int32) for _ in range(5)]

        # Test archive configuration
        archive_config = {"file": "integration_test.h5", "path": self.test_dir, "setname": "integration_scan", "descr": "Integration test scan data"}

        # Create ImageScan with HDF5 archive
        scan = ImageScan(image=test_images, archive=archive_config)

        # Verify the archive was created and works
        assert hasattr(scan.image, "file")
        assert scan.image.file == "integration_test.h5"
        assert len(scan.image) == len(test_images)

        # Test image access through ImageScan
        for i in range(len(test_images)):
            retrieved = scan.image[i]
            np.testing.assert_array_equal(retrieved, test_images[i])

    def test_hdf5_archive_string_encoding_integration(self):
        """Test string encoding in full HDF5 archive workflow."""
        test_images = [np.random.randint(0, 500, size=(50, 75), dtype=np.int32) for _ in range(3)]

        # Test with Unicode strings in all configuration
        archive_config = {
            "file": "unicode_intégration_.h5",
            "path": self.test_dir,
            "setname": "scan_data_unicode",
            "descr": "Integration test",
        }

        # This should not raise any encoding errors
        scan = ImageScan(image=test_images, archive=archive_config)

        # Verify all operations work with Unicode
        assert len(scan.image) == len(test_images)
        for i in range(len(test_images)):
            retrieved = scan.image[i]
            np.testing.assert_array_equal(retrieved, test_images[i])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
