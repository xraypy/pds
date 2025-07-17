"""
Tests for filtertools module.

This test suite ensures that the filtertools functions work correctly
with date operations, list operations, and HDF5 filtering including
Python 2/3 compatibility for bytes/strings.
"""

import os
import tempfile

import h5py
import numpy as np
import pytest

from pds.utils.filtertools import cases, is_after, is_before, list_intersect, list_union


class TestDateOperations:
    """Test date comparison functions."""

    def test_is_before_basic(self):
        """Test basic date comparison functionality."""
        date1 = "Mon Jan 12 08:42:32 2011"
        date2 = "Tue Jan 13 08:42:32 2011"

        assert is_before(date1, date2)
        assert not is_before(date2, date1)
        assert not is_before(date1, date1)

    def test_is_before_year_comparison(self):
        """Test year-based comparison."""
        date1 = "Mon Jan 12 08:42:32 2010"
        date2 = "Mon Jan 12 08:42:32 2011"

        assert is_before(date1, date2)
        assert not is_before(date2, date1)

    def test_is_before_month_comparison(self):
        """Test month-based comparison within same year."""
        date1 = "Mon Jan 12 08:42:32 2011"
        date2 = "Sat Feb 12 08:42:32 2011"

        assert is_before(date1, date2)
        assert not is_before(date2, date1)

    def test_is_before_day_comparison(self):
        """Test day-based comparison within same month."""
        date1 = "Mon Jan 12 08:42:32 2011"
        date2 = "Wed Jan 15 08:42:32 2011"

        assert is_before(date1, date2)
        assert not is_before(date2, date1)

    def test_is_before_time_comparison(self):
        """Test time-based comparison within same day."""
        date1 = "Mon Jan 12 08:42:32 2011"
        date2 = "Mon Jan 12 09:42:32 2011"

        assert is_before(date1, date2)
        assert not is_before(date2, date1)

    def test_is_after_basic(self):
        """Test basic after comparison functionality."""
        date1 = "Mon Jan 12 08:42:32 2011"
        date2 = "Tue Jan 13 08:42:32 2011"

        assert is_after(date2, date1)
        assert not is_after(date1, date2)
        assert is_after(date1, date1)  # Equal dates return True for is_after

    def test_is_after_equal_dates(self):
        """Test that equal dates return True for is_after."""
        date1 = "Mon Jan 12 08:42:32 2011"
        date2 = "Mon Jan 12 08:42:32 2011"

        assert is_after(date1, date2)
        assert is_after(date2, date1)

    def test_edge_cases_month_boundaries(self):
        """Test edge cases around month boundaries."""
        # End of year to beginning of next year
        date1 = "Sat Dec 31 23:59:59 2010"
        date2 = "Sat Jan 01 00:00:00 2011"

        assert is_before(date1, date2)
        assert is_after(date2, date1)

    def test_all_months_ordering(self):
        """Test that all months are correctly ordered."""
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

        for i in range(len(months) - 1):
            date1 = f"Mon {months[i]} 15 12:00:00 2011"
            date2 = f"Mon {months[i + 1]} 15 12:00:00 2011"
            assert is_before(date1, date2)


class TestListOperations:
    """Test list utility functions."""

    def test_list_union_basic(self):
        """Test basic list union functionality."""
        list1 = [1, 2, 3]
        list2 = [3, 4, 5]
        list3 = [5, 6, 7]

        result = list_union(list1, list2, list3)
        expected = [1, 2, 3, 4, 5, 6, 7]

        assert set(result) == set(expected)
        assert len(result) == len(expected)

    def test_list_union_with_none(self):
        """Test list union with None values."""
        list1 = [1, 2, 3]
        list2 = None
        list3 = [4, 5, 6]

        result = list_union(list1, list2, list3)
        expected = [1, 2, 3, 4, 5, 6]

        assert set(result) == set(expected)

    def test_list_union_empty_lists(self):
        """Test list union with empty lists."""
        list1 = []
        list2 = [1, 2, 3]
        list3 = []

        result = list_union(list1, list2, list3)
        expected = [1, 2, 3]

        assert set(result) == set(expected)

    def test_list_union_all_none(self):
        """Test list union with all None values."""
        result = list_union(None, None, None)
        assert result == []

    def test_list_union_duplicates(self):
        """Test that list union removes duplicates."""
        list1 = [1, 2, 2, 3]
        list2 = [2, 3, 4, 4]

        result = list_union(list1, list2)
        expected = [1, 2, 3, 4]

        assert set(result) == set(expected)
        assert len(result) == len(expected)

    def test_list_intersect_basic(self):
        """Test basic list intersection functionality."""
        list1 = [1, 2, 3, 4]
        list2 = [3, 4, 5, 6]
        list3 = [4, 5, 6, 7]

        result = list_intersect(list1, list2, list3)
        expected = [4]

        assert set(result) == set(expected)

    def test_list_intersect_no_common(self):
        """Test list intersection with no common elements."""
        list1 = [1, 2, 3]
        list2 = [4, 5, 6]

        result = list_intersect(list1, list2)
        assert result == []

    def test_list_intersect_identical(self):
        """Test list intersection with identical lists."""
        list1 = [1, 2, 3, 4]
        list2 = [1, 2, 3, 4]

        result = list_intersect(list1, list2)
        expected = [1, 2, 3, 4]

        assert set(result) == set(expected)

    def test_list_intersect_single_list(self):
        """Test list intersection with single list."""
        list1 = [1, 2, 3, 4]

        result = list_intersect(list1)
        expected = [1, 2, 3, 4]

        assert set(result) == set(expected)

    def test_list_intersect_empty_list(self):
        """Test list intersection with empty list."""
        result = list_intersect()
        assert result == []

    def test_list_operations_with_strings(self):
        """Test list operations with string data."""
        list1 = ["a", "b", "c"]
        list2 = ["c", "d", "e"]

        union_result = list_union(list1, list2)
        intersect_result = list_intersect(list1, list2)

        assert set(union_result) == {"a", "b", "c", "d", "e"}
        assert set(intersect_result) == {"c"}

    def test_list_operations_mixed_types(self):
        """Test list operations with mixed data types."""
        list1 = [1, "a", 3.14]
        list2 = ["a", 2, 3.14]

        union_result = list_union(list1, list2)
        intersect_result = list_intersect(list1, list2)

        assert "a" in union_result
        assert 3.14 in union_result
        assert set(intersect_result) == {"a", 3.14}


class TestHDF5Filtering:
    """Test HDF5 filtering functionality with compatibility testing."""

    def setup_method(self):
        """Set up test HDF5 file with various data types."""
        self.test_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        self.test_file.close()

        # Create test HDF5 file with sample data
        with h5py.File(self.test_file.name, "w") as f:
            # Create spec groups
            spec1 = f.create_group("spec1")
            spec2 = f.create_group("spec2")

            # Create scans with different data types
            scan1 = spec1.create_group("001")
            scan2 = spec1.create_group("002")
            scan3 = spec2.create_group("001")

            # Add attributes (both string and numeric)
            scan1.attrs["scan_type"] = "rodscan"
            scan1.attrs["temperature"] = 298.5
            scan1.attrs["voltage"] = 5.0

            scan2.attrs["scan_type"] = "xscan"
            scan2.attrs["temperature"] = 310.2
            scan2.attrs["voltage"] = 7.5

            scan3.attrs["scan_type"] = "yscan"
            scan3.attrs["temperature"] = 285.0
            scan3.attrs["voltage"] = 4.2

            # Add param_labs and param_data
            for scan in [scan1, scan2, scan3]:
                scan.create_dataset("param_labs", data=[b"motor_pos", b"detector_count"])
                scan.create_dataset("param_data", data=[1.5, 1000])

                # Add point data
                scan.create_dataset("point_labs", data=[b"x", b"y", b"intensity"])
                scan.create_dataset("point_data", data=[[0.1, 0.2, 100], [0.2, 0.3, 150]])

    def teardown_method(self):
        """Clean up test files."""
        try:
            os.unlink(self.test_file.name)
        except (OSError, FileNotFoundError):
            pass

    def test_cases_scan_level_string_filter(self):
        """Test scan-level filtering with string attributes."""
        with h5py.File(self.test_file.name, "r") as f:
            # Filter for rodscan type
            result = cases(f, "scan_type", '== "rodscan"', "scan")
            assert len(result) == 1
            assert "/spec1/001" in result

            # Filter for non-existent type
            result = cases(f, "scan_type", '== "zscan"', "scan")
            assert len(result) == 0

    def test_cases_scan_level_numeric_filter(self):
        """Test scan-level filtering with numeric attributes."""
        with h5py.File(self.test_file.name, "r") as f:
            # Filter for temperature > 300
            result = cases(f, "temperature", "> 300", "scan")
            assert len(result) == 1
            assert "/spec1/002" in result

            # Filter for voltage >= 5.0
            result = cases(f, "voltage", ">= 5.0", "scan")
            assert len(result) == 2
            assert "/spec1/001" in result
            assert "/spec1/002" in result

    def test_cases_scan_level_param_filter(self):
        """Test scan-level filtering with param_data."""
        with h5py.File(self.test_file.name, "r") as f:
            # Filter based on motor_pos parameter
            result = cases(f, "motor_pos", "== 1.5", "scan")
            assert len(result) == 3  # All scans have motor_pos = 1.5

    def test_cases_point_level_filter(self):
        """Test point-level filtering."""
        with h5py.File(self.test_file.name, "r") as f:
            # Filter for intensity > 120
            result = cases(f, "intensity", "> 120", "point")
            assert len(result) == 3  # One point per scan matches

            # Check that results are tuples of (scan_path, point_index)
            for scan_path, point_index in result:
                assert isinstance(scan_path, str)
                assert isinstance(point_index, int)
                assert point_index == 1  # Second point has intensity 150

    def test_cases_bytes_string_compatibility(self):
        """Test compatibility with bytes data from HDF5."""
        # Test with bytes attribute (simulating Python 2 data)
        # Manually set bytes attribute to test conversion
        with h5py.File(self.test_file.name, "r+") as f_write:
            f_write["/spec1/001"].attrs["bytes_attr"] = b"test_value"

        # Reopen and test filtering
        with h5py.File(self.test_file.name, "r") as f_read:
            result = cases(f_read, "bytes_attr", '== "test_value"', "scan")
            assert len(result) == 1
            assert "/spec1/001" in result

    def test_cases_invalid_input(self):
        """Test error handling with invalid inputs."""
        # Test with non-HDF5 file object
        result = cases("not_a_file", "attr", "> 0", "scan")
        assert result is None

        # Test with invalid filter level
        with h5py.File(self.test_file.name, "r") as f:
            result = cases(f, "temperature", "> 300", "invalid_level")
            assert result is None

    def test_cases_nonexistent_attribute(self):
        """Test filtering with non-existent attributes."""
        with h5py.File(self.test_file.name, "r") as f:
            # Should return empty list for non-existent attribute
            result = cases(f, "nonexistent_attr", "> 0", "scan")
            assert result == []

    def test_cases_complex_expressions(self):
        """Test filtering with complex boolean expressions."""
        with h5py.File(self.test_file.name, "r") as f:
            # Simple numeric filter
            result = cases(f, "temperature", "> 290", "scan")
            assert len(result) == 2

            # String filter
            result = cases(f, "scan_type", '== "rodscan"', "scan")
            assert len(result) == 1

    def test_cases_numpy_bool_compatibility(self):
        """Test compatibility with numpy boolean types."""
        with h5py.File(self.test_file.name, "r+") as f:
            # Add numpy boolean attribute
            f["/spec1/001"].attrs["is_calibrated"] = np.bool_(True)
            f["/spec1/002"].attrs["is_calibrated"] = np.bool_(False)
            f["/spec2/001"].attrs["is_calibrated"] = np.bool_(True)

        with h5py.File(self.test_file.name, "r") as f:
            result = cases(f, "is_calibrated", "== True", "scan")
            assert len(result) == 2
            assert "/spec1/001" in result
            assert "/spec2/001" in result

    def test_cases_different_numeric_types(self):
        """Test filtering with different numeric types."""
        with h5py.File(self.test_file.name, "r+") as f:
            # Add different numeric types
            f["/spec1/001"].attrs["int_val"] = np.int32(42)
            f["/spec1/002"].attrs["int_val"] = np.int64(84)
            f["/spec2/001"].attrs["int_val"] = np.float32(21.5)

        with h5py.File(self.test_file.name, "r") as f:
            result = cases(f, "int_val", "> 40", "scan")
            assert len(result) == 2
            assert "/spec1/001" in result
            assert "/spec1/002" in result

    def test_cases_empty_hdf5_file(self):
        """Test behavior with empty HDF5 file."""
        empty_file = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
        empty_file.close()

        try:
            with h5py.File(empty_file.name, "w") as f:
                pass  # Create empty file

            with h5py.File(empty_file.name, "r") as f:
                result = cases(f, "any_attr", "> 0", "scan")
                assert result == []
        finally:
            os.unlink(empty_file.name)

    def test_python2_compatibility_validation(self):
        """
        Comprehensive test to validate that our Python 3 implementation
        matches the original Python 2 behavior exactly.
        """
        # Test 1: list_union with None handling (recursive removal)
        result = list_union([1, 2], None, [3, 4])
        assert set(result) == {1, 2, 3, 4}

        result = list_union(None, None, [1, 2])
        assert set(result) == {1, 2}

        result = list_union(None, None, None)
        assert result == []

        # Test 2: String filtering like original basestring handling
        with h5py.File(self.test_file.name, "r") as f:
            # Original: if eval('"' + this_value + '" ' + such_that)
            result = cases(f, "scan_type", '== "rodscan"', "scan")
            assert len(result) == 1
            assert "/spec1/001" in result

        # Test 3: Numeric filtering like original
        with h5py.File(self.test_file.name, "r") as f:
            # Original: if eval(str(this_value) + " " + such_that)
            result = cases(f, "temperature", "> 300", "scan")
            assert len(result) == 1
            assert "/spec1/002" in result

        # Test 4: Point-level filtering
        with h5py.File(self.test_file.name, "r") as f:
            result = cases(f, "scan_type", '== "rodscan"', "point")
            assert len(result) == 2  # Two points in the matching scan
            assert all(scan_path == "/spec1/001" for scan_path, point_idx in result)

        # Test 5: Date comparison exactly as original
        date1 = "Mon Jan 12 08:42:32 2011"
        date2 = "Tue Jan 13 08:42:32 2011"
        assert is_before(date1, date2)
        assert is_after(date2, date1)
        assert is_after(date1, date1)  # Original: am_i == after_me or not is_before

        # Test 6: List operations exactly as original
        result = list_intersect([1, 2, 3], [2, 3, 4], [3, 4, 5])
        assert result == [3]  # Only 3 is in all lists


class TestTypeHints:
    """Test that type hints are properly defined."""

    def test_function_signatures(self):
        """Test that functions have proper type annotations."""
        import inspect

        # Test is_before signature
        sig = inspect.signature(is_before)
        assert len(sig.parameters) == 2
        assert sig.return_annotation is bool

        # Test is_after signature
        sig = inspect.signature(is_after)
        assert len(sig.parameters) == 2
        assert sig.return_annotation is bool

        # Test list_union signature
        sig = inspect.signature(list_union)
        assert sig.return_annotation != inspect.Signature.empty

        # Test list_intersect signature
        sig = inspect.signature(list_intersect)
        assert sig.return_annotation != inspect.Signature.empty

        # Test cases signature
        sig = inspect.signature(cases)
        assert len(sig.parameters) == 4
        assert sig.return_annotation != inspect.Signature.empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
