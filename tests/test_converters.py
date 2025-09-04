import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from pds.utils.converters import bytes_to_str, extract_value, safe_eval_hdf


class TestBytesToStr:
    """Test bytes to string conversion function."""

    def test_bytes_input(self):
        """Test conversion of bytes to string."""
        assert bytes_to_str(b"hello") == "hello"
        assert bytes_to_str(b"utf-8 test") == "utf-8 test"
        assert bytes_to_str(b"") == ""

    def test_string_input(self):
        """Test that strings are returned unchanged."""
        assert bytes_to_str("world") == "world"
        assert bytes_to_str("") == ""
        assert bytes_to_str("unicode 测试") == "unicode 测试"

    def test_other_types(self):
        """Test that non-bytes/string types are returned unchanged."""
        assert bytes_to_str(123) == 123
        assert bytes_to_str(45.67) == 45.67
        assert bytes_to_str(None) is None
        assert bytes_to_str([1, 2, 3]) == [1, 2, 3]
        assert bytes_to_str({"key": "value"}) == {"key": "value"}


class TestSafeEvalHdf:
    """Test safe HDF value evaluation function."""

    def test_boolean_strings(self):
        """Test conversion of boolean string representations."""
        assert safe_eval_hdf("true") is True
        assert safe_eval_hdf("TRUE") is True
        assert safe_eval_hdf("True") is True
        assert safe_eval_hdf("false") is False
        assert safe_eval_hdf("FALSE") is False
        assert safe_eval_hdf("False") is False

    def test_numeric_values(self):
        """Test that numeric values are returned unchanged."""
        assert safe_eval_hdf(123) == 123
        assert safe_eval_hdf(45.67) == 45.67
        assert safe_eval_hdf(-89) == -89
        assert safe_eval_hdf(0.0) == 0.0

    def test_numpy_arrays(self):
        """Test that numpy arrays are returned unchanged."""
        arr = np.array([1, 2, 3])
        result = safe_eval_hdf(arr)
        np.testing.assert_array_equal(result, arr)
        assert result is arr  # Should be the same object

    def test_string_evaluation(self):
        """Test evaluation of string representations."""
        assert safe_eval_hdf('"test"') == "test"
        assert safe_eval_hdf("'hello'") == "hello"
        assert safe_eval_hdf("[1, 2, 3]") == [1, 2, 3]
        assert safe_eval_hdf('{"key": "value"}') == {"key": "value"}

    def test_bytes_literal_strings(self):
        """Test evaluation of bytes literal string representations."""
        # Test with valid Python data structures in bytes literals
        assert safe_eval_hdf("b'[1, 2, 3]'") == [1, 2, 3]
        # Test with strings that can't be evaluated (return original)
        assert safe_eval_hdf("b'hello'") == "b'hello'"
        assert safe_eval_hdf('b"world"') == 'b"world"'

    def test_bytes_input(self):
        """Test handling of bytes input."""
        assert safe_eval_hdf(b'"test"') == "test"  # Bytes containing quoted string
        assert safe_eval_hdf(b"[1, 2, 3]") == [1, 2, 3]  # Bytes containing list
        assert safe_eval_hdf(b"123") == 123  # Bytes containing number

    def test_empty_values(self):
        """Test handling of empty values."""
        assert safe_eval_hdf("[]") == []
        assert safe_eval_hdf("{}") == {}
        assert safe_eval_hdf('""') == ""

    def test_error_handling(self):
        """Test that invalid inputs return original value."""
        invalid_input = object()
        assert safe_eval_hdf(invalid_input) is invalid_input

        # Test malformed string that can't be evaluated
        malformed = "invalid python syntax {"
        assert safe_eval_hdf(malformed) == malformed

    def test_none_input(self):
        """Test handling of None input."""
        assert safe_eval_hdf(None) is None

    def test_complex_data_structures(self):
        """Test handling of complex nested data structures."""
        complex_dict = "{'nested': {'key': [1, 2, 3]}, 'other': 'value'}"
        expected = {"nested": {"key": [1, 2, 3]}, "other": "value"}
        assert safe_eval_hdf(complex_dict) == expected

    def test_array_like_objects(self):
        """Test handling of objects with array-like properties."""

        class MockArray:
            def __init__(self):
                self.shape = (3,)

        mock_arr = MockArray()
        result = safe_eval_hdf(mock_arr)
        assert result is mock_arr


class TestExtractValue:
    """Test extract_value function for handling scalar and array data."""

    def test_scalar_float(self):
        """Test extraction from scalar float value."""
        assert extract_value(3.14, 0) == 3.14
        assert extract_value(3.14, 5) == 3.14  # Point index ignored for scalars
        assert isinstance(extract_value(3.14, 0), float)

    def test_scalar_int(self):
        """Test extraction from scalar integer value."""
        assert extract_value(42, 0) == 42.0
        assert extract_value(42, 10) == 42.0  # Point index ignored for scalars
        assert isinstance(extract_value(42, 0), float)

    def test_list_valid_index(self):
        """Test extraction from list with valid index."""
        data = [1.0, 2.5, 3.7, 4.2]
        assert extract_value(data, 0) == 1.0
        assert extract_value(data, 1) == 2.5
        assert extract_value(data, 2) == 3.7
        assert extract_value(data, 3) == 4.2
        assert isinstance(extract_value(data, 1), float)

    def test_list_invalid_index(self):
        """Test extraction from list with invalid index falls back to first element."""
        data = [1.0, 2.5, 3.7]
        # Index out of range should return the first element
        result = extract_value(data, 10)
        assert result == 1.0
        assert isinstance(result, float)

    def test_numpy_array_valid_index(self):
        """Test extraction from numpy array with valid index."""
        data = np.array([5.5, 6.6, 7.7, 8.8])
        assert extract_value(data, 0) == 5.5
        assert extract_value(data, 1) == 6.6
        assert extract_value(data, 2) == 7.7
        assert extract_value(data, 3) == 8.8
        assert isinstance(extract_value(data, 1), float)

    def test_numpy_array_invalid_index(self):
        """Test extraction from numpy array with invalid index falls back to first element."""
        data = np.array([5.5, 6.6, 7.7])
        # Index out of range should return the first element
        result = extract_value(data, 10)
        assert result == 5.5
        assert isinstance(result, float)

    def test_numpy_scalar(self):
        """Test extraction from numpy scalar values."""
        data = np.float64(9.99)
        assert extract_value(data, 0) == 9.99
        assert extract_value(data, 5) == 9.99  # Point index ignored for scalars
        assert isinstance(extract_value(data, 0), float)

    def test_string_input(self):
        """Test that strings are treated as non-indexable and converted to float."""
        # This should raise ValueError as strings can't be converted to float
        with pytest.raises(ValueError):
            extract_value("hello", 0)

    def test_numeric_string(self):
        """Test extraction from numeric strings."""
        assert extract_value("3.14", 0) == 3.14
        assert extract_value("42", 5) == 42.0
        assert isinstance(extract_value("3.14", 0), float)

    def test_mixed_types_in_list(self):
        """Test extraction from list with mixed numeric types."""
        data = [1, 2.5, np.float64(3.7), 4]
        assert extract_value(data, 0) == 1.0
        assert extract_value(data, 1) == 2.5
        assert extract_value(data, 2) == 3.7
        assert extract_value(data, 3) == 4.0
        assert all(isinstance(extract_value(data, i), float) for i in range(4))

    def test_empty_list(self):
        """Test extraction from empty list with fallback to scalar conversion."""
        data = []
        # This should raise ValueError as empty list can't be converted to float
        with pytest.raises(ValueError):
            extract_value(data, 0)

    def test_single_element_list(self):
        """Test extraction from single-element list."""
        data = [7.5]
        assert extract_value(data, 0) == 7.5
        # Index out of range should fall back to first element
        result = extract_value(data, 1)
        assert result == 7.5
        assert isinstance(result, float)

    def test_negative_values(self):
        """Test extraction of negative values."""
        data = [-1.5, -2.7, -3.9]
        assert extract_value(data, 0) == -1.5
        assert extract_value(data, 1) == -2.7
        assert extract_value(data, 2) == -3.9

        # Test scalar negative value
        assert extract_value(-5.5, 0) == -5.5

    def test_zero_values(self):
        """Test extraction of zero values."""
        data = [0.0, 0, -0.0]
        assert extract_value(data, 0) == 0.0
        assert extract_value(data, 1) == 0.0
        assert extract_value(data, 2) == 0.0

        # Test scalar zero
        assert extract_value(0, 0) == 0.0
        assert extract_value(0.0, 0) == 0.0


if __name__ == "__main__":
    pytest.main([__file__])
