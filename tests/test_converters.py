import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from pds.utils.converters import bytes_to_str, safe_eval_hdf


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


if __name__ == "__main__":
    pytest.main([__file__])
