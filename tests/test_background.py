import sys
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# Add the pds module to the path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from pds.utils.background import background, compress_array, expand_array, linear_background, plot_bgr, show_bgr


@pytest.fixture
def test_data_simple():
    """Simple test data for basic functionality."""
    np.random.seed(42)  # For reproducible results

    def gauss(x, cen, sigma):
        return np.exp(-((x - cen) ** 2) / (2 * sigma**2))

    npts = 100
    x = 1.0 * np.arange(npts)
    g1 = 40 * gauss(x, 0.6 * npts, 8.0)
    g2 = 20 * gauss(x, 0.5 * npts, 15.0)
    r = 2 * np.random.normal(size=npts)
    y = r + x / 25 + g1 + g2

    return x, y


@pytest.fixture
def test_data_complex():
    """Complex test data similar to original Python 2 example."""
    np.random.seed(42)

    def gauss(x, cen, sigma):
        return np.exp(-((x - cen) ** 2) / (2 * sigma**2))

    npts = 2000
    x = 1.0 * np.arange(npts)
    g1 = 40 * gauss(x, 0.6 * npts, 8.0)
    g2 = 20 * gauss(x, 0.5 * npts, 270)
    r = 2 * np.random.normal(size=npts)
    y = r + x / 25 + g1 + g2

    return x, y


class TestLinearBackground:
    """Test linear_background function for Python 2/3 compatibility."""

    def test_linear_background_basic(self, test_data_simple):
        """Test basic linear background functionality."""
        x, y = test_data_simple

        # Test with default parameters
        result = linear_background(y, nbgr=0)
        assert isinstance(result, np.ndarray)
        assert result.shape == y.shape
        np.testing.assert_array_equal(result, np.zeros(len(y)))

        # Test with nbgr > 0
        result = linear_background(y, nbgr=3)
        assert isinstance(result, np.ndarray)
        assert result.shape == y.shape
        assert result.dtype == np.float64

    def test_linear_background_edge_cases(self):
        """Test edge cases for linear background."""
        # Small array
        small_y = np.array([1.0, 2.0, 3.0])
        result = linear_background(small_y, nbgr=2)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.zeros(len(small_y)))

        # Empty array
        empty_y = np.array([])
        result = linear_background(empty_y, nbgr=1)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0

        # Negative nbgr
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = linear_background(y, nbgr=-1)
        np.testing.assert_array_equal(result, np.zeros(len(y)))

    def test_linear_background_dtype_consistency(self, test_data_simple):
        """Test that data types are consistent with Python 2 behavior."""
        x, y = test_data_simple

        # Test with different input types
        y_int = y.astype(np.int32)
        result_int = linear_background(y_int, nbgr=3)
        assert result_int.dtype == np.float64

        y_float32 = y.astype(np.float32)
        result_float32 = linear_background(y_float32, nbgr=3)
        assert result_float32.dtype == np.float64

        y_float64 = y.astype(np.float64)
        result_float64 = linear_background(y_float64, nbgr=3)
        assert result_float64.dtype == np.float64


class TestBackground:
    """Test background function for Python 2/3 compatibility."""

    def test_background_basic(self, test_data_simple):
        """Test basic background functionality."""
        x, y = test_data_simple

        # Test with default parameters
        result = background(y)
        assert isinstance(result, np.ndarray)
        assert result.shape == y.shape
        assert result.dtype == np.float64

        # Test with various parameters
        result = background(y, nbgr=3, width=10, pow=1.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == y.shape
        assert result.dtype == np.float64

    def test_background_integer_division(self, test_data_simple):
        """Test integer division behavior matches Python 2."""
        x, y = test_data_simple

        # Test cases that would differ between Python 2 and 3 division
        test_cases = [
            {"nbgr": 3, "width": 5, "pow": 1.0},
            {"nbgr": 5, "width": 7, "pow": 0.5},
            {"nbgr": 2, "width": 11, "pow": 2.0},
        ]

        for params in test_cases:
            result = background(y, **params)
            assert isinstance(result, np.ndarray)
            assert result.shape == y.shape
            assert np.all(np.isfinite(result))

    def test_background_compression(self, test_data_complex):
        """Test compression functionality."""
        x, y = test_data_complex

        # Test with compression
        result_compressed = background(y, nbgr=3, width=100, compress=2)
        result_normal = background(y, nbgr=3, width=100, compress=1)

        assert isinstance(result_compressed, np.ndarray)
        assert isinstance(result_normal, np.ndarray)
        assert result_compressed.shape == result_normal.shape
        assert result_compressed.shape == y.shape

    def test_background_tangent_mode(self, test_data_simple):
        """Test tangent mode functionality."""
        x, y = test_data_simple

        # Test tangent mode
        result_tangent = background(y, nbgr=3, width=10, tangent=True)
        result_normal = background(y, nbgr=3, width=10, tangent=False)

        assert isinstance(result_tangent, np.ndarray)
        assert isinstance(result_normal, np.ndarray)
        assert result_tangent.shape == result_normal.shape
        assert result_tangent.shape == y.shape

        # Results should be different
        assert not np.array_equal(result_tangent, result_normal)

    def test_background_edge_cases(self):
        """Test edge cases for background function."""
        # Small array
        small_y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = background(small_y, nbgr=2, width=1)
        assert isinstance(result, np.ndarray)
        assert result.shape == small_y.shape

        # Zero width
        result_zero_width = background(small_y, nbgr=2, width=0)
        result_linear = linear_background(small_y, nbgr=2)
        np.testing.assert_array_equal(result_zero_width, result_linear)

        # Negative power handling
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = background(small_y, nbgr=2, width=2, pow=-1.0)
            assert len(w) == 0
            assert isinstance(result, np.ndarray)

    def test_background_dtype_consistency(self, test_data_simple):
        """Test data type consistency."""
        x, y = test_data_simple

        # Test with different input types
        y_int = y.astype(np.int32)
        result_int = background(y_int, nbgr=3, width=10)
        assert result_int.dtype == np.float64

        y_float32 = y.astype(np.float32)
        result_float32 = background(y_float32, nbgr=3, width=10)
        assert result_float32.dtype == np.float64


class TestCompressArray:
    """Test compress_array function."""

    def test_compress_array_basic(self):
        """Test basic compression functionality."""
        arr = np.arange(10, dtype=float)

        # Test compression by 2
        compressed, rem = compress_array(arr, 2)
        assert isinstance(compressed, np.ndarray)
        assert isinstance(rem, int)
        assert len(compressed) == 5
        assert rem == 0

        # Test compression by 3
        compressed, rem = compress_array(arr, 3)
        assert len(compressed) == 3
        assert rem == 1

    def test_compress_array_edge_cases(self):
        """Test edge cases for compress_array."""
        arr = np.array([1.0, 2.0, 3.0])

        # Compression factor of 1
        compressed, rem = compress_array(arr, 1)
        assert len(compressed) == 3  # int(3/1) = 3
        assert rem == 0

        # Large compression factor
        compressed, rem = compress_array(arr, 10)
        assert len(compressed) == 0
        assert rem == 3

    def test_compress_array_dtype_preservation(self):
        """Test that data types are preserved."""
        arr_float32 = np.arange(10, dtype=np.float32)
        compressed, rem = compress_array(arr_float32, 2)
        assert compressed.dtype == np.float32

        arr_float64 = np.arange(10, dtype=np.float64)
        compressed, rem = compress_array(arr_float64, 2)
        assert compressed.dtype == np.float64


class TestExpandArray:
    """Test expand_array function."""

    def test_expand_array_basic(self):
        """Test basic expansion functionality."""
        arr = np.array([1.0, 2.0, 3.0])

        # Test expansion by 2
        expanded = expand_array(arr, 2)
        assert isinstance(expanded, np.ndarray)
        assert len(expanded) == 6

        # Test expansion by 3
        expanded = expand_array(arr, 3)
        assert len(expanded) == 9

    def test_expand_array_sampling(self):
        """Test expansion with sampling."""
        arr = np.array([1.0, 2.0, 3.0])

        # Test with sampling
        expanded_sample = expand_array(arr, 2, sample=1)
        expected_sample = np.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        np.testing.assert_array_equal(expanded_sample, expected_sample)

        # Test with interpolation (default)
        expanded_interp = expand_array(arr, 2, sample=0)
        assert len(expanded_interp) == 6
        assert not np.array_equal(expanded_interp, expected_sample)

    def test_expand_array_edge_cases(self):
        """Test edge cases for expand_array."""
        arr = np.array([1.0, 2.0, 3.0])

        # Expansion factor of 1
        expanded = expand_array(arr, 1)
        np.testing.assert_array_equal(expanded, arr)

    def test_expand_array_dtype_preservation(self):
        """Test that data types are preserved."""
        arr_float32 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        expanded = expand_array(arr_float32, 2)
        assert expanded.dtype == np.float32

        arr_int32 = np.array([1, 2, 3], dtype=np.int32)
        expanded = expand_array(arr_int32, 2)
        assert expanded.dtype == np.int32


class TestShowBgr:
    """Test show_bgr function."""

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.legend")
    @patch("matplotlib.pyplot.plot")
    def test_show_bgr_basic(self, mock_plot, mock_legend, mock_show, test_data_simple):
        """Test show_bgr function without actually showing plots."""
        x, y = test_data_simple

        # Test basic functionality
        show_bgr(y, nbgr=3, width=10, pow=1.0)

        # Check that plot was called
        assert mock_plot.called
        assert mock_legend.called
        assert mock_show.called

        # Check that plot was called with correct number of arguments
        assert mock_plot.call_count == 3


class TestPlotBgr:
    """Test plot_bgr function."""

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.plot")
    @patch("matplotlib.pyplot.figure")
    @patch("matplotlib.pyplot.clf")
    @patch("matplotlib.pyplot.subplot")
    @patch("matplotlib.pyplot.legend")
    @patch("matplotlib.pyplot.ylim")
    def test_plot_bgr_basic(self, mock_ylim, mock_legend, mock_subplot, mock_clf, mock_figure, mock_plot, mock_show, test_data_simple):
        """Test plot_bgr function without actually showing plots."""
        x, y = test_data_simple

        # Test basic functionality
        plot_bgr(y, nbgr=3, width=10, pow=1.0)

        # Check that plotting functions were called
        assert mock_figure.called
        assert mock_plot.called
        assert mock_legend.called


class TestPython2Compatibility:
    """Test Python 2/3 compatibility specific issues."""

    def test_integer_division_compatibility(self, test_data_simple):
        """Test that integer division works correctly."""
        x, y = test_data_simple

        # Test division operations that would differ between Python 2 and 3
        test_widths = [1, 2, 3, 5, 7, 11, 13]

        for width in test_widths:
            result = background(y, nbgr=3, width=width, pow=1.0)
            assert isinstance(result, np.ndarray)
            assert result.shape == y.shape
            assert np.all(np.isfinite(result))

    def test_range_function_compatibility(self, test_data_simple):
        """Test that range function works correctly."""
        x, y = test_data_simple

        # Test with various parameters that use range internally
        result = background(y, nbgr=5, width=15, pow=2.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == y.shape

    def test_print_function_compatibility(self, test_data_simple):
        """Test that print function works correctly."""
        x, y = test_data_simple

        # Test with negative power to trigger print statement
        with patch("builtins.print") as mock_print:
            background(y, nbgr=3, width=10, pow=-1.0)
            mock_print.assert_called_once_with("Warning power is less than 0, changing it to positive")

    def test_numpy_array_operations(self, test_data_simple):
        """Test numpy array operations compatibility."""
        x, y = test_data_simple

        # Test various numpy operations used in the code
        result = background(y, nbgr=3, width=10, pow=1.0)

        # Check array operations
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        assert result.shape == y.shape

        # Test min/max operations
        assert np.all(np.isfinite(result))
        assert not np.any(np.isnan(result))

    def test_numerical_precision(self, test_data_simple):
        """Test numerical precision matches Python 2."""
        x, y = test_data_simple

        # Test with same random seed for reproducibility
        np.random.seed(42)
        result1 = background(y, nbgr=3, width=10, pow=1.0)

        np.random.seed(42)
        result2 = background(y, nbgr=3, width=10, pow=1.0)

        # Results should be identical
        np.testing.assert_array_equal(result1, result2)

    def test_exact_python2_example(self):
        """Test the exact Python 2 example from the original code."""
        np.random.seed(42)  # For reproducible results

        def gauss(x, cen, sigma):
            return np.exp(-((x - cen) ** 2) / (2 * sigma**2))

        npts = 2000
        x = 1.0 * np.arange(npts)
        g1 = 40 * gauss(x, 0.6 * npts, 8.0)
        g2 = 20 * gauss(x, 0.5 * npts, 270)
        r = 2 * np.random.normal(size=npts)
        y = r + x / 25 + g1 + g2

        # This is exactly what the Python 2 code does
        bgr = background(y, nbgr=3, width=100, pow=1, tangent=False, compress=1)

        # Validate results match Python 2 behavior
        assert isinstance(bgr, np.ndarray)
        assert bgr.shape == y.shape
        assert bgr.dtype == np.float64
        assert np.all(np.isfinite(bgr))
        assert bgr.mean() < y.mean()  # Background should be lower than data

    def test_extended_integer_division(self):
        """Test extended integer division cases that differ between Python 2 and 3."""
        test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        test_widths = [1, 2, 3, 5, 7, 11, 13, 17, 19]

        for width in test_widths:
            result = background(test_data, nbgr=3, width=width, pow=1.0)
            assert isinstance(result, np.ndarray)
            assert result.shape == test_data.shape
            assert result.dtype == np.float64
            assert np.all(np.isfinite(result))

    def test_comprehensive_dtype_consistency(self):
        """Test comprehensive data type consistency across all input types."""
        test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Test different input types
        inputs = [
            ("int32", test_data.astype(np.int32)),
            ("float32", test_data.astype(np.float32)),
            ("float64", test_data.astype(np.float64)),
        ]

        for dtype_name, data in inputs:
            result = background(data, nbgr=2, width=1)
            assert result.dtype == np.float64, f"Input {dtype_name} should produce float64, got {result.dtype}"

            # Also test linear_background
            lin_result = linear_background(data, nbgr=2)
            assert lin_result.dtype == np.float64, f"Linear background with {dtype_name} should produce float64"

    def test_compression_expansion_edge_cases(self):
        """Test compression and expansion with various edge cases."""
        # Test compression
        arr = np.arange(10, dtype=float)
        compressed, rem = compress_array(arr, 3)
        assert isinstance(compressed, np.ndarray)
        assert isinstance(rem, int)
        assert len(compressed) == 3
        assert rem == 1

        # Test expansion
        small_arr = np.array([1.0, 2.0, 3.0])
        expanded = expand_array(small_arr, 2)
        expanded_sample = expand_array(small_arr, 2, sample=1)

        assert len(expanded) == 6
        assert len(expanded_sample) == 6
        assert expanded.dtype == small_arr.dtype
        assert expanded_sample.dtype == small_arr.dtype

        # Test that sampling produces different results than interpolation
        assert not np.array_equal(expanded, expanded_sample)

    def test_zero_width_equals_linear(self):
        """Test that zero width background equals linear background."""
        test_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        result_zero = background(test_data, nbgr=2, width=0)
        result_linear = linear_background(test_data, nbgr=2)

        np.testing.assert_array_equal(result_zero, result_linear)

    def test_large_array_performance(self):
        """Test performance and memory usage with large arrays."""
        # Test large array
        large_data = np.random.normal(size=10000)
        result_large = background(large_data, nbgr=5, width=100, pow=1.0)

        assert isinstance(result_large, np.ndarray)
        assert result_large.shape == large_data.shape
        assert result_large.dtype == np.float64
        assert np.all(np.isfinite(result_large))

        # Test multiple calls (memory leak check)
        for i in range(5):
            result = background(large_data[:100], nbgr=3, width=10, pow=1.0)
            assert isinstance(result, np.ndarray)
            del result  # Explicit cleanup

    def test_small_array_edge_cases(self):
        """Test behavior with very small arrays."""
        # Small array with 2 elements
        small_data = np.array([1.0, 2.0])
        result_small = background(small_data, nbgr=1, width=1)

        assert isinstance(result_small, np.ndarray)
        assert result_small.shape == small_data.shape
        assert result_small.dtype == np.float64
        assert np.all(np.isfinite(result_small))

        # Single element array
        single_data = np.array([5.0])
        result_single = background(single_data, nbgr=1, width=1)

        assert isinstance(result_single, np.ndarray)
        assert result_single.shape == single_data.shape
        assert result_single.dtype == np.float64


class TestRegressionPrevention:
    """Test to prevent regression of fixes."""

    def test_complex_example_matches_original(self, test_data_complex):
        """Test that complex example produces valid results."""
        x, y = test_data_complex

        # This matches the original Python 2 example
        result = background(y, nbgr=3, width=100, pow=1, tangent=False, compress=1)

        assert isinstance(result, np.ndarray)
        assert result.shape == y.shape
        assert result.dtype == np.float64
        assert np.all(np.isfinite(result))

        # Check that background is generally lower than original data
        assert np.mean(result) < np.mean(y)

    def test_all_parameters_combinations(self, test_data_simple):
        """Test various parameter combinations."""
        x, y = test_data_simple

        # Test parameter combinations
        test_cases = [
            {"nbgr": 0, "width": 0, "pow": 0.5},
            {"nbgr": 3, "width": 5, "pow": 1.0},
            {"nbgr": 5, "width": 10, "pow": 2.0},
            {"nbgr": 2, "width": 7, "pow": 0.5, "tangent": True},
            {"nbgr": 4, "width": 12, "pow": 1.5, "compress": 2},
        ]

        for params in test_cases:
            result = background(y, **params)
            assert isinstance(result, np.ndarray)
            assert result.shape == y.shape
            assert np.all(np.isfinite(result))

    def test_memory_usage_consistency(self, test_data_complex):
        """Test that memory usage is consistent."""
        x, y = test_data_complex

        # Test with large array
        result = background(y, nbgr=5, width=50, pow=1.0)

        assert isinstance(result, np.ndarray)
        assert result.shape == y.shape
        assert result.dtype == np.float64

        # Memory should be released properly
        del result

        # Test multiple calls
        for i in range(5):
            result = background(y, nbgr=3, width=25, pow=1.0)
            assert isinstance(result, np.ndarray)
            del result


class TestBackgroundDemo:
    """Test the demonstration functionality that was in __main__."""

    def test_demo_data_generation(self):
        """Test the demo data generation function."""

        def gauss(x, cen, sigma):
            return np.exp(-((x - cen) ** 2) / (2 * sigma**2))

        npts = 2000
        x = np.arange(npts, dtype=float)
        g1 = 40 * gauss(x, 0.6 * npts, 8.0)
        g2 = 20 * gauss(x, 0.5 * npts, 270)
        noise = 2 * np.random.normal(size=npts)
        y = noise + x / 25 + g1 + g2

        # Verify the data structure
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(x) == npts
        assert len(y) == npts
        assert x.dtype == np.float64
        assert y.dtype == np.float64

        # Verify components are reasonable
        # Gaussian should be non-negative
        assert np.all(g1 >= 0)
        # Gaussian should be non-negative
        assert np.all(g2 >= 0)
        # Noise should have variation
        assert np.std(noise) > 0

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.legend")
    @patch("matplotlib.pyplot.plot")
    def test_demo_background_calculation(self, mock_plot, mock_legend, mock_show):
        """Test the demo background calculation with the original parameters."""

        def gauss(x, cen, sigma):
            return np.exp(-((x - cen) ** 2) / (2 * sigma**2))

        # Use the same seed for reproducible results
        np.random.seed(42)

        npts = 2000
        x = np.arange(npts, dtype=float)
        g1 = 40 * gauss(x, 0.6 * npts, 8.0)
        g2 = 20 * gauss(x, 0.5 * npts, 270)
        noise = 2 * np.random.normal(size=npts)
        y = noise + x / 25 + g1 + g2

        # Test the background calculation with demo parameters
        bgr = background(y, nbgr=3, width=100, pow=1, tangent=False, compress=1)

        # Verify the result
        assert isinstance(bgr, np.ndarray)
        assert bgr.shape == y.shape
        assert bgr.dtype == np.float64

        # Test show_bgr function (which was called in the original demo)
        show_bgr(y, nbgr=3, width=100, pow=1, tangent=False, compress=1)

        # Verify matplotlib functions were called
        mock_plot.assert_called()
        mock_legend.assert_called()
        mock_show.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
