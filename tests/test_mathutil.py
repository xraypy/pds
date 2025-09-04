import sys
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from pds.utils.mathutil import (
    arccosd,
    arcsind,
    arctand,
    ave,
    cartesian_angle,
    cartesian_mag,
    cosd,
    line,
    minimize,
    random,
    random_seed,
    sind,
    square,
    std,
    tand,
)


class TestBasicNumpyWrappers:
    """Test numpy wrapper functions that replaced deprecated numpy.ave()"""

    def test_ave_simple_array(self):
        """Test average calculation matches numpy.mean behavior"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ave(x)
        # numpy.ave() was deprecated, replaced with numpy.mean()
        expected = np.mean(x)
        assert result == expected
        assert result == 3.0

    def test_ave_negative_values(self):
        """Test average with negative values"""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = ave(x)
        expected = 0.0
        assert result == expected

    def test_ave_single_value(self):
        """Test average of single value"""
        x = np.array([42.0])
        result = ave(x)
        assert result == 42.0

    def test_std_simple_array(self):
        """Test standard deviation calculation"""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = std(x)
        expected = np.std(x)
        assert result == expected
        # Manual calculation: sqrt(sum((x-mean)^2)/n)
        mean_x = 3.0
        manual_std = np.sqrt(np.sum((x - mean_x) ** 2) / len(x))
        npt.assert_almost_equal(result, manual_std, decimal=10)

    def test_std_zero_variance(self):
        """Test standard deviation of constant array"""
        x = np.array([5.0, 5.0, 5.0, 5.0])
        result = std(x)
        assert result == 0.0


class TestTrigonometricFunctions:
    """Test trigonometric functions with degree conversion"""

    def test_cosd_basic_angles(self):
        """Test cosine of degrees for common angles"""
        # Test exact values
        assert cosd(0) == 1.0
        assert cosd(90) == pytest.approx(0.0, abs=1e-15)
        assert cosd(180) == pytest.approx(-1.0, abs=1e-15)
        assert cosd(270) == pytest.approx(0.0, abs=1e-15)
        assert cosd(360) == pytest.approx(1.0, abs=1e-15)

    def test_sind_basic_angles(self):
        """Test sine of degrees for common angles"""
        assert sind(0) == pytest.approx(0.0, abs=1e-15)
        assert sind(90) == pytest.approx(1.0, abs=1e-15)
        assert sind(180) == pytest.approx(0.0, abs=1e-15)
        assert sind(270) == pytest.approx(-1.0, abs=1e-15)
        assert sind(360) == pytest.approx(0.0, abs=1e-15)

    def test_tand_basic_angles(self):
        """Test tangent of degrees for common angles"""
        assert tand(0) == pytest.approx(0.0, abs=1e-15)
        assert tand(45) == pytest.approx(1.0, abs=1e-15)
        # tand(90) would be infinite, so we skip that
        assert tand(135) == pytest.approx(-1.0, abs=1e-15)
        assert tand(180) == pytest.approx(0.0, abs=1e-15)

    def test_arccosd_basic_values(self):
        """Test arccos in degrees for common values"""
        assert arccosd(1.0) == pytest.approx(0.0, abs=1e-15)
        assert arccosd(0.0) == pytest.approx(90.0, abs=1e-15)
        assert arccosd(-1.0) == pytest.approx(180.0, abs=1e-15)
        assert arccosd(0.5) == pytest.approx(60.0, abs=1e-12)

    def test_arcsind_basic_values(self):
        """Test arcsin in degrees for common values"""
        assert arcsind(0.0) == pytest.approx(0.0, abs=1e-15)
        assert arcsind(1.0) == pytest.approx(90.0, abs=1e-15)
        assert arcsind(-1.0) == pytest.approx(-90.0, abs=1e-15)
        assert arcsind(0.5) == pytest.approx(30.0, abs=1e-12)

    def test_arctand_basic_values(self):
        """Test arctan in degrees for common values"""
        assert arctand(0.0) == pytest.approx(0.0, abs=1e-15)
        assert arctand(1.0) == pytest.approx(45.0, abs=1e-15)
        assert arctand(-1.0) == pytest.approx(-45.0, abs=1e-15)
        # tan(30°) = 1/sqrt(3)
        assert arctand(1.0 / np.sqrt(3)) == pytest.approx(30.0, abs=1e-12)


class TestVectorOperations:
    """Test vector magnitude and angle calculations"""

    def test_cartesian_mag_unit_vectors(self):
        """Test magnitude calculation for unit vectors"""
        # Unit vectors along axes
        assert cartesian_mag(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)
        assert cartesian_mag(np.array([0.0, 1.0, 0.0])) == pytest.approx(1.0)
        assert cartesian_mag(np.array([0.0, 0.0, 1.0])) == pytest.approx(1.0)

    def test_cartesian_mag_known_vectors(self):
        """Test magnitude for vectors with known lengths"""
        # 3-4-5 triangle in 2D
        v = np.array([3.0, 4.0])
        assert cartesian_mag(v) == pytest.approx(5.0)

        # 3D vector
        v = np.array([2.0, 3.0, 6.0])
        expected = np.sqrt(4 + 9 + 36)
        assert cartesian_mag(v) == pytest.approx(expected)

    def test_cartesian_mag_zero_vector(self):
        """Test magnitude of zero vector"""
        v = np.array([0.0, 0.0, 0.0])
        assert cartesian_mag(v) == 0.0

    def test_cartesian_angle_orthogonal_vectors(self):
        """Test angle between orthogonal vectors"""
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 1.0, 0.0])
        angle = cartesian_angle(u, v)
        assert angle == pytest.approx(90.0, abs=1e-12)

    def test_cartesian_angle_parallel_vectors(self):
        """Test angle between parallel vectors"""
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([2.0, 4.0, 6.0])
        angle = cartesian_angle(u, v)
        assert angle == pytest.approx(0.0, abs=1e-12)

    def test_cartesian_angle_antiparallel_vectors(self):
        """Test angle between antiparallel vectors"""
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([-1.0, 0.0, 0.0])
        angle = cartesian_angle(u, v)
        assert angle == pytest.approx(180.0, abs=1e-12)

    def test_cartesian_angle_zero_vector(self):
        """Test angle with zero vector returns 0"""
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.0, 0.0, 0.0])
        angle = cartesian_angle(u, v)
        assert angle == 0.0

    def test_cartesian_angle_60_degrees(self):
        """Test specific angle calculation"""
        # Two vectors at 60 degrees
        u = np.array([1.0, 0.0])
        v = np.array([0.5, np.sqrt(3) / 2])
        angle = cartesian_angle(u, v)
        assert angle == pytest.approx(60.0, abs=1e-12)


class TestUtilityFunctions:
    """Test basic utility functions"""

    def test_line_calculation(self):
        """Test line function y = slope*x + offset"""
        # Test with scalar
        result = line(2.0, 3.0, 4.0)
        assert result == 11.0

        # Test with array
        x = np.array([0.0, 1.0, 2.0])
        result = line(x, 1.0, 2.0)
        expected = np.array([1.0, 3.0, 5.0])
        npt.assert_array_equal(result, expected)

    def test_square_function(self):
        """Test square function"""
        assert square(3.0) == 9.0
        assert square(-4.0) == 16.0
        assert square(0.0) == 0.0
        assert square(2.5) == 6.25


class TestMinimizeFunction:
    """Test scipy.optimize.leastsq wrapper function"""

    def test_minimize_linear_fit(self):
        """Test minimizing a linear function"""

        # Define a linear function to fit
        def linear_func(x, a, b):
            return a * x + b

        # Generate test data: y = 2*x + 3 with some noise
        x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_true = 2.0 * x_data + 3.0
        y_data = y_true

        # Initial guess for parameters [a, b]
        initial_params = (1.0, 1.0)

        # Minimize
        result = minimize(linear_func, x_data, y_data, initial_params)

        # Should recover original parameters
        assert result[0] == pytest.approx(2.0, abs=1e-10)
        assert result[1] == pytest.approx(3.0, abs=1e-10)

    def test_minimize_quadratic_fit(self):
        """Test minimizing a quadratic function"""

        def quadratic_func(x, a, b, c):
            return a * x**2 + b * x + c

        # Generate test data: y = 1*x^2 + 2*x + 3
        x_data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_data = 1.0 * x_data**2 + 2.0 * x_data + 3.0

        # Initial guess for parameters [a, b, c]
        initial_params = (0.5, 1.0, 2.0)

        # Minimize
        result = minimize(quadratic_func, x_data, y_data, initial_params)

        # Should recover original parameters
        assert result[0] == pytest.approx(1.0, abs=1e-10)
        assert result[1] == pytest.approx(2.0, abs=1e-10)
        assert result[2] == pytest.approx(3.0, abs=1e-10)

    def test_minimize_with_extra_args(self):
        """Test minimize with additional function arguments"""

        def func_with_args(x, a, b, c):
            return a * x + b + c

        x_data = np.array([1.0, 2.0, 3.0])
        c_value = 10.0  # Fixed parameter
        y_data = 2.0 * x_data + 5.0 + c_value

        # Minimize with c as additional argument
        result = minimize(func_with_args, x_data, y_data, (1.0, 1.0), c_value)

        assert result[0] == pytest.approx(2.0, abs=1e-10)
        assert result[1] == pytest.approx(5.0, abs=1e-10)


class TestRandomFunctions:
    """Test random number generation and seeding"""

    def test_random_seed_reproducibility(self):
        """Test that random seed produces reproducible results"""
        # Set seed and generate numbers
        random_seed(42)
        first_random = random(npts=5, distribution="normal")

        # Reset seed and generate again
        random_seed(42)
        second_random = random(npts=5, distribution="normal")

        # Should be identical
        npt.assert_array_equal(first_random, second_random)

    def test_random_seed_none(self):
        """Test random seed with None (should not crash)"""
        random_seed(None)  # Should work without error
        result = random(npts=3, distribution="normal")
        assert len(result) == 3

    def test_random_normal_distribution(self):
        """Test normal distribution generation"""
        random_seed(123)  # For reproducibility
        result = random(a=0, b=1, npts=1000, distribution="normal")

        # Check basic properties
        assert len(result) == 1000
        # For normal distribution, about 68% should be within 1 std dev
        within_one_std = np.sum(np.abs(result) <= 1)
        assert within_one_std > 600  # Rough check

    def test_random_uniform_distribution(self):
        """Test uniform distribution generation"""
        random_seed(456)
        result = random(a=0, b=10, npts=100, distribution="uniform")

        # All values should be in range [0, 10]
        assert np.all(result >= 0)
        assert np.all(result <= 10)
        # Mean should be approximately 5
        assert np.mean(result) == pytest.approx(5.0, abs=1.0)

    def test_random_exponential_distribution(self):
        """Test exponential distribution generation"""
        random_seed(789)
        scale = 2.0
        result = random(a=scale, npts=1000, distribution="exponential")

        # All values should be positive
        assert np.all(result >= 0)
        # Mean should be approximately equal to scale
        assert np.mean(result) == pytest.approx(scale, abs=0.5)

    def test_random_single_value(self):
        """Test single random value generation"""
        result = random(npts=1, distribution="normal")
        assert isinstance(result, (float, np.floating))

    def test_random_multiple_values(self):
        """Test multiple random values generation"""
        result = random(npts=5, distribution="normal")
        assert isinstance(result, np.ndarray)
        assert len(result) == 5

    def test_random_poisson_distribution(self):
        """Test Poisson distribution generation"""
        random_seed(101)
        lam = 3.0  # Lambda parameter
        result = random(a=lam, npts=1000, distribution="poisson")

        # All values should be non-negative integers
        assert np.all(result >= 0)
        # Mean should be approximately lambda
        assert np.mean(result) == pytest.approx(lam, abs=0.3)

    def test_random_binomial_distribution(self):
        """Test binomial distribution generation"""
        random_seed(202)
        n, p = 10, 0.3  # Parameters
        result = random(a=n, b=p, npts=1000, distribution="binomial")

        # All values should be in range [0, n]
        assert np.all(result >= 0)
        assert np.all(result <= n)
        # Mean should be approximately n*p
        assert np.mean(result) == pytest.approx(n * p, abs=0.5)

    def test_random_gamma_distribution(self):
        """Test gamma distribution generation"""
        random_seed(303)
        shape, scale = 2.0, 1.5
        result = random(a=shape, b=scale, npts=1000, distribution="gamma")

        # All values should be positive
        assert np.all(result > 0)
        # Mean should be approximately shape*scale
        expected_mean = shape * scale
        assert np.mean(result) == pytest.approx(expected_mean, abs=0.5)

    def test_random_beta_distribution(self):
        """Test beta distribution generation"""
        random_seed(404)
        alpha, beta = 2.0, 3.0
        result = random(a=alpha, b=beta, npts=1000, distribution="beta")

        # All values should be in range [0, 1]
        assert np.all(result >= 0)
        assert np.all(result <= 1)
        # Mean should be approximately alpha/(alpha+beta)
        expected_mean = alpha / (alpha + beta)
        assert np.mean(result) == pytest.approx(expected_mean, abs=0.1)


class TestPython2Compatibility:
    """Tests specifically for Python 2 compatibility"""

    def test_ave_vs_numpy_ave_deprecated(self):
        """Verify our ave() matches numpy.mean() since numpy.ave() was deprecated"""
        x = np.array([1.5, 2.7, 3.1, 4.9, 5.3])
        our_result = ave(x)
        numpy_result = np.mean(x)

        # In Python 2, numpy.ave() was just an alias for numpy.mean()
        assert our_result == numpy_result

        # Verify exact calculation
        expected = sum(x) / len(x)
        assert our_result == pytest.approx(expected)

    def test_type_checking_modernized(self):
        """Test that type checking works with modern Python 3 types"""
        # Original Python 2 used types.DictionaryType
        # Modern Python 3 uses dict or type checking

        def test_func(x, a, b, **kwargs):
            return a * x + b

        x_data = np.array([1.0, 2.0])
        y_data = np.array([3.0, 5.0])  # y = 2*x + 1

        # Test with keyword arguments (dict)
        result = minimize(test_func, x_data, y_data, (1.0, 1.0), d=1.0)

        assert result[0] == pytest.approx(2.0, abs=1e-10)
        assert result[1] == pytest.approx(1.0, abs=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
