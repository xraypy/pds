import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import pytest
from matplotlib import pyplot

# Add the project root to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pds.utils import gonio_psic
from pds.utils.active_area import (
    _area_polygon,
    _area_round,
    active_area,
    calc_surf_transform,
    surface_intercept,
    surface_intercept_bounds,
)
from pds.utils.mathutil import cartesian_mag
from tests.test_gonio_psic import psic_spec_array


class TestSurfaceIntercept:
    """Test surface intercept calculations."""

    def test_surface_intercept_basic(self):
        """Test basic surface intercept calculation."""
        k = np.array([0.0, 1.0, -0.1])
        v = np.array([0.2, 0.0, 1.0])

        result = surface_intercept(k, v)

        # Expected calculation: x = 0.2 - (0/(-0.1))*1 = 0.2, y = 0 - (1/(-0.1))*1 = 10
        expected = np.array([0.2, 10.0])
        npt.assert_array_almost_equal(result, expected)

    def test_surface_intercept_parallel_k(self):
        """Test surface intercept when k is parallel to surface."""
        k = np.array([1.0, 1.0, 0.0])  # k[2] = 0
        v = np.array([0.2, 0.0, 1.0])

        result = surface_intercept(k, v)

        assert result is None

    def test_surface_intercept_origin(self):
        """Test surface intercept with vector at origin in z."""
        k = np.array([0.0, 1.0, -1.0])
        v = np.array([1.0, 1.0, 0.0])

        result = surface_intercept(k, v)

        # Expected: x = 1 - (0/(-1))*0 = 1, y = 1 - (1/(-1))*0 = 1
        expected = np.array([1.0, 1.0])
        npt.assert_array_almost_equal(result, expected)

    def test_surface_intercept_bounds_within_diameter(self):
        """Test bounded intercept when point is within diameter."""
        k = np.array([0.0, 1.0, -1.0])
        v = np.array([0.1, 0.0, 1.0])
        diameter = 2.0

        result = surface_intercept_bounds(k, v, diameter)

        # Point should be within diameter, so should equal unbounded result
        unbounded = surface_intercept(k, v)
        npt.assert_array_almost_equal(result, unbounded, decimal=2)

    def test_surface_intercept_bounds_outside_diameter(self):
        """Test bounded intercept when point is outside diameter."""
        k = np.array([0.0, 1.0, -0.1])
        v = np.array([2.0, 0.0, 1.0])
        diameter = 1.0

        result = surface_intercept_bounds(k, v, diameter)

        # Result should be constrained to diameter/2 = 0.5
        assert cartesian_mag(result) == pytest.approx(0.5, abs=1e-10)

    def test_surface_intercept_bounds_vertical_k(self):
        """Test bounded intercept with vertical k vector."""
        k = np.array([0.0, 0.0, -1.0])  # ks = [0, 0], ks_mag = 0
        v = np.array([2.0, 0.0, 1.0])
        diameter = 1.0

        result = surface_intercept_bounds(k, v, diameter)

        # Should scale vi directly since ks_mag = 0
        assert cartesian_mag(result) == pytest.approx(0.5, abs=1e-10)


class TestSurfaceTransform:
    """Test surface coordinate system transformation."""

    def test_calc_surf_transform_identity(self):
        """Test transformation with surface normal along z-axis."""
        nm = np.array([0.0, 0.0, 1.0])  # Surface normal along z

        M = calc_surf_transform(nm)

        # Should be close to identity for this case
        assert M.shape == (3, 3)
        assert np.allclose(np.linalg.det(M), 1.0, atol=1e-10)  # Orthogonal matrix

    def test_calc_surf_transform_orthogonal(self):
        """Test that transformation matrix is orthogonal."""
        nm = np.array([1.0, 1.0, 1.0])  # Arbitrary surface normal

        M = calc_surf_transform(nm)

        # Check orthogonality: M @ M.T should be identity
        should_be_identity = M @ M.T
        npt.assert_array_almost_equal(should_be_identity, np.eye(3), decimal=10)

    def test_calc_surf_transform_determinant(self):
        """Test that transformation preserves orientation (det = 1)."""
        nm = np.array([0.5, -0.5, 1.0])

        M = calc_surf_transform(nm)

        # Should be proper rotation (det = 1, not -1)
        assert np.allclose(np.linalg.det(M), 1.0, atol=1e-10)

    def test_calc_surf_transform_normalized_input(self):
        """Test that function works with non-normalized input."""
        nm_unnormalized = np.array([2.0, 0.0, 2.0])
        nm_normalized = nm_unnormalized / cartesian_mag(nm_unnormalized)

        M1 = calc_surf_transform(nm_unnormalized)
        M2 = calc_surf_transform(nm_normalized)

        # Should give same result regardless of normalization
        npt.assert_array_almost_equal(M1, M2, decimal=10)


class TestAreaCalculations:
    """Test area calculation functions."""

    def test_area_round_no_detector(self):
        """Test round area calculation without detector."""
        beam_poly = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
        det_poly = None
        diameter = None

        A_beam, A_int = _area_round(beam_poly, det_poly, diameter=diameter)

        assert A_beam == pytest.approx(4.0, abs=1e-10)
        assert A_int == pytest.approx(4.0, abs=1e-10)

    def test_area_round_with_diameter(self):
        """Test round area calculation with sample diameter constraint."""
        beam_poly = [[2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]]
        det_poly = None
        diameter = 2.0  # Circle smaller than beam

        A_beam, A_int = _area_round(beam_poly, det_poly, diameter=diameter)

        assert A_beam == pytest.approx(16.0, abs=1e-10)
        # A_int should be less than A_beam due to circular constraint
        assert A_int < A_beam
        assert A_int > 0

    def test_area_round_with_detector(self):
        """Test round area calculation with detector polygon."""
        beam_poly = [[2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]]
        det_poly = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
        diameter = None

        A_beam, A_int = _area_round(beam_poly, det_poly, diameter=diameter)

        assert A_beam == pytest.approx(16.0, abs=1e-10)
        assert A_int == pytest.approx(8.0, abs=1e-10)
        """Test round area calculation with zero diameter."""
        beam_poly = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
        det_poly = None
        diameter = 0.0  # Should be treated as None

        A_beam, A_int = _area_round(beam_poly, det_poly, diameter=diameter)

        assert A_beam == A_int  # No diameter constraint

    def test_area_round_none_beam(self):
        """Test round area calculation with None beam."""
        beam_poly = None
        det_poly = None
        diameter = 1.0

        A_beam, A_int = _area_round(beam_poly, det_poly, diameter=diameter)

        assert A_beam == 0.0
        assert A_int == 0.0

    def test_area_polygon_basic(self):
        """Test polygon area calculation."""
        beam_poly = [[2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]]
        det_poly = None
        sam_poly = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]

        A_beam, A_int = _area_polygon(beam_poly, det_poly, sam_poly)

        assert A_beam == pytest.approx(16.0, abs=1e-10)
        assert A_int == pytest.approx(8.0, abs=1e-10)
        """Test polygon area calculation without sample polygon."""
        beam_poly = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
        det_poly = None
        sam_poly = None

        A_beam, A_int = _area_polygon(beam_poly, det_poly, sam_poly)

        assert A_beam == pytest.approx(4.0, abs=1e-10)
        assert A_int == pytest.approx(4.0, abs=1e-10)

    def test_area_polygon_with_detector(self):
        """Test polygon area calculation with detector."""
        beam_poly = [[3.0, 3.0], [-3.0, 3.0], [-3.0, -3.0], [3.0, -3.0]]
        det_poly = [[2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]]
        sam_poly = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]

        A_beam, A_int = _area_polygon(beam_poly, det_poly, sam_poly)

        assert A_beam == pytest.approx(36.0, abs=1e-10)
        assert A_int == pytest.approx(18.0, abs=1e-10)

    def test_area_polygon_none_beam(self):
        """Test polygon area calculation with None beam."""
        beam_poly = None
        det_poly = None
        sam_poly = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]

        A_beam, A_int = _area_polygon(beam_poly, det_poly, sam_poly)

        assert A_beam == 0.0
        assert A_int == 0.0


class TestActiveAreaMain:
    """Test main active_area function."""

    def test_active_area_basic(self):
        """Test basic active area calculation."""
        nm = np.array([0.0, 0.0, 1.0])
        ki = np.array([0.0, 1.0, -0.1])
        kr = np.array([0.0, 1.0, -0.1])
        beam = [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]
        det = None
        sample = 2.0  # Round sample

        result = active_area(nm, ki, kr, beam, det, sample)

        assert isinstance(result, tuple)
        assert len(result) == 2
        A_beam, A_int = result
        assert A_beam > 0
        assert A_int > 0
        assert A_int <= A_beam

    def test_active_area_polygon_sample(self):
        """Test active area with polygon sample."""
        nm = np.array([0.0, 0.0, 1.0])
        ki = np.array([0.0, 1.0, -0.1])
        kr = np.array([0.0, 1.0, -0.1])
        beam = [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]
        sample = [[0.5, 0.0, 0.5], [-0.5, 0.0, 0.5], [-0.5, 0.0, -0.5], [0.5, 0.0, -0.5]]

        result = active_area(nm, ki=ki, kr=kr, beam=beam, sample=sample)

        A_beam, A_int = result
        assert A_beam > A_int

    def test_active_area_invalid_beam(self):
        """Test active area with invalid beam specification."""
        nm = np.array([0.0, 0.0, 1.0])
        beam = [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]

        result = active_area(nm, beam=beam)

        assert result == (0.0, 0.0)

    def test_active_area_invalid_detector(self):
        """Test active area with invalid detector specification."""
        nm = np.array([0.0, 0.0, 1.0])
        beam = [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]
        det = [[1.0, 0.0, 1.0]]

        result = active_area(nm, beam=beam, det=det)

        assert result == (0.0, 0.0)

    def test_active_area_invalid_sample(self):
        """Test active area with invalid sample specification."""
        nm = np.array([0.0, 0.0, 1.0])
        beam = [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]
        sample = [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]

        result = active_area(nm, beam=beam, sample=sample)

        assert result == (0.0, 0.0)

    def test_active_area_no_beam(self):
        """Test active area with no beam specification."""
        nm = np.array([0.0, 0.0, 1.0])

        result = active_area(nm, beam=None)

        assert result == (0.0, 0.0)

    def test_active_area_none_sample(self):
        """Test active area with None sample (infinite sample)."""
        nm = np.array([0.0, 0.0, 1.0])
        beam = [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]
        sample = None

        result = active_area(nm, beam=beam, sample=sample)

        A_beam, A_int = result
        assert A_beam == A_int

    def test_active_area_with_plotting(self):
        """Test active area calculation with plotting enabled."""
        nm = np.array([0.0, 0.0, 1.0])
        beam = [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]
        sample = 1.0

        # Should not raise any errors
        result = active_area(nm, beam=beam, sample=sample, plot=True, fig=99)

        assert isinstance(result, tuple)
        assert len(result) == 2
        pyplot.close(99)

    def test_active_area_parallel_k_vector(self):
        """Test active area when k vector is parallel to surface."""
        nm = np.array([0.0, 0.0, 1.0])
        ki = np.array([1.0, 0.0, 0.0])
        beam = [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]

        # Should handle gracefully by filtering out None intercepts
        active_area(nm, ki=ki, beam=beam)

    def test_empty_beam_list_compatibility(self):
        """Test that empty beam lists are handled the same as None (Python 2 compatibility)."""
        nm = np.array([0.0, 0.0, 1.0])

        # Empty list should be treated same as None
        result_empty = active_area(nm, beam=[])
        result_none = active_area(nm, beam=None)

        assert result_empty == result_none, "Empty beam list should be treated same as None for Python 2 compatibility"

        # Should both return default behavior
        assert result_empty == (0.0, 0.0)
        assert result_none == (0.0, 0.0)

    def test_insufficient_beam_vertices(self):
        """Test that beam lists with insufficient vertices are handled correctly."""
        nm = np.array([0.0, 0.0, 1.0])

        # Test single vertex
        result = active_area(nm, beam=[[1.0, 0.0, 0.0]])
        assert result == (0.0, 0.0)

        # Test two vertices
        result = active_area(nm, beam=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert result == (0.0, 0.0)


class TestCompatibilityWithOriginal:
    """Test compatibility with original Python 2 behavior."""

    def test_types_compatibility(self):
        """Test that function handles various input types like original."""
        nm = np.array([0.0, 0.0, 1.0])

        # Test with lists (original used these)
        beam_list = [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]
        result1 = active_area(nm, beam=beam_list)

        # Test with numpy arrays
        beam_array = [np.array([1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 0.0, -1.0]), np.array([1.0, 0.0, -1.0])]
        result2 = active_area(nm, beam=beam_array)

        # Should get similar results
        assert abs(result1[0] - result2[0]) < 1e-10
        assert abs(result1[1] - result2[1]) < 1e-10

    def test_division_behavior_python3(self):
        """Test that division behaves consistently in Python 3."""
        # In Python 2, / was integer division for ints, // was floor division
        # In Python 3, / is true division, // is floor division
        nm = np.array([0.0, 0.0, 1.0])
        ki = np.array([0.0, 1.0, -0.1])
        kr = np.array([0.0, 1.0, -0.1])

        # Use coordinates that would show division differences
        beam = [[1.0, 0.0, 3.0], [-1.0, 0.0, 3.0], [-1.0, 0.0, -3.0], [1.0, 0.0, -3.0]]
        sample = 1.5

        result = active_area(nm, ki=ki, kr=kr, beam=beam, sample=sample)

        # Should get reasonable floating point results
        assert result[0] > 0
        assert result[1] > 0
        assert not np.isnan(result[0])
        assert not np.isnan(result[1])


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_very_small_vectors(self):
        """Test with very small vectors."""
        nm = np.array([0.0, 0.0, 1e-10])
        beam = [[1e-10, 0.0, 1e-10], [-1e-10, 0.0, 1e-10], [-1e-10, 0.0, -1e-10], [1e-10, 0.0, -1e-10]]

        # Should handle gracefully
        result = active_area(nm, beam=beam)
        assert isinstance(result, tuple)

    def test_very_large_vectors(self):
        """Test with very large vectors."""
        nm = np.array([0.0, 0.0, 1e6])
        beam = [[1e6, 0.0, 1e6], [-1e6, 0.0, 1e6], [-1e6, 0.0, -1e6], [1e6, 0.0, -1e6]]

        result = active_area(nm, beam=beam)
        assert isinstance(result, tuple)
        assert not np.isinf(result[0])
        assert not np.isinf(result[1])

    def test_zero_magnitude_normal(self):
        """Test with zero magnitude normal vector."""
        nm = np.array([0.0, 0.0, 0.0])
        beam = [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]

        # Should raise an error or handle gracefully
        with pytest.warns(RuntimeWarning, match="invalid value encountered in divide"):
            with pytest.raises((ZeroDivisionError, ValueError)):
                active_area(nm, beam=beam)

    def test_negative_sample_diameter(self):
        """Test with negative sample diameter."""
        nm = np.array([0.0, 0.0, 1.0])
        beam = [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]
        sample = -1.0

        result = active_area(nm, beam=beam, sample=sample)

        # Negative diameter should be treated as None (infinite sample)
        A_beam, A_int = result
        assert A_beam == A_int

    def test_sample_height_warning(self):
        """Test warning for sample height problems."""
        nm = np.array([0.0, 0.0, 1.0])
        ki = np.array([0.0, 1.0, -0.1])  # Incident beam
        kr = np.array([0.0, 1.0, -0.1])  # Angled beam
        beam = [[1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, -1.0], [1.0, 0.0, -1.0]]
        sample = [[1.0, 0.0, 0.1], [-1.0, 0.0, 0.1], [-1.0, 0.0, -0.1], [1.0, 0.0, -0.1]]  # High z values

        with patch("builtins.print") as mock_print:
            active_area(nm, ki=ki, kr=kr, beam=beam, sample=sample)
            # Should print warning about sample height
            assert any("Warning" in str(call) for call in mock_print.call_args_list)


class TestOriginalTestFunctions:
    """Test the original test1 and test2 functions from the Python 2 code."""

    def test_original_test1_functionality(self):
        """Test surface intercept functions with simple circular geometry (original test1)."""
        r = 0.5
        k = [0.0, 1.0, -0.1]
        v = [0.2, 0.0, 1.0]

        # Test the functions work without errors
        v1 = surface_intercept(k, v)
        v2 = surface_intercept_bounds(k, v, 2.0 * r)

        # Verify results are reasonable
        assert v1 is not None
        assert isinstance(v1, np.ndarray)
        assert len(v1) == 2

        assert v2 is not None
        assert isinstance(v2, np.ndarray)
        assert len(v2) == 2

        # Test with plotting (should not raise errors)
        from pds.utils.polygon import plot_circle, plot_points

        plot_circle(r)
        plot_points([v1, v2])
        pyplot.grid()
        pyplot.clf()  # Clean up

    def test_original_test2_functionality_with_mock_gonio(self):
        """Test complete active area calculation using mock gonio_psic geometry (original test2)."""

        # Create mock surface normal and k vectors similar to what gonio_psic would provide
        nm = np.array([0.1, 0.2, 0.95])  # Slightly tilted surface
        ki = np.array([0.0, 0.9, -0.1])  # Incident beam
        kr = np.array([0.1, 0.9, -0.2])  # Diffracted beam

        # Create beam and detector vectors similar to gonio_psic output
        beam = [
            [1.3, 0.0, 1.0],
            [-1.3, 0.0, 1.0],
            [-1.3, 0.0, -1.0],
            [1.3, 0.0, -1.0],
            [1.0, 0.1, 1.0],
            [-1.0, 0.1, 1.0],
            [-1.0, 0.1, -1.0],
            [1.0, 0.1, -1.0],
        ]

        det = [
            [2.0, 0.0, 1.5],
            [-2.0, 0.0, 1.5],
            [-2.0, 0.0, -1.5],
            [2.0, 0.0, -1.5],
            [1.5, 0.0, 1.0],
            [-1.5, 0.0, 1.0],
            [-1.5, 0.0, -1.0],
            [1.5, 0.0, -1.0],
        ]

        # Create sample vectors (polygon)
        sample = [
            [1.0, 1.0, 0.0],
            [0.5, 1.5, 0.0],
            [-1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [0.0, 0.5, 0.0],
            [1.0, -1.0, 0.0],
        ]

        # Test the active area calculation
        result = active_area(nm, ki=ki, kr=kr, beam=beam, det=det, sample=sample, plot=False)

        # Verify we get reasonable results
        assert isinstance(result, tuple)
        assert len(result) == 2
        A_beam, A_int = result
        assert A_beam >= 0
        assert A_int >= 0
        assert A_int <= A_beam

        # Test with plotting enabled (should not raise errors)
        result_with_plot = active_area(nm, ki=ki, kr=kr, beam=beam, det=det, sample=sample, plot=True)
        assert result_with_plot == result  # Should get same numerical result
        pyplot.clf()  # Clean up

    def test_original_test2_with_real_gonio_if_available(self):
        """Test with real gonio_psic using test_psic_spec_array function."""

        # Use the available psic_spec_array function from test_gonio_psic
        psic = psic_spec_array(show=False)

        # Get beam and detector vectors using current angles from psic
        beam = gonio_psic.beam_vectors(h=1.3, v=0.1)
        det = gonio_psic.det_vectors(h=2.0, v=1.5, nu=psic.angles["nu"], delta=psic.angles["delta"])

        # Get sample vectors
        sample = [[1.0, 1.0], [0.5, 1.5], [-1.0, 1.0], [-1.0, -1.0], [0.0, 0.5], [1.0, -1.0]]
        angles = {"phi": 108.0007, "chi": 0.4831}
        sample_vectors = gonio_psic.sample_vectors(sample, angles=angles, gonio=psic)

        # Compute active area
        result = active_area(psic.nm, ki=psic.ki, kr=psic.kr, beam=beam, det=det, sample=sample_vectors, plot=False)

        # Verify results
        assert isinstance(result, tuple)
        assert len(result) == 2
        A_beam, A_int = result
        assert A_beam >= 0
        assert A_int >= 0


if __name__ == "__main__":
    pytest.main([__file__])
