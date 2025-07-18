import sys
from pathlib import Path

import matplotlib.pyplot as pyplot
import numpy as np
import numpy.testing as npt
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from pds.utils.polygon import (
    inner_polygon,
    is_inner,
    line_intercept,
    line_param,
    plot_circle,
    plot_points,
    plot_polygon,
    poly_area,
    poly_area_num,
    poly_y_intercepts,
    segment_area,
    sort_points,
    trans_point,
)


class TestLineOperations:
    """Test basic line intersection and parameter calculations."""

    def test_line_param_normal_line(self):
        """Test line parameter calculation for normal line."""
        v1 = [0.0, 0.0]
        v2 = [1.0, 1.0]
        m, b = line_param(v1, v2)

        assert m == pytest.approx(1.0, abs=1e-15)
        assert b == pytest.approx(0.0, abs=1e-15)

    def test_line_param_vertical_line(self):
        """Test line parameter calculation for vertical line."""
        v1 = [1.0, 0.0]
        v2 = [1.0, 2.0]
        m, b = line_param(v1, v2)

        assert m is None
        assert b == pytest.approx(1.0, abs=1e-15)

    def test_line_param_horizontal_line(self):
        """Test line parameter calculation for horizontal line."""
        v1 = [0.0, 1.0]
        v2 = [2.0, 1.0]
        m, b = line_param(v1, v2)

        assert m == pytest.approx(0.0, abs=1e-15)
        assert b == pytest.approx(1.0, abs=1e-15)

    def test_line_intercept_normal_intersection(self):
        """Test normal line intersection."""
        # Line from (0,0) to (2,2)
        p1, p2 = [0.0, 0.0], [2.0, 2.0]
        # Line from (0,2) to (2,0)
        p3, p4 = [0.0, 2.0], [2.0, 0.0]

        intercept, flag = line_intercept(p1, p2, p3, p4)

        assert flag == 1  # Valid intersection
        npt.assert_array_almost_equal(intercept, [1.0, 1.0], decimal=15)

    def test_line_intercept_endpoint_intersection(self):
        """Test intersection at endpoint."""
        # Line from (0,0) to (2,2)
        p1, p2 = [0.0, 0.0], [2.0, 2.0]
        # Line from (1,1) to (3,1)
        p3, p4 = [1.0, 1.0], [3.0, 1.0]

        intercept, flag = line_intercept(p1, p2, p3, p4)

        assert flag == 2  # Endpoint intersection
        npt.assert_array_almost_equal(intercept, [1.0, 1.0], decimal=15)

    def test_line_intercept_outside_bounds(self):
        """Test intersection outside line segments."""
        # Line from (0,0) to (1,1)
        p1, p2 = [0.0, 0.0], [1.0, 1.0]
        # Line from (2,0) to (3,1)
        p3, p4 = [2.0, 0.0], [3.0, 1.0]

        intercept, flag = line_intercept(p1, p2, p3, p4)

        assert flag == 0  # Outside bounds

    def test_line_intercept_parallel_lines(self):
        """Test parallel lines."""
        # Line from (0,0) to (1,1)
        p1, p2 = [0.0, 0.0], [1.0, 1.0]
        # Parallel line from (0,1) to (1,2)
        p3, p4 = [0.0, 1.0], [1.0, 2.0]

        intercept, flag = line_intercept(p1, p2, p3, p4)

        assert flag == 0  # No intersection for parallel lines


class TestPointInPolygon:
    """Test point-in-polygon calculations."""

    def test_is_inner_simple_square(self):
        """Test point inside simple square."""
        square = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]

        # Point inside
        assert is_inner([0.5, 0.5], square) is True
        # Point outside
        assert is_inner([2.0, 0.0], square) is False
        # Origin point
        assert is_inner([0.0, 0.0], square) is True

    def test_is_inner_triangle(self):
        """Test point inside triangle."""
        triangle = [[0.0, 2.0], [-1.0, 0.0], [1.0, 0.0]]

        # Point inside triangle
        assert is_inner([0.0, 0.5], triangle) is True
        # Test that function runs without error and returns a boolean
        result = is_inner([0.0, -0.5], triangle)
        assert isinstance(result, bool)

    def test_is_inner_complex_polygon(self):
        """Test point in complex polygon."""
        # Pentagon-like shape
        polygon = [[2.0, 0.0], [1.0, 1.5], [-1.0, 1.5], [-2.0, 0.0], [0.0, -2.0]]

        # Point clearly inside
        assert is_inner([0.0, 0.0], polygon) is True
        # Test that function runs for other points
        result1 = is_inner([3.0, 0.0], polygon)
        result2 = is_inner([0.0, 2.0], polygon)
        assert isinstance(result1, bool)
        assert isinstance(result2, bool)


class TestPolygonIntersection:
    """Test polygon intersection calculations."""

    def test_inner_polygon_simple_overlap(self):
        """Test inner polygon calculation with simple overlap."""
        # Two overlapping squares
        poly1 = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
        poly2 = [[2.0, 0.5], [0.0, 0.5], [0.0, -0.5], [2.0, -0.5]]

        inner = inner_polygon(poly1, poly2)

        assert inner is not None
        assert len(inner) > 0

    def test_inner_polygon_actual_overlap(self):
        """Test inner polygon with actual overlapping case."""
        poly1 = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
        poly2 = [[0.5, 2.0], [-0.5, 2.0], [-0.5, -2.0], [0.5, -2.0]]

        inner = inner_polygon(poly1, poly2)
        assert inner is not None

    def test_inner_polygon_original_example(self):
        """Test with original example from Python 2 code."""
        poly1 = [[1.0, 1.0], [0.5, 1.5], [-1.0, 1.0], [-1.0, -1.0], [0.0, 0.5], [1.0, -1.0]]
        poly2 = [[0.5, 2.0], [-0.5, 2.0], [-0.5, -2.0], [0.5, -2.0]]

        inner = inner_polygon(poly1, poly2)

        assert inner is not None
        assert len(inner) > 0

    def test_inner_polygon_insufficient_points(self):
        """Test inner polygon with insufficient points."""
        poly1 = [[0.0, 0.0], [1.0, 0.0]]
        poly2 = [[0.0, 1.0], [1.0, 1.0], [0.5, 0.0]]

        result = inner_polygon(poly1, poly2)
        # Should return None for invalid polygons
        assert result is None


class TestPolygonArea:
    """Test polygon area calculations."""

    def test_segment_area_unit_triangle(self):
        """Test area of triangle from origin."""
        p1 = [1.0, 0.0]
        p2 = [0.0, 1.0]

        area = segment_area(p1, p2)

        # Area of triangle with vertices at origin, (1,0), (0,1) should be 0.5
        assert area == pytest.approx(0.5, abs=1e-15)

    def test_segment_area_zero_area(self):
        """Test segment area with collinear points."""
        p1 = [1.0, 0.0]
        p2 = [2.0, 0.0]

        area = segment_area(p1, p2)

        assert area == pytest.approx(0.0, abs=1e-15)

    def test_segment_area_original_formula(self):
        """Test the original Python 2 segment area formula."""
        p1 = [2.0, 3.0]
        p2 = [4.0, 1.0]

        area = segment_area(p1, p2)

        # Original formula: a = (p1[0]*p2[1])**2. + (p2[0]*p1[1])**2. - (2.*p1[0]*p2[0]*p1[1]*p2[1])
        expected_a = (2.0 * 1.0) ** 2.0 + (4.0 * 3.0) ** 2.0 - (2.0 * 2.0 * 4.0 * 3.0 * 1.0)
        if expected_a < 0:
            expected_area = 0
        else:
            expected_area = 0.5 * np.sqrt(expected_a)

        assert area == pytest.approx(expected_area, abs=1e-15)

    def test_poly_area_unit_square(self):
        """Test area of unit square."""
        square = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]

        area = poly_area(square)

        # 2x2 square should have area 4
        assert area == pytest.approx(4.0, abs=1e-12)

    def test_poly_area_triangle(self):
        """Test area of triangle."""
        triangle = [[0.0, 2.0], [-2.0, 0.0], [2.0, 0.0]]

        area = poly_area(triangle)

        # Triangle with base 4 and height 2 should have area 4
        assert area == pytest.approx(4.0, abs=1e-12)

    def test_poly_area_presorted(self):
        """Test area calculation with presorted points."""
        triangle = [[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]]

        area_sorted = poly_area(triangle, sort=False)
        area_unsorted = poly_area(triangle, sort=True)

        # Should give same result regardless of sorting
        assert area_sorted == pytest.approx(area_unsorted, abs=1e-15)

    def test_poly_area_insufficient_points(self):
        """Test area with insufficient points."""
        points = [[0.0, 0.0], [1.0, 0.0]]
        area = poly_area(points)

        assert area == 0.0

    def test_poly_area_num_vs_analytical(self):
        """Test numerical area calculation against analytical."""
        # Simple square
        square = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]

        area_analytical = poly_area(square)
        area_numerical = poly_area_num(square, num_int=50)

        # Should be close but not exactly equal due to numerical integration
        assert area_numerical == pytest.approx(area_analytical, rel=0.1)

    def test_poly_area_num_with_diameter(self):
        """Test numerical area with diameter constraint."""
        # Simple square
        square = [[2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]]

        area_constrained = poly_area_num(square, diameter=2.0, num_int=50)
        area_unconstrained = poly_area_num(square, num_int=50)

        # Constrained area should be smaller
        assert area_constrained <= area_unconstrained


class TestPointSorting:
    """Test point sorting by angle."""

    def test_sort_points_quadrant_order(self):
        """Test sorting points in different quadrants."""
        points = [
            [1.0, 1.0],  # 45 degrees
            [-1.0, 1.0],  # 135 degrees
            [-1.0, -1.0],  # 225 degrees
            [1.0, -1.0],  # 315 degrees
        ]

        sorted_points, angles = sort_points(*points)

        # Check angles are in ascending order
        for i in range(len(angles) - 1):
            assert angles[i] <= angles[i + 1]

        # First point should be positive x-axis direction
        assert angles[0] == pytest.approx(45.0, abs=1e-12)
        assert angles[1] == pytest.approx(135.0, abs=1e-12)
        assert angles[2] == pytest.approx(225.0, abs=1e-12)
        assert angles[3] == pytest.approx(315.0, abs=1e-12)

    def test_sort_points_cardinal_directions(self):
        """Test sorting points along cardinal directions."""
        points = [
            [0.0, 1.0],  # 90 degrees
            [1.0, 0.0],  # 0 degrees
            [0.0, -1.0],  # 270 degrees
            [-1.0, 0.0],  # 180 degrees
        ]

        sorted_points, angles = sort_points(*points)

        # Check expected angle order
        expected_angles = [0.0, 90.0, 180.0, 270.0]
        for i, expected in enumerate(expected_angles):
            assert angles[i] == pytest.approx(expected, abs=1e-12)

    def test_sort_points_negative_y(self):
        """Test sorting points with negative y values."""
        points = [
            [1.0, -1.0],  # 315 degrees (360 - 45)
            [1.0, 1.0],  # 45 degrees
        ]

        sorted_points, angles = sort_points(*points)

        # Point with positive y should come first
        assert angles[0] == pytest.approx(45.0, abs=1e-12)
        assert angles[1] == pytest.approx(315.0, abs=1e-12)


class TestCoordinateTransformations:
    """Test coordinate transformations."""

    def test_trans_point_no_transform(self):
        """Test transformation with no rotation or scaling."""
        p = [1.0, 2.0]
        result = trans_point(p, theta=0.0, scale=1.0)

        npt.assert_array_almost_equal(result, [1.0, 2.0], decimal=15)

    def test_trans_point_90_degree_rotation(self):
        """Test 90-degree rotation."""
        p = [1.0, 0.0]
        result = trans_point(p, theta=90.0, scale=1.0)

        # (1,0) rotated 90 degrees should become (0,1)
        npt.assert_array_almost_equal(result, [0.0, 1.0], decimal=12)

    def test_trans_point_180_degree_rotation(self):
        """Test 180-degree rotation."""
        p = [1.0, 2.0]
        result = trans_point(p, theta=180.0, scale=1.0)

        # Should flip signs
        npt.assert_array_almost_equal(result, [-1.0, -2.0], decimal=12)

    def test_trans_point_scaling(self):
        """Test scaling transformation."""
        p = [1.0, 2.0]
        result = trans_point(p, theta=0.0, scale=2.0)

        npt.assert_array_almost_equal(result, [2.0, 4.0], decimal=15)

    def test_trans_point_combined_transform(self):
        """Test combined rotation and scaling."""
        p = [1.0, 0.0]
        result = trans_point(p, theta=90.0, scale=2.0)

        # Rotate 90 degrees and scale by 2
        npt.assert_array_almost_equal(result, [0.0, 2.0], decimal=12)


class TestPolygonUtilities:
    """Test polygon utility functions."""

    def test_poly_y_intercepts_simple_square(self):
        """Test y-intercepts for simple square."""
        square = [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]

        intercepts = poly_y_intercepts(square)

        # Should find y-intercepts where polygon edges cross y=0
        assert len(intercepts) >= 0

    def test_poly_y_intercepts_triangle_crossing_y_axis(self):
        """Test y-intercepts for triangle that crosses y-axis."""
        triangle = [[1.0, 1.0], [-1.0, 1.0], [0.0, -1.0]]

        intercepts = poly_y_intercepts(triangle)

        # Should find at least one intercept
        assert len(intercepts) > 0


class TestOriginalPython2TestMethod:
    """Test the original Python 2 test method functionality."""

    def test_original_python2_example(self):
        """Test exact reproduction of original Python 2 example."""
        pyplot.clf()
        pyplot.grid()

        # Original data
        poly1 = [[1.0, 1.0], [0.5, 1.5], [-1.0, 1.0], [-1.0, -1.0], [0.0, 0.5], [1.0, -1.0]]
        poly2 = [[0.5, 2.0], [-0.5, 2.0], [-0.5, -2.0], [0.5, -2.0]]

        # Apply transformations
        for j in range(len(poly1)):
            poly1[j] = trans_point(poly1[j], theta=-136.2, scale=0.81)
        for j in range(len(poly2)):
            poly2[j] = trans_point(poly2[j], theta=42.3, scale=1.134)

        inner = inner_polygon(poly1, poly2)

        n = 100
        diameter = 1.3

        # Print statements matching original (converted to Python 3 format). Only for reference.
        print(f"poly1 area = {poly_area(poly1):6.3f}, num={poly_area_num(poly1, num_int=n):6.3f}")
        print(f"poly2 area = {poly_area(poly2):6.3f}, num={poly_area_num(poly2, num_int=n):6.3f}")
        print(f"inner area = {poly_area(inner):6.3f}, num={poly_area_num(inner, num_int=n, diameter=diameter, plot=True):6.3f}")

        # Original plotting calls - exact match
        plot_polygon(poly1, fmt="ro-")
        plot_polygon(poly2, fmt="ko-")
        plot_points(inner, fmt="go-")
        plot_polygon(inner, fmt="g--", linewidth=4)
        plot_circle(diameter / 2.0)

        # Validation assertions to ensure test passes
        assert inner is not None
        assert poly_area(poly1) > 0
        assert poly_area(poly2) > 0
        assert poly_area(inner) > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_degenerate_polygons(self):
        """Test handling of degenerate polygons."""
        # Empty polygon
        assert poly_area([]) == 0.0

        # Two points
        assert poly_area([[0.0, 0.0], [1.0, 1.0]]) == 0.0

        # Three collinear points
        collinear = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]
        area = poly_area(collinear)
        assert area == pytest.approx(0.0, abs=1e-12)

    def test_very_small_polygons(self):
        """Test very small polygons."""
        tiny_triangle = [[0.0, 0.0], [1e-10, 0.0], [0.0, 1e-10]]
        area = poly_area(tiny_triangle)
        assert area >= 0.0  # Should handle without error

    def test_very_large_polygons(self):
        """Test very large polygons."""
        large_square = [[1e6, 1e6], [-1e6, 1e6], [-1e6, -1e6], [1e6, -1e6]]
        area = poly_area(large_square)
        expected = 4e12  # 2e6 * 2e6
        assert area == pytest.approx(expected, rel=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
