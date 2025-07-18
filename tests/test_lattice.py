import sys
from pathlib import Path

import numpy as np
import pytest

# Add the pds module to the path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from pds.utils.lattice import Lattice, LatticeTransform


class TestLatticeBasics:
    """Test basic Lattice class functionality."""

    def test_lattice_initialization_default(self):
        """Test default lattice initialization."""
        cell = Lattice()
        assert cell.a == 10.0
        assert cell.b == 10.0
        assert cell.c == 10.0
        assert cell.alpha == 90.0
        assert cell.beta == 90.0
        assert cell.gamma == 90.0
        assert cell.lam == 1.5406

    def test_lattice_initialization_custom(self):
        """Test custom lattice initialization."""
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)
        assert cell.a == 5.6
        assert cell.b == 5.6
        assert cell.c == 13.0
        assert cell.alpha == 90.0
        assert cell.beta == 90.0
        assert cell.gamma == 120.0

    def test_lattice_repr(self):
        """Test lattice string representation."""
        cell = Lattice(5.0, 6.0, 7.0, 80.0, 85.0, 110.0, lam=1.2)
        repr_str = repr(cell)
        assert "a=5.00000" in repr_str
        assert "b=6.00000" in repr_str
        assert "c=7.00000" in repr_str
        assert "alpha=80.00000" in repr_str
        assert "beta=85.00000" in repr_str
        assert "gamma=110.00000" in repr_str
        assert "Default wavelength" in repr_str
        assert "1.20000" in repr_str

    def test_lattice_update(self):
        """Test lattice parameter updates."""
        cell = Lattice()
        cell.update(a=8.0, gamma=120.0, lam=1.0)
        assert cell.a == 8.0
        assert cell.b == 10.0
        assert cell.gamma == 120.0
        assert cell.lam == 1.0

    def test_cell_array(self):
        """Test cell parameter array."""
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)
        cell_params = cell.cell()
        expected = np.array([5.6, 5.6, 13.0, 90.0, 90.0, 120.0])
        np.testing.assert_array_almost_equal(cell_params, expected)

    def test_rcell_array(self):
        """Test reciprocal cell parameter array."""
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)
        rcell_params = cell.rcell()
        assert len(rcell_params) == 6
        assert rcell_params[0] > 0
        assert rcell_params[1] > 0
        assert rcell_params[2] > 0


class TestMetricTensor:
    """Test metric tensor calculations."""

    def test_cubic_metric_tensor(self):
        """Test metric tensor for cubic lattice."""
        cell = Lattice(5.0, 5.0, 5.0, 90, 90, 90)
        expected_g = np.array([[25.0, 0.0, 0.0], [0.0, 25.0, 0.0], [0.0, 0.0, 25.0]])
        np.testing.assert_array_almost_equal(cell.g, expected_g)

    def test_hexagonal_metric_tensor(self):
        """Test metric tensor for hexagonal lattice."""
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)
        # Check diagonal terms
        assert abs(cell.g[0, 0] - 5.6**2) < 1e-10
        assert abs(cell.g[1, 1] - 5.6**2) < 1e-10
        assert abs(cell.g[2, 2] - 13**2) < 1e-10
        # Check off-diagonal terms for 120° gamma
        expected_ab = 5.6 * 5.6 * np.cos(np.radians(120))
        assert abs(cell.g[0, 1] - expected_ab) < 1e-10

    def test_reciprocal_metric_tensor(self):
        """Test reciprocal metric tensor is inverse of real."""
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)
        identity = np.dot(cell.g, cell.gr)
        np.testing.assert_array_almost_equal(identity, np.eye(3), decimal=10)


class TestVolumeCalculations:
    """Test unit cell volume calculations."""

    def test_cubic_volume(self):
        """Test volume of cubic cell."""
        cell = Lattice(5.0, 5.0, 5.0, 90, 90, 90)
        vol = cell.vol()
        assert abs(vol - 125.0) < 1e-10

    def test_hexagonal_volume(self):
        """Test volume of hexagonal cell."""
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)
        vol = cell.vol()
        # Expected: a*b*c*sqrt(3)/2 for hexagonal
        expected = 5.6 * 5.6 * 13 * np.sqrt(3) / 2
        assert abs(vol - expected) < 1e-10

    def test_reciprocal_volume(self):
        """Test reciprocal volume calculation."""
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)
        vol_real = cell.vol()
        vol_recip = cell.vol(recip=True)
        # Reciprocal volume should be 1/real_volume
        assert abs(vol_real * vol_recip - 1.0) < 1e-10


class TestVectorOperations:
    """Test vector operations with metric tensor."""

    def test_dot_product_cubic(self):
        """Test dot product in cubic lattice."""
        cell = Lattice(5.0, 5.0, 5.0, 90, 90, 90)
        u = [1, 0, 0]
        v = [0, 1, 0]
        dot = cell.dot(u, v)
        assert abs(dot) < 1e-10

        u = [1, 1, 0]
        v = [1, 1, 0]
        dot = cell.dot(u, v)
        expected = 2 * 25.0
        assert abs(dot - expected) < 1e-10

    def test_vector_magnitude(self):
        """Test vector magnitude calculations."""
        cell = Lattice(5.0, 5.0, 5.0, 90, 90, 90)
        v = [1, 0, 0]
        mag = cell.mag(v)
        assert abs(mag - 5.0) < 1e-10

        v = [1, 1, 1]
        mag = cell.mag(v)
        expected = 5.0 * np.sqrt(3)
        assert abs(mag - expected) < 1e-10

    def test_vector_angle(self):
        """Test angle calculations between vectors."""
        cell = Lattice(5.0, 5.0, 5.0, 90, 90, 90)
        u = [1, 0, 0]
        v = [0, 1, 0]
        angle = cell.angle(u, v)
        assert abs(angle - 90.0) < 1e-10

        u = [1, 1, 0]
        v = [1, 0, 0]
        angle = cell.angle(u, v)
        assert abs(angle - 45.0) < 1e-10

    def test_real_recip_angle(self):
        """Test angle between real and reciprocal vectors."""
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)
        real_vec = [0, 0, 1]
        recip_vec = [0, 0, 1]
        angle = cell.angle_rr(real_vec, recip_vec)
        assert abs(angle - 0.0) < 1e-5


class TestDiffraction:
    """Test d-spacing and diffraction angle calculations."""

    def test_d_spacing_cubic(self):
        """Test d-spacing calculations for cubic lattice."""
        cell = Lattice(5.0, 5.0, 5.0, 90, 90, 90)
        d001 = cell.d([0, 0, 1])
        assert abs(d001 - 5.0) < 1e-10

        d100 = cell.d([1, 0, 0])
        assert abs(d100 - 5.0) < 1e-10

        d111 = cell.d([1, 1, 1])
        expected = 5.0 / np.sqrt(3)
        assert abs(d111 - expected) < 1e-10

    def test_d_spacing_hexagonal(self):
        """Test d-spacing for hexagonal lattice."""
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)
        d001 = cell.d([0, 0, 1])
        assert abs(d001 - 13.0) < 1e-10

    def test_two_theta_calculation(self):
        """Test 2θ diffraction angle calculations."""
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)
        tth = cell.tth([0, 0, 1], lam=1.0)
        expected = 2 * np.degrees(np.arcsin(1.0 / (2 * 13.0)))
        assert abs(tth - expected) < 1e-10

    def test_dvec_calculation(self):
        """Test real space d-vector calculation."""
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)
        dv = cell.dvec([0, 0, 1])
        # Check magnitude equals d-spacing
        mag = cell.mag(dv)
        d_spacing = cell.d([0, 0, 1])
        assert abs(mag - d_spacing) < 1e-10

    def test_error_cases(self):
        """Test error handling for invalid inputs."""
        cell = Lattice()
        # Test invalid hkl length
        d = cell.d([1, 0])
        assert d == 0.0

        # Test zero magnitude hkl
        d = cell.d([0, 0, 0])
        assert d == 0.0


class TestCoordinateTransforms:
    """Test coordinate transformations between real and reciprocal space."""

    def test_recip_to_real_transform(self):
        """Test reciprocal to real space transformation."""
        cell = Lattice(5.0, 5.0, 5.0, 90, 90, 90)
        hkl = [1, 0, 0]
        real_vec = cell.recip_to_real(hkl)
        expected = np.array([0.04, 0.0, 0.0])
        np.testing.assert_array_almost_equal(real_vec, expected, decimal=10)

    def test_real_to_recip_transform(self):
        """Test real to reciprocal space transformation."""
        cell = Lattice(5.0, 5.0, 5.0, 90, 90, 90)
        real_vec = [1, 0, 0]
        hkl = cell.real_to_recip(real_vec)
        expected = np.array([25.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(hkl, expected)

    def test_roundtrip_transforms(self):
        """Test roundtrip transformations preserve vectors."""
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)
        original_hkl = [1, 2, 3]
        real_vec = cell.recip_to_real(original_hkl)
        recovered_hkl = cell.real_to_recip(real_vec)

        # Should recover original up to scaling
        scaling = recovered_hkl[0] / original_hkl[0]
        expected = np.array(original_hkl) * scaling
        np.testing.assert_array_almost_equal(recovered_hkl, expected, decimal=10)


class TestLatticeTransform:
    """Test LatticeTransform class functionality."""

    def test_transform_initialization(self):
        """Test lattice transform initialization."""
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)
        transform = LatticeTransform(cell)
        # Default should be identity transform
        np.testing.assert_array_almost_equal(transform.Va, [1, 0, 0])
        np.testing.assert_array_almost_equal(transform.Vb, [0, 1, 0])
        np.testing.assert_array_almost_equal(transform.Vc, [0, 0, 1])

    def test_rhombohedral_transform(self):
        """Test rhombohedral transformation from original test."""
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)
        Va_rhom = [0.6667, 0.3333, 0.3333]
        Vb_rhom = [-0.3333, 0.3333, 0.3333]
        Vc_rhom = [-0.3333, -0.6667, 0.3333]
        t = LatticeTransform(cell, Va=Va_rhom, Vb=Vb_rhom, Vc=Vc_rhom)

        # Test forward and reverse hkl transforms
        hkl_hex = [0, 0, 1]
        hkl_rhom = t.hp(hkl_hex)
        recovered_hkl = t.h(hkl_rhom)
        np.testing.assert_array_almost_equal(recovered_hkl, hkl_hex, decimal=10)

    def test_cartesian_transform(self):
        """Test cartesian coordinate transformation."""
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)
        t = LatticeTransform(cell)
        t.cartesian()
        cart = t.plat()

        # Test vector magnitude preservation
        hex_vec = [1, 1, 0]
        cart_vec = t.xp(hex_vec)
        hex_mag = cell.mag(hex_vec)
        cart_mag = cart.mag(cart_vec)
        assert abs(hex_mag - cart_mag) < 1e-10

    def test_vector_transforms(self):
        """Test vector transformations between bases."""
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)
        Va = [1, 0, 0]
        Vb = [0, 1, 0]
        Vc = [0, 0, 2]
        t = LatticeTransform(cell, Va=Va, Vb=Vb, Vc=Vc)

        # Test roundtrip
        original = [1, 2, 3]
        transformed = t.xp(original)
        recovered = t.x(transformed)
        np.testing.assert_array_almost_equal(recovered, original, decimal=10)

    def test_shift_transform(self):
        """Test transformations with origin shift."""
        cell = Lattice(5.0, 5.0, 5.0, 90, 90, 90)
        shift = [0.5, 0.5, 0.5]
        t = LatticeTransform(cell, shift=shift)

        # Test shift is applied correctly
        original = [1, 1, 1]
        transformed = t.xp(original)
        expected = np.array([0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(transformed, expected)


class TestOriginalTestFunctions:
    """Reproduce the exact behavior of original Python 2 test functions."""

    def test_original_test_lattice(self):
        """Test equivalent to original test_lattice() function."""
        # Create a new lattice instance
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)

        # Test lattice parameters and volumes
        cell_str = str(cell)
        assert "a=5.60000" in cell_str
        assert "gamma=120.00000" in cell_str

        vol_real = cell.vol()
        vol_recip = cell.vol(recip=True)
        assert vol_real > 0
        assert vol_recip > 0
        assert abs(vol_real * vol_recip - 1.0) < 1e-10

        # Test d-spacing and 2θ
        d001 = cell.d([0, 0, 1])
        assert abs(d001 - 13.0) < 1e-10

        tth = cell.tth([0, 0, 1], lam=1.0)
        expected_tth = 2 * np.degrees(np.arcsin(1.0 / (2 * 13.0)))
        assert abs(tth - expected_tth) < 1e-10

        # Test dvec calculation
        dv = cell.dvec([0, 0, 1])
        dv_mag = cell.mag(dv)
        assert abs(dv_mag - d001) < 1e-10

        # Test angles between vectors
        # Reciprocal lattice angles
        angle_recip_100_010 = cell.angle([1, 0, 0], [0, 1, 0], recip=True)
        angle_recip_100_001 = cell.angle([1, 0, 0], [0, 0, 1], recip=True)
        angle_recip_001_111 = cell.angle([0, 0, 1], [1, 1, 1], recip=True)

        # Real lattice angles
        angle_real_100_010 = cell.angle([1, 0, 0], [0, 1, 0])
        angle_real_100_001 = cell.angle([1, 0, 0], [0, 0, 1])
        angle_real_001_111 = cell.angle([0, 0, 1], [1, 1, 1])

        # Real-reciprocal angles
        angle_rr_001_001 = cell.angle_rr([0, 0, 1], [0, 0, 1])
        angle_rr_111_001 = cell.angle_rr([1, 1, 1], [0, 0, 1])
        angle_rr_dv_001 = cell.angle_rr(dv, [0, 0, 1])
        angle_rr_dv_110 = cell.angle_rr(dv, [1, 1, 0])

        # All angles should be reasonable values
        for angle in [
            angle_recip_100_010,
            angle_recip_100_001,
            angle_recip_001_111,
            angle_real_100_010,
            angle_real_100_001,
            angle_real_001_111,
            angle_rr_001_001,
            angle_rr_111_001,
            angle_rr_dv_001,
            angle_rr_dv_110,
        ]:
            assert 0 <= angle <= 180

    def test_original_test_transform(self):
        """Test equivalent to original test_transform() function."""
        # Create a new hexagonal lattice instance
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)

        # Test lattice transform
        Va_rhom = [0.6667, 0.3333, 0.3333]
        Vb_rhom = [-0.3333, 0.3333, 0.3333]
        Vc_rhom = [-0.3333, -0.6667, 0.3333]
        t = LatticeTransform(cell, Va=Va_rhom, Vb=Vb_rhom, Vc=Vc_rhom)

        # Create an instance of the rhombohedral lattice
        rhomb_lattice = t.plat()
        assert isinstance(rhomb_lattice, Lattice)

        # Test hkl transformations
        hkl_rhom = t.hp([0, 0, 1])
        assert len(hkl_rhom) == 3

        hkl_hex = t.h([1, 1, 1])
        assert len(hkl_hex) == 3

        # Test cartesian representation
        t.cartesian()
        cart = t.plat()
        assert isinstance(cart, Lattice)

        # Convert a [1,1,0] vector in hex lattice to cartesian
        vc = t.xp([1, 1, 0])
        assert len(vc) == 3

        # Check that the vector is the same length
        hex_length = cell.mag([1, 1, 0])
        cart_length = cart.mag(vc)
        assert abs(hex_length - cart_length) < 1e-10


class TestPython2Compatibility:
    """Test specific Python 2/3 compatibility issues."""

    def test_division_consistency(self):
        """Test that division operations are consistent between Python versions."""
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)

        # Test d-spacing calculation (1/magnitude)
        d = cell.d([1, 0, 0])
        mag = cell.mag([1, 0, 0], recip=True)
        assert abs(d - 1.0 / mag) < 1e-15

    def test_type_consistency(self):
        """Test that data types are consistent between Python versions."""
        cell = Lattice(5, 6, 7)  # Integer inputs
        assert isinstance(cell.a, float)
        assert isinstance(cell.b, float)
        assert isinstance(cell.c, float)

        # Test array operations return correct types
        vol = cell.vol()
        assert isinstance(vol, (float, np.floating))


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_lattice_parameters(self):
        """Test behavior with zero lattice parameters."""
        with pytest.raises((ZeroDivisionError, np.linalg.LinAlgError)):
            cell = Lattice(0, 1, 1)
            # Trigger metric tensor calculation that should fail
            _ = cell.vol()

    def test_negative_lattice_parameters(self):
        """Test behavior with negative lattice parameters."""
        cell = Lattice(-5, 5, 5)  # Should take absolute value or handle gracefully
        assert cell.a == -5.0  # Implementation choice - may want to change

    def test_extreme_angles(self):
        """Test with extreme angle values that are still physically valid."""
        # Very small angles that are still physically meaningful
        cell = Lattice(1, 1, 1, 10.0, 10.0, 10.0)
        vol = cell.vol()
        assert vol >= 0

        # Small angles
        cell = Lattice(1, 1, 1, 30.0, 30.0, 30.0)
        vol = cell.vol()
        assert vol >= 0

        # Test with realistic extreme crystallographic angles
        cell = Lattice(5, 6, 7, 70.0, 80.0, 85.0)
        vol = cell.vol()
        assert vol >= 0

        # Triclinic with obtuse angles
        cell = Lattice(5, 6, 7, 100.0, 110.0, 120.0)
        vol = cell.vol()
        assert vol >= 0

        # Hexagonal lattice with 120° gamma
        cell = Lattice(5, 5, 10, 90.0, 90.0, 120.0)
        vol = cell.vol()
        assert vol >= 0

    def test_wavelength_edge_cases(self):
        """Test extreme wavelength values."""
        cell = Lattice()

        # Very small wavelength
        tth = cell.tth([1, 0, 0], lam=1e-10)
        assert 0 <= tth <= 180

        # Very large wavelength
        tth = cell.tth([1, 0, 0], lam=1e10)
        assert 0 <= tth <= 180

    def test_large_miller_indices(self):
        """Test with large Miller indices."""
        cell = Lattice()
        hkl = [100, 100, 100]
        d = cell.d(hkl)
        assert d > 0
        assert d < 1.0


class TestNumericalPrecision:
    """Test numerical precision and stability."""

    def test_metric_tensor_symmetry(self):
        """Test that metric tensor is symmetric."""
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)
        g = cell.g
        np.testing.assert_array_almost_equal(g, g.T, decimal=15)

    def test_reciprocal_lattice_consistency(self):
        """Test reciprocal lattice calculations are consistent."""
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)

        # Test that g * gr = I
        identity = np.dot(cell.g, cell.gr)
        np.testing.assert_array_almost_equal(identity, np.eye(3), decimal=12)

    def test_vector_operations_precision(self):
        """Test precision of vector operations."""
        cell = Lattice(np.pi, np.e, np.sqrt(2), 90, 90, 90)

        v = [1, 1, 1]
        dot_self = cell.dot(v, v)
        mag_squared = cell.mag(v) ** 2
        assert abs(dot_self - mag_squared) < 1e-14

    def test_angle_calculation_precision(self):
        """Test precision of angle calculations."""
        cell = Lattice()

        # Test orthogonal vectors
        angle = cell.angle([1, 0, 0], [0, 1, 0])
        assert abs(angle - 90.0) < 1e-12

        # Test parallel vectors
        angle = cell.angle([1, 1, 1], [2, 2, 2])
        assert abs(angle) < 2e-6


class TestMemoryAndPerformance:
    """Test memory usage and performance considerations."""

    def test_large_vector_operations(self):
        """Test operations with large vectors (stress test)."""
        cell = Lattice()

        # Large vector
        v = [1000, 1000, 1000]
        mag = cell.mag(v)
        assert mag > 0
        assert np.isfinite(mag)

    def test_repeated_calculations(self):
        """Test that repeated calculations are consistent."""
        cell = Lattice(5.6, 5.6, 13, 90, 90, 120)

        # Calculate same thing multiple times
        results = []
        for _ in range(100):
            d = cell.d([1, 1, 0])
            results.append(d)

        # All results should be identical
        assert all(abs(r - results[0]) < 1e-15 for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
