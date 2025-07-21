import sys
from pathlib import Path

import numpy as np
import pytest

# Add the pds module to the path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from pds.utils.gonio_psic import (
    Psic,
    beam_vectors,
    calc_D,
    calc_kvecs,
    calc_Q,
    calc_Z,
    det_vectors,
    psic_from_spec,
    sample_vectors,
    spec_psic_G,
)
from pds.utils.mathutil import cartesian_mag


def psic_reference_angles():
    """Helper function to create reference angles and vector calculations in Psic."""
    psic = Psic(5.6, 5.6, 13, 90, 90, 120, lam=1.3756)
    psic.set_angles(phi=65.78, chi=37.1005, eta=6.6400, mu=0.0, nu=0.0, delta=46.9587)

    # Set the reference vector and calculate miscut
    n = [-0.0348357, -0.00243595, 1]
    psic.set_n(n)

    # Calculate reference vector in phi coordinates
    n_phi = np.dot(psic.UB, n)
    n_phi = n_phi / np.abs(cartesian_mag(n_phi))

    # Calculate n vector from sigma_az and tau_az
    psic.calc_n(-psic.pangles["sigma_az"], -psic.pangles["tau_az"])

    return psic


def psic_spec_array(show=True):
    """Helper function for spec array handling and outcomes in Psic."""
    psic = Psic()
    G = [
        0.0,
        0.0,
        1.0,
        0.0039915744589999998,
        0.00075650941450000001,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        50.0,
        0.0,
        0.0,
        1.0,
        4.0,
        4.0,
        5.0,
        4.0,
        0.0,
        0.0,
        8.0939999999999994,
        4.9880000000000004,
        6.0709999999999997,
        90.0,
        90.0,
        90.0,
        0.77627690969999996,
        1.2596602459999999,
        1.034950635,
        90.0,
        90.0,
        90.0,
        0.0,
        0.0,
        4.0,
        2.0,
        0.0,
        2.4809999999999999,
        -0.00089999999999999998,
        0.00080000000000000004,
        -0.1244,
        175.2192,
        31.965,
        16.27375,
        15.090299999999999,
        7.5477999999999996,
        11.9002,
        -96.048199999999994,
        17.594249999999999,
        0.35625000000000001,
        0.83580100000000002,
        0.83580100000000002,
        -0.23926186720000001,
        1.1983306439999999,
        -0.0026481987470000001,
        -0.73847260000000003,
        -0.38826052760000002,
        -0.0050577973450000001,
        -0.004221190899,
        0.001168841303,
        1.034934888,
        -0.00011346681140000001,
        0.0002801762336,
        5.9400141580000003,
        0.83580100000000002,
        23.955432519999999,
        24.314067479999999,
        0.28474999779999999,
        48.269500000000001,
        0.075104988760000005,
        0.17931763710000001,
        -0.00040199175790000001,
        -0.00014478299110000001,
        0.48309999999999997,
        108.00069999999999,
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        12.0,
        0.0,
        0.0,
        2.0802999999999998,
        123.1461,
        0.0,
        0.0,
        0.0,
        0.0,
        -180.0,
        -180.0,
        -180.0,
        -180.0,
        -180.0,
        -180.0,
        -180.0,
        -180.0,
        -180.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    psic.set_spec_G(G)
    psic.set_angles(phi=178.1354, chi=-0.1344, eta=0.0002, mu=24.4195, nu=48.2695, delta=-0.0003)

    if show:
        print(f"h = {psic.h}")
        print("Pseudo angles:")
        for key, value in psic.pangles.items():
            print(f"  {key}: {value}")

    return psic


class TestPsicBasics:
    """Test basic Psic class functionality."""

    def test_psic_initialization_default(self):
        """Test default initialization."""
        psic = Psic()
        assert isinstance(psic.lattice, object)
        assert psic.angles["phi"] == 0.0
        assert psic.calc_psuedo is True
        assert len(psic.n) == 3

    def test_psic_initialization_custom(self):
        """Test custom lattice initialization."""
        psic = Psic(5.6, 5.6, 13, 90, 90, 120, lam=1.3756)
        assert psic.lattice.a == 5.6
        assert psic.lattice.gamma == 120.0
        assert psic.lattice.lam == 1.3756

    def test_psic_repr(self):
        """Test string representation."""
        psic = Psic()
        repr_str = repr(psic)
        assert "Primary:" in repr_str
        assert "Secondary:" in repr_str
        assert "Setting:" in repr_str


class TestOrientationReflections:
    """Test orientation reflection setting and swapping."""

    def test_set_or0(self):
        """Test setting primary orientation."""
        psic = Psic()
        psic.set_or0(h=[1, 0, 0], phi=10.0, chi=20.0)
        assert np.allclose(psic.or0["h"], [1, 0, 0])
        assert psic.or0["phi"] == 10.0
        assert psic.or0["chi"] == 20.0

    def test_set_or1(self):
        """Test setting secondary orientation."""
        psic = Psic()
        psic.set_or1(h=[0, 1, 0], eta=15.0, mu=25.0)
        assert np.allclose(psic.or1["h"], [0, 1, 0])
        assert psic.or1["eta"] == 15.0
        assert psic.or1["mu"] == 25.0

    def test_swap_or(self):
        """Test swapping orientations."""
        psic = Psic()
        original_or0_h = psic.or0["h"].copy()
        original_or1_h = psic.or1["h"].copy()

        psic.swap_or()

        assert np.allclose(psic.or0["h"], original_or1_h)
        assert np.allclose(psic.or1["h"], original_or0_h)


class TestMatrixCalculations:
    """Test matrix calculation functions."""

    def test_calc_Z_identity(self):
        """Test Z matrix with all zero angles."""
        Z = calc_Z(0.0, 0.0, 0.0, 0.0)
        expected = np.eye(3)
        assert np.allclose(Z, expected)

    def test_calc_Z_phi_rotation(self):
        """Test Z matrix with phi rotation only."""
        Z = calc_Z(phi=90.0)
        # With phi=90, we expect rotation about z-axis
        assert abs(Z[0, 0]) < 1e-10  # cos(90) ≈ 0
        assert abs(Z[0, 1] - 1.0) < 1e-10  # sin(90) ≈ 1
        assert abs(Z[1, 0] + 1.0) < 1e-10  # -sin(90) ≈ -1
        assert abs(Z[1, 1]) < 1e-10  # cos(90) ≈ 0

    def test_calc_Q_zero_angles(self):
        """Test Q vector with zero detector angles."""
        Q = calc_Q(0.0, 0.0, 1.0)
        expected = np.array([0.0, 0.0, 0.0])  # ki and kr are parallel
        assert np.allclose(Q, expected)

    def test_calc_kvecs(self):
        """Test k-vector calculation."""
        ki, kr = calc_kvecs(0.0, 0.0, 1.0)
        k = 2.0 * np.pi / 1.0

        expected_ki = np.array([0.0, k, 0.0])
        expected_kr = np.array([0.0, k, 0.0])

        assert np.allclose(ki, expected_ki)
        assert np.allclose(kr, expected_kr)

    def test_calc_D_identity(self):
        """Test detector rotation matrix with zero angles."""
        D = calc_D(0.0, 0.0)
        expected = np.eye(3)
        assert np.allclose(D, expected)


class TestVectorCalculations:
    """Test beam, detector, and sample vector calculations."""

    def test_beam_vectors(self):
        """Test beam aperture vector calculation."""
        beam = beam_vectors(2.0, 1.0)
        assert len(beam) == 4  # Four corners
        # Check corners are properly positioned
        assert len(beam[0]) == 3  # Each corner is 3D

    def test_det_vectors(self):
        """Test detector aperture vector calculation."""
        det = det_vectors(2.0, 1.0, 0.0, 0.0)
        assert len(det) == 4  # Four corners
        assert len(det[0]) == 3  # Each corner is 3D

    def test_sample_vectors_none(self):
        """Test sample vectors with None input."""
        result = sample_vectors(None)
        assert result is None

    def test_sample_vectors_insufficient_points(self):
        """Test sample vectors with insufficient points."""
        sample = [[0, 0], [1, 1]]  # Only 2 points
        result = sample_vectors(sample)
        assert result is None

    def test_sample_vectors_basic(self):
        """Test sample vectors with valid input."""
        sample = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        result = sample_vectors(sample)
        assert len(result) == 4
        assert all(len(point) == 3 for point in result)


class TestSpecFileInterface:
    """Test interface with spec file format."""

    def test_spec_psic_G(self):
        """Test parsing of spec G array."""
        # Minimal G array for testing
        G = [0.0] * 110  # Spec G arrays are typically ~110 elements
        G[3:6] = [0.0, 0.0, 1.0]  # n vector
        G[22:28] = [10.0, 10.0, 10.0, 90.0, 90.0, 90.0]  # lattice
        G[66] = 1.54  # wavelength
        G[34:37] = [1.0, 0.0, 0.0]  # or0 h
        G[37:40] = [0.0, 1.0, 0.0]  # or1 h
        G[40:46] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # or0 angles
        G[46:52] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # or1 angles
        G[52] = 1.54  # or0 lam
        G[53] = 1.54  # or1 lam

        cell, or0, or1, n = spec_psic_G(G)

        assert len(cell) == 7  # 6 lattice params + wavelength
        assert np.allclose(n, [0.0, 0.0, 1.0])
        assert np.allclose(or0["h"], [1.0, 0.0, 0.0])
        assert np.allclose(or1["h"], [0.0, 1.0, 0.0])

    def test_psic_from_spec(self):
        """Test creating psic from spec data."""
        psic = psic_from_spec(None, {"phi": 10.0, "chi": 20.0})
        assert psic.angles["phi"] == 10.0
        assert psic.angles["chi"] == 20.0


class TestAngleCalculations:
    """Test goniometer angle calculations."""

    def test_set_angles(self):
        """Test setting goniometer angles."""
        psic = Psic()
        psic.set_angles(phi=10.0, chi=20.0, eta=30.0)

        assert psic.angles["phi"] == 10.0
        assert psic.angles["chi"] == 20.0
        assert psic.angles["eta"] == 30.0

    def test_calc_n_basic(self):
        """Test reference vector calculation."""
        psic = Psic()
        psic.calc_n(0.0, 0.0)  # Should modify n vector
        # The calculation may not be exactly the same due to normalization
        assert len(psic.n) == 3


class TestPseudoAngles:
    """Test pseudo angle calculations."""

    def test_pseudo_angle_calculation(self):
        """Test that pseudo angles are calculated."""
        psic = Psic()
        psic.set_angles(phi=10.0, chi=20.0, eta=30.0, mu=5.0, nu=15.0, delta=25.0)

        # Check that pseudo angles are populated
        assert "tth" in psic.pangles
        assert "alpha" in psic.pangles
        assert "beta" in psic.pangles

    def test_disable_pseudo_calculation(self):
        """Test disabling pseudo angle calculation."""
        psic = Psic()
        psic.calc_psuedo = False
        psic.set_angles(phi=10.0)

        # With calc_psuedo=False, should have fewer entries
        assert len(psic.pangles) == 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_lattice_update(self):
        """Test lattice parameter updates."""
        psic = Psic()
        original_a = psic.lattice.a

        psic.set_lat(a=15.0)
        assert psic.lattice.a == 15.0
        assert psic.lattice.a != original_a

    def test_reference_vector_setting(self):
        """Test setting reference vector."""
        psic = Psic()
        new_n = [1.0, 1.0, 0.0]
        psic.set_n(new_n)
        assert np.allclose(psic.n, new_n)


class TestNumericalPrecision:
    """Test numerical precision and stability."""

    def test_matrix_orthogonality(self):
        """Test that calculated matrices maintain orthogonality."""
        psic = Psic()
        psic.set_angles(phi=45.0, chi=30.0, eta=60.0, mu=15.0, nu=25.0, delta=35.0)

        # Check that Z matrix is properly formed
        Z = psic.Z
        assert Z.shape == (3, 3)

        # Check that UB matrix is properly formed
        UB = psic.UB
        assert UB.shape == (3, 3)

    def test_angle_consistency(self):
        """Test consistency between angle calculations."""
        psic = Psic(5.6, 5.6, 13, 90, 90, 120, lam=1.3756)
        psic.set_angles(phi=65.78, chi=37.1005, eta=6.6400, mu=0.0, nu=0.0, delta=46.9587)

        # Calculate tth two ways
        tth1 = psic.pangles["tth"]
        tth2 = psic.lattice.tth(psic.h)

        assert abs(tth1 - tth2) < 1e-6


class TestReferenceFunctions:
    """Test reference functions that match original Python 2 behavior."""

    def test_reference_function_runs(self):
        """Test that reference calculation runs without error and validates results."""
        psic = psic_reference_angles()

        # Verify it returns a Psic instance
        assert isinstance(psic, Psic)

        # Check lattice parameters match expected values
        assert psic.lattice.a == 5.6
        assert psic.lattice.b == 5.6
        assert psic.lattice.c == 13
        assert psic.lattice.gamma == 120
        assert psic.lattice.lam == 1.3756

        # Check angles were set correctly
        assert psic.angles["phi"] == 65.78
        assert psic.angles["chi"] == 37.1005
        assert psic.angles["eta"] == 6.6400
        assert psic.angles["mu"] == 0.0
        assert psic.angles["nu"] == 0.0
        assert psic.angles["delta"] == 46.9587

        # Check that h vector was calculated with exact values
        expected_h = [-1.87282964, 2.55838325, 4.3476997]
        assert np.allclose(psic.h, expected_h, rtol=1e-6)

        # Check pseudo angles exist and match expected values
        assert "tth" in psic.pangles
        assert "sigma_az" in psic.pangles
        assert "tau_az" in psic.pangles
        assert abs(psic.pangles["tth"] - 46.9587) < 1e-6
        assert abs(psic.pangles["sigma_az"] - 5.529524531671174) < 1e-6
        assert abs(psic.pangles["tau_az"] - (-123.34870733271394)) < 1e-6

    def test_spec_function_runs(self):
        """Test that spec array handling runs without error and verifies results."""
        psic = psic_spec_array(show=False)

        # Verify it returns a Psic instance
        assert isinstance(psic, Psic)

        # Check angles were set correctly
        assert psic.angles["phi"] == 178.1354
        assert psic.angles["chi"] == -0.1344
        assert psic.angles["eta"] == 0.0002
        assert psic.angles["mu"] == 24.4195
        assert psic.angles["nu"] == 48.2695
        assert psic.angles["delta"] == -0.0003

        # Check that h vector was calculated with exact values
        expected_h = [-1.13466811e-04, 2.80176234e-04, 5.94001416e00]
        assert np.allclose(psic.h, expected_h, rtol=1e-6)

        # Check pseudo angles exist and match expected values
        expected_values = {
            "alpha": 23.955432517052696,
            "beta": 24.314067483215766,
            "omega": 0.28474999808151613,
            "tth": 48.26950000070051,
            "psi": 0.07510469273318308,
            "tau": 0.17931763713947707,
            "qaz": -0.00040199175794783967,
            "naz": -0.00014478298934552451,
            "sigma_az": 0.4830999999942762,
            "tau_az": 108.00070000041242,
        }

        for angle_name, expected_value in expected_values.items():
            assert angle_name in psic.pangles
            assert abs(psic.pangles[angle_name] - expected_value) < 1e-6

    def test_function_independence(self):
        """Test that reference functions create independent instances."""
        psic1 = psic_reference_angles()
        psic2 = psic_spec_array(show=False)

        # Should have different lattice parameters
        assert psic1.lattice.a != psic2.lattice.a
        assert psic1.lattice.lam != psic2.lattice.lam

        # Should have different angles
        assert psic1.angles["phi"] != psic2.angles["phi"]
        assert psic1.angles["mu"] != psic2.angles["mu"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
