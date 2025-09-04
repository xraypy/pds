import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from pds.utils.ctr_data import CtrCorrectionPsic, _get_corr, _update_psic_angles, image_point_F


# Mock classes/functions if required
class MockGoniometer:
    """Mock goniometer for testing."""

    def __init__(self):
        self.pangles = {"alpha": 25.0, "beta": 30.0, "sigma_az": 45.0, "tau_az": 90.0}
        self.angles = {"delta": 45.0, "nu": 30.0, "phi": 10.0, "chi": 20.0, "eta": 5.0, "mu": 15.0}
        self.calc_psuedo = False
        self.ki = np.array([0.0, 1.0, 0.0])
        self.kr = np.array([0.0, 0.8, 0.2])
        self.nm = np.array([0.0, 0.0, 1.0])

    def set_angles(self, phi=None, chi=None, eta=None, mu=None, nu=None, delta=None):
        """Set goniometer angles."""
        if phi is not None:
            self.angles["phi"] = phi
        if chi is not None:
            self.angles["chi"] = chi
            self.pangles["alpha"] = chi  # Simple mock relationship
        if eta is not None:
            self.angles["eta"] = eta
            self.pangles["beta"] = eta  # Simple mock relationship
        if mu is not None:
            self.angles["mu"] = mu
        if nu is not None:
            self.angles["nu"] = nu
        if delta is not None:
            self.angles["delta"] = delta

    def _update_psuedo(self):
        """Mock pseudo angle update."""
        pass


class MockCtrCorrectionPsic:
    """Mock correction class for testing."""

    def __init__(self, gonio=None, beam_slits={}, det_slits=None, sample={}):
        self.gonio = gonio or MockGoniometer()
        self.beam_slits = beam_slits or {}
        self.det_slits = det_slits
        self.sample = sample or {}
        self.fh = 1.0

    def ctot_stationary(self, plot=False, fig=None):
        """Mock total correction factor."""
        return 1.5

    def polarization(self):
        """Mock polarization correction."""
        return 1.1

    def lorentz_stationary(self):
        """Mock Lorentz correction."""
        return 1.2

    def active_area(self, plot=False, fig=None):
        """Mock active area correction."""
        if not self.beam_slits:
            return 1.0
        return 1.3


# Fixtures
@pytest.fixture
def simple_scan_data():
    """Basic scan data for testing."""
    return {
        "I": [100.0, 200.0, 300.0],
        "io": [1000.0, 1100.0, 1200.0],
        "Ierr": [10.0, 15.0, 20.0],
        "Ibgr": [5.0, 8.0, 12.0],
        "transm": [0.9, 0.85, 0.8],
        "G": "mock",
        "phi": [0.0, 0.1, 0.2],
        "chi": [0.0, 0.1, 0.2],
        "eta": [0.0, 0.1, 0.2],
        "mu": [0.0, 0.1, 0.2],
        "nu": [0.0, 0.1, 0.2],
        "del": [0.0, 0.1, 0.2],
        "dims": (3, 1),
        "name": "test_scan",
    }


@pytest.fixture
def scan_no_gonio():
    """Scan data without goniometer for testing."""
    return {
        "I": [150.0, 250.0, 350.0],
        "io": [1200.0, 1300.0, 1400.0],
        "Ierr": [12.0, 18.0, 22.0],
        "Ibgr": [6.0, 9.0, 13.0],
        "transm": [0.95, 0.90, 0.85],
    }


@pytest.fixture
def numpy_scan_data():
    """Scan data with numpy arrays."""
    return {
        "I": np.array([110.0, 210.0, 310.0, 410.0]),
        "io": np.array([1100.0, 1200.0, 1300.0, 1400.0]),
        "Ierr": np.array([11.0, 16.0, 21.0, 26.0]),
        "Ibgr": np.array([5.5, 8.5, 12.5, 16.5]),
        "transm": np.array([0.92, 0.88, 0.84, 0.80]),
        "G": "mock_gonio",
        "phi": np.array([0.0, 0.05, 0.1, 0.15]),
        "chi": np.array([0.0, 0.05, 0.1, 0.15]),
        "eta": np.array([0.0, 0.05, 0.1, 0.15]),
        "mu": np.array([0.0, 0.05, 0.1, 0.15]),
        "nu": np.array([0.0, 0.05, 0.1, 0.15]),
        "del": np.array([0.0, 0.05, 0.1, 0.15]),
        "dims": (4, 1),
        "name": b"numpy_scan",  # bytes name for testing
    }


@pytest.fixture
def mock_gonio():
    """Mock goniometer for testing."""
    return MockGoniometer()


@pytest.fixture
def correction_params():
    """Standard correction parameters."""
    return {"geom": "psic", "scale": 1.5, "beam_slits": {"horz": 2.0, "vert": 1.5}, "det_slits": {"horz": 3.0, "vert": 2.5}, "sample": {"dia": 10.0}}


# Tests
class TestImagePointF:
    def test_basic_functionality(self, simple_scan_data, monkeypatch):
        """Test basic computation without corrections."""
        corr_params = {}

        # Mock the correction to return None
        monkeypatch.setattr("pds.utils.ctr_data._get_corr", lambda *args, **kwargs: None)

        result = image_point_F(simple_scan_data, 1, corr_params=corr_params)
        assert isinstance(result["F"], float)
        assert isinstance(result["Ferr"], float)

    def test_with_corrections(self, simple_scan_data, monkeypatch):
        """Test computation with mock corrections."""
        corr_params = {"geom": "psic"}

        # Mock the correction to return a mocked object
        monkeypatch.setattr("pds.utils.ctr_data._get_corr", lambda *args, **kwargs: MockCtrCorrectionPsic())

        result = image_point_F(simple_scan_data, 1, corr_params=corr_params)
        assert "ctot" in result
        assert "alpha" in result
        assert "beta" in result

    def test_invalid_index(self, simple_scan_data, monkeypatch):
        """Test behavior with an invalid index."""
        # Mock to avoid goniometer parsing issues
        monkeypatch.setattr("pds.utils.ctr_data._get_corr", lambda *args, **kwargs: None)

        corr_params = {}
        result = image_point_F(simple_scan_data, 10, corr_params=corr_params)
        # When index is out of bounds, extract_value falls back to first element
        # F = sqrt(1.0 / 0.9 * 1.0 * 100.0 / 1000.0) = sqrt(0.1111...) ≈ 0.333...
        assert abs(result["F"] - 1 / 3) < 0.01  # approximately 0.333...
        assert result["Ferr"] > 0  # Some positive error value

    def test_scalar_values(self, monkeypatch):
        """Test with scalar input values."""
        # Mock to avoid goniometer parsing issues
        monkeypatch.setattr("pds.utils.ctr_data._get_corr", lambda *args, **kwargs: None)

        scan_data_scalar = {
            "I": 150.0,
            "io": 1050.0,
            "Ierr": 12.0,
            "Ibgr": 7.0,
            "transm": 0.88,
        }

        result = image_point_F(scan_data_scalar, 0)
        assert isinstance(result["F"], float)
        assert isinstance(result["Ferr"], float)


class TestGetCorr:
    """Test _get_corr function."""

    def test_no_goniometer_data(self, simple_scan_data):
        """Test that _get_corr returns None when no goniometer data is available."""
        scan = {**simple_scan_data}
        del scan["G"]
        result = _get_corr(scan, 1, {}, False)
        assert result is None

    def test_psic_geometry(self, simple_scan_data, monkeypatch):
        """Test that _get_corr successfully returns a CtrCorrectionPsic object for psic geometry."""

        def mock_psic_from_spec(*args, **kwargs):
            return MockGoniometer()

        monkeypatch.setattr("pds.utils.ctr_data.psic_from_spec", mock_psic_from_spec)

        scan = simple_scan_data
        corr_params = {"geom": "psic"}
        result = _get_corr(scan, 1, corr_params, False)
        assert isinstance(result, CtrCorrectionPsic)

    def test_bytes_geometry(self, simple_scan_data, monkeypatch):
        """Test geometry specified as bytes."""

        def mock_psic_from_spec(*args, **kwargs):
            return MockGoniometer()

        monkeypatch.setattr("pds.utils.ctr_data.psic_from_spec", mock_psic_from_spec)

        scan = simple_scan_data
        corr_params = {"geom": b"psic"}
        result = _get_corr(scan, 1, corr_params, False)
        assert isinstance(result, CtrCorrectionPsic)

    def test_invalid_geometry(self, simple_scan_data):
        """Test that _get_corr raises NotImplementedError for unsupported geometry."""
        scan = simple_scan_data
        corr_params = {"geom": "invalid_geom"}
        with pytest.raises(NotImplementedError):
            _get_corr(scan, 1, corr_params, False)

    def test_with_correction_parameters(self, simple_scan_data, monkeypatch, correction_params):
        """Test _get_corr with full correction parameters."""

        def mock_psic_from_spec(*args, **kwargs):
            return MockGoniometer()

        monkeypatch.setattr("pds.utils.ctr_data.psic_from_spec", mock_psic_from_spec)

        result = _get_corr(simple_scan_data, 1, correction_params, False)
        assert isinstance(result, CtrCorrectionPsic)
        assert result.beam_slits == correction_params["beam_slits"]
        assert result.det_slits == correction_params["det_slits"]
        assert result.sample == correction_params["sample"]


class TestUpdatePsicAngles:
    """Test _update_psic_angles function."""

    def test_basic_angle_update(self, simple_scan_data, mock_gonio):
        """Test basic angle updating functionality."""
        _update_psic_angles(mock_gonio, simple_scan_data, 1, verbose=False)
        # Check that angles were set (mock implementation)
        assert mock_gonio.angles["phi"] == 0.1
        assert mock_gonio.angles["chi"] == 0.1
        assert mock_gonio.angles["eta"] == 0.1

    def test_scalar_angles(self, mock_gonio):
        """Test angle updating with scalar values."""
        scan_data = {"phi": 1.5, "chi": 2.0, "eta": 2.5, "mu": 3.0, "nu": 3.5, "del": 4.0, "dims": (1, 0), "name": "scalar_test"}
        _update_psic_angles(mock_gonio, scan_data, 0, verbose=False)
        assert mock_gonio.angles["phi"] == 1.5
        assert mock_gonio.angles["chi"] == 2.0
        assert mock_gonio.angles["eta"] == 2.5

    def test_missing_angles(self, mock_gonio, capsys):
        """Test behavior with missing angle data."""
        scan_data = {"dims": (1, 0), "name": "missing_test"}
        _update_psic_angles(mock_gonio, scan_data, 0, verbose=True)

        # Should default to 0.0 for missing angles
        assert mock_gonio.angles["phi"] == 0.0
        assert mock_gonio.angles["chi"] == 0.0

        # Check error messages were printed
        captured = capsys.readouterr()
        assert "Error getting phi angle" in captured.out

    def test_numpy_arrays(self, numpy_scan_data, mock_gonio):
        """Test angle updating with numpy arrays."""
        _update_psic_angles(mock_gonio, numpy_scan_data, 2, verbose=False)
        assert mock_gonio.angles["phi"] == 0.1
        assert mock_gonio.angles["chi"] == 0.1
        assert mock_gonio.angles["eta"] == 0.1

    def test_bytes_scan_name(self, numpy_scan_data, mock_gonio):
        """Test handling of bytes scan names."""
        _update_psic_angles(mock_gonio, numpy_scan_data, 1, verbose=False)
        # Should handle bytes name conversion without error
        assert True  # If we get here, bytes_to_str worked

    def test_beta_correction(self, simple_scan_data, mock_gonio, capsys):
        """Test beta angle correction when negative."""
        # Set up gonio with negative beta and override set_angles to maintain it
        mock_gonio.pangles["beta"] = -5.0

        # Override set_angles to not change beta
        original_set_angles = mock_gonio.set_angles

        def mock_set_angles_keeping_beta(*args, **kwargs):
            original_set_angles(*args, **kwargs)
            mock_gonio.pangles["beta"] = -5.0  # Keep beta negative

        mock_gonio.set_angles = mock_set_angles_keeping_beta

        _update_psic_angles(mock_gonio, simple_scan_data, 0, verbose=True)

        # Should be corrected to 0.0
        assert mock_gonio.pangles["beta"] == 0.0

        # Check warning was printed
        captured = capsys.readouterr()
        assert "Warning: beta is less than 0.0, setting to 0.0" in captured.out


class TestCtrCorrectionPsic:
    """Test CtrCorrectionPsic class."""

    def test_initialization(self, mock_gonio):
        """Test CtrCorrectionPsic initialization."""
        beam_slits = {"horz": 2.0, "vert": 1.5}
        det_slits = {"horz": 3.0, "vert": 2.5}
        sample = {"dia": 10.0}

        corr = CtrCorrectionPsic(gonio=mock_gonio, beam_slits=beam_slits, det_slits=det_slits, sample=sample)

        assert corr.gonio is mock_gonio
        assert corr.beam_slits == beam_slits
        assert corr.det_slits == det_slits
        assert corr.sample == sample
        assert corr.fh == 1.0

    def test_ctot_stationary(self, mock_gonio, capsys, monkeypatch):
        """Test ctot_stationary method."""

        def mock_active_area(*args, **kwargs):
            return (1.5, 1.0)  # A_beam, A_int

        def mock_beam_vectors(*args, **kwargs):
            return np.array([[0, 0, 0], [1, 0, 0]])

        monkeypatch.setattr("pds.utils.ctr_data.active_area", mock_active_area)
        monkeypatch.setattr("pds.utils.ctr_data.beam_vectors", mock_beam_vectors)

        corr = CtrCorrectionPsic(gonio=mock_gonio, beam_slits={"horz": 1.0, "vert": 1.0})
        result = corr.ctot_stationary(plot=False)
        assert isinstance(result, float)
        assert result > 0

    def test_lorentz_stationary(self, mock_gonio):
        """Test Lorentz correction calculation."""
        corr = CtrCorrectionPsic(gonio=mock_gonio)
        result = corr.lorentz_stationary()
        assert isinstance(result, float)
        # Should be sin(beta) where beta is the mock value
        expected = np.sin(np.radians(mock_gonio.pangles["beta"]))
        assert abs(result - expected) < 1e-10

    def test_polarization(self, mock_gonio):
        """Test polarization correction calculation."""
        corr = CtrCorrectionPsic(gonio=mock_gonio)
        result = corr.polarization()
        assert isinstance(result, float)
        assert result > 0

    def test_polarization_zero_case(self, mock_gonio):
        """Test polarization when p=0."""
        # Set angles to create p=0 condition
        mock_gonio.angles["delta"] = 0.0
        mock_gonio.angles["nu"] = 90.0
        corr = CtrCorrectionPsic(gonio=mock_gonio)
        result = corr.polarization()
        assert result == 0.0

    def test_active_area_no_beam_slits(self, mock_gonio, capsys):
        """Test active area with no beam slits."""
        corr = CtrCorrectionPsic(gonio=mock_gonio, beam_slits={})
        result = corr.active_area()
        assert result == 1.0

        captured = capsys.readouterr()
        assert "Warning beam slits not specified" in captured.out

    def test_active_area_negative_alpha(self, mock_gonio, capsys):
        """Test active area with negative alpha."""
        mock_gonio.pangles["alpha"] = -5.0
        corr = CtrCorrectionPsic(gonio=mock_gonio, beam_slits={"horz": 1.0, "vert": 1.0})
        result = corr.active_area()
        assert result == 0.0

        captured = capsys.readouterr()
        assert "alpha is less than 0.0" in captured.out

    def test_active_area_negative_beta(self, mock_gonio, capsys):
        """Test active area with negative beta."""
        mock_gonio.pangles["beta"] = -3.0
        corr = CtrCorrectionPsic(gonio=mock_gonio, beam_slits={"horz": 1.0, "vert": 1.0})
        result = corr.active_area()
        assert result == 0.0

        captured = capsys.readouterr()
        assert "beta is less than 0.0" in captured.out

    def test_active_area_with_det_slits(self, mock_gonio, monkeypatch):
        """Test active area calculation with detector slits."""

        def mock_active_area(*args, **kwargs):
            return (2.0, 1.0)  # A_beam, A_int

        def mock_beam_vectors(*args, **kwargs):
            return np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

        def mock_det_vectors(*args, **kwargs):
            return np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

        monkeypatch.setattr("pds.utils.ctr_data.active_area", mock_active_area)
        monkeypatch.setattr("pds.utils.ctr_data.beam_vectors", mock_beam_vectors)
        monkeypatch.setattr("pds.utils.ctr_data.det_vectors", mock_det_vectors)

        corr = CtrCorrectionPsic(gonio=mock_gonio, beam_slits={"horz": 1.0, "vert": 1.0}, det_slits={"horz": 2.0, "vert": 1.5})
        result = corr.active_area()
        assert result == 2.0  # A_beam / A_int^2 = 2.0 / 1.0^2

    def test_active_area_with_sample_dict(self, mock_gonio, monkeypatch):
        """Test active area with sample as dictionary."""

        def mock_active_area(*args, **kwargs):
            return (1.5, 1.0)

        def mock_beam_vectors(*args, **kwargs):
            return np.array([[0, 0, 0], [1, 0, 0]])

        def mock_sample_vectors(*args, **kwargs):
            return np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

        monkeypatch.setattr("pds.utils.ctr_data.active_area", mock_active_area)
        monkeypatch.setattr("pds.utils.ctr_data.beam_vectors", mock_beam_vectors)
        monkeypatch.setattr("pds.utils.ctr_data.sample_vectors", mock_sample_vectors)

        sample = {"polygon": [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], "angles": {}, "dia": 0.0}

        corr = CtrCorrectionPsic(gonio=mock_gonio, beam_slits={"horz": 1.0, "vert": 1.0}, sample=sample)
        result = corr.active_area()
        assert result == 1.5

    def test_active_area_with_sample_diameter(self, mock_gonio, monkeypatch):
        """Test active area with sample diameter."""

        def mock_active_area(*args, **kwargs):
            return (3.0, 2.0)

        def mock_beam_vectors(*args, **kwargs):
            return np.array([[0, 0, 0], [1, 0, 0]])

        monkeypatch.setattr("pds.utils.ctr_data.active_area", mock_active_area)
        monkeypatch.setattr("pds.utils.ctr_data.beam_vectors", mock_beam_vectors)

        sample = {"dia": 5.0}

        corr = CtrCorrectionPsic(gonio=mock_gonio, beam_slits={"horz": 1.0, "vert": 1.0}, sample=sample)
        result = corr.active_area()
        assert result == 0.75  # 3.0 / 2.0^2


class TestImagePointFComprehensive:
    """Additional comprehensive tests for image_point_F."""

    def test_numpy_arrays(self, numpy_scan_data, monkeypatch):
        """Test with numpy array inputs."""
        # Mock to avoid goniometer parsing issues
        monkeypatch.setattr("pds.utils.ctr_data._get_corr", lambda *args, **kwargs: None)

        result = image_point_F(numpy_scan_data, 2)
        assert isinstance(result["F"], float)
        assert isinstance(result["Ferr"], float)
        assert result["I"] == 310.0
        assert result["Inorm"] == 1300.0

    def test_zero_intensity(self):
        """Test with zero intensity values."""
        scan_data = {"I": [0.0, 0.0, 0.0], "io": [1000.0, 1100.0, 1200.0], "Ierr": [0.0, 0.0, 0.0], "Ibgr": [0.0, 0.0, 0.0], "transm": [0.9, 0.85, 0.8]}
        result = image_point_F(scan_data, 1)
        assert result["F"] == 0.0
        assert result["Ferr"] == 0.0

    def test_zero_normalization(self):
        """Test with zero normalization values."""
        scan_data = {"I": [100.0, 200.0, 300.0], "io": [0.0, 0.0, 0.0], "Ierr": [10.0, 15.0, 20.0], "Ibgr": [5.0, 8.0, 12.0], "transm": [0.9, 0.85, 0.8]}
        result = image_point_F(scan_data, 1)
        assert result["F"] == 0.0
        assert result["Ferr"] == 0.0

    def test_custom_column_names(self, simple_scan_data):
        """Test with custom column names."""
        scan_data = {
            "intensity": [100.0, 200.0, 300.0],
            "normalization": [1000.0, 1100.0, 1200.0],
            "error": [10.0, 15.0, 20.0],
            "background": [5.0, 8.0, 12.0],
            "transmission": [0.9, 0.85, 0.8],
        }
        result = image_point_F(scan_data, 1, Iitg="intensity", Inorm="normalization", Ierr="error", Ibgr="background", transm="transmission")
        assert isinstance(result["F"], float)
        assert result["I"] == 200.0
        assert result["Inorm"] == 1100.0

    def test_with_scale_factor(self, simple_scan_data, monkeypatch):
        """Test with scale factor in correction parameters."""
        monkeypatch.setattr("pds.utils.ctr_data._get_corr", lambda *args, **kwargs: None)

        corr_params = {"scale": 2.0}
        result = image_point_F(simple_scan_data, 1, corr_params=corr_params)

        # F should be calculated with scale factor
        assert isinstance(result["F"], float)
        assert result["F"] > 0

    def test_return_dict_structure(self, simple_scan_data, monkeypatch):
        """Test that return dictionary has correct structure."""
        # Mock to avoid goniometer parsing issues
        monkeypatch.setattr("pds.utils.ctr_data._get_corr", lambda *args, **kwargs: None)

        result = image_point_F(simple_scan_data, 1)

        expected_keys = {"I", "Inorm", "Ierr", "Ibgr", "transm", "F", "Ferr", "ctot", "alpha", "beta"}
        assert set(result.keys()) == expected_keys

        # All values should be floats
        for key, value in result.items():
            assert isinstance(value, float)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_missing_required_columns(self):
        """Test behavior with missing required columns."""
        incomplete_scan = {"I": [100.0, 200.0]}

        with pytest.raises(KeyError):
            image_point_F(incomplete_scan, 0)

    def test_mixed_data_types(self):
        """Test with mixed data types in arrays."""
        scan_data = {
            "I": [100, 200.0, np.float32(300.0)],  # Mixed int, float, numpy
            "io": np.array([1000, 1100, 1200]),
            "Ierr": [10.0, 15.0, 20.0],
            "Ibgr": [5.0, 8.0, 12.0],
            "transm": [0.9, 0.85, 0.8],
        }
        result = image_point_F(scan_data, 1)
        assert isinstance(result["I"], float)
        assert result["I"] == 200.0

    def test_very_large_values(self):
        """Test with very large numerical values."""
        scan_data = {"I": [1e10, 2e10, 3e10], "io": [1e12, 1.1e12, 1.2e12], "Ierr": [1e8, 1.5e8, 2e8], "Ibgr": [1e7, 1.2e7, 1.5e7], "transm": [0.9, 0.85, 0.8]}
        result = image_point_F(scan_data, 1)
        assert isinstance(result["F"], float)
        assert np.isfinite(result["F"])
        assert np.isfinite(result["Ferr"])

    def test_very_small_values(self):
        """Test with very small numerical values."""
        scan_data = {
            "I": [1e-10, 2e-10, 3e-10],
            "io": [1e-8, 1.1e-8, 1.2e-8],
            "Ierr": [1e-12, 1.5e-12, 2e-12],
            "Ibgr": [1e-13, 1.2e-13, 1.5e-13],
            "transm": [0.9, 0.85, 0.8],
        }
        result = image_point_F(scan_data, 1)
        assert isinstance(result["F"], float)
        assert np.isfinite(result["F"])
        assert np.isfinite(result["Ferr"])


class TestPython2Compatibility:
    """Test Python 2/3 compatibility features."""

    def test_bytes_string_handling(self, monkeypatch):
        """Test proper handling of bytes vs strings."""

        def mock_psic_from_spec(*args, **kwargs):
            return MockGoniometer()

        monkeypatch.setattr("pds.utils.ctr_data.psic_from_spec", mock_psic_from_spec)

        scan_data = {
            "I": [100.0, 200.0, 300.0],
            "io": [1000.0, 1100.0, 1200.0],
            "Ierr": [10.0, 15.0, 20.0],
            "Ibgr": [5.0, 8.0, 12.0],
            "transm": [0.9, 0.85, 0.8],
            "G": "mock",
            "phi": [0.0, 0.1, 0.2],
            "chi": [0.0, 0.1, 0.2],
            "eta": [0.0, 0.1, 0.2],
            "mu": [0.0, 0.1, 0.2],
            "nu": [0.0, 0.1, 0.2],
            "del": [0.0, 0.1, 0.2],
            "dims": (3, 1),
            "name": b"bytes_scan_name",  # bytes name
        }

        # Should handle bytes geometry specification
        corr_params = {"geom": b"psic"}
        result = _get_corr(scan_data, 1, corr_params, False)
        assert isinstance(result, CtrCorrectionPsic)

    def test_integer_division_compatibility(self):
        """Test that calculations work correctly with Python 3 division."""
        scan_data = {
            "I": [100.0, 200.0, 300.0],
            "io": [1000.0, 1100.0, 1200.0],
            "Ierr": [10.0, 15.0, 20.0],
            "Ibgr": [5.0, 8.0, 12.0],
            "transm": [0.9, 0.85, 0.8],
        }

        result = image_point_F(scan_data, 1)

        # Ensure results are floats, not integers
        assert isinstance(result["F"], float)
        assert isinstance(result["Ferr"], float)

        # Test division behavior
        test_division = 7 / 2
        assert test_division == 3.5
        assert isinstance(test_division, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
