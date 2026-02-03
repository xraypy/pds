#!/usr/bin/python
# ----------------------------------------------------------------------------------
# Project: pds
# File: tests/spec/test_spec_hdf5_converter.py
# ----------------------------------------------------------------------------------
# Purpose:
# Tests for pds.core.spec.hdf5_converter (SPEC to HDF5 conversion).
# ----------------------------------------------------------------------------------
# Author: Christofanis Skordas
#
# Copyright (C) 2025-2026 GSECARS, The University of Chicago, USA
# Copyright (C) 2025-2026 NSF SEES, USA
# ----------------------------------------------------------------------------------

import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pds.core.spec.hdf5_converter import Hdf5Converter
from tests.spec import minimal_spec_content


class TestHdf5Converter:
    """Test SPEC to HDF5 converter."""

    def test_run_writes_hdf5_structure(self):
        """Conversion creates spec group, scan group, point_data/point_labs/param_*."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / "sample.spec"
            spec_path.write_text(minimal_spec_content(), encoding="utf-8")
            out_path = Path(tmpdir) / "out.h5"

            converter = Hdf5Converter(spec_path, out_path, verbose=False)
            result = converter.run()
            assert result is True
            assert out_path.exists()

            with h5py.File(out_path, "r") as f:
                assert spec_path.name in f
                spec_group = f[spec_path.name]
                assert "1" in spec_group
                scan_group = spec_group["1"]
                assert "point_data" in scan_group
                assert "point_labs" in scan_group
                assert "param_labs" in scan_group
                assert "param_data" in scan_group
                assert scan_group.attrs["index"] == 1
                assert "ascan" in scan_group.attrs["cmd"]
                np.testing.assert_array_equal(scan_group["point_data"][()], np.array([[0.0, 100.0], [10.0, 200.0]]))

    def test_run_missing_spec_returns_false(self):
        """Missing SPEC file returns False and does not create output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / "nonexistent.spec"
            out_path = Path(tmpdir) / "out.h5"
            assert not spec_path.exists()

            converter = Hdf5Converter(spec_path, out_path, verbose=False)
            result = converter.run()
            assert result is False
            assert not out_path.exists()

    def test_run_idempotent_skips_complete_scan(self, capsys):
        """Second run with same data skips scan (already complete)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / "sample.spec"
            spec_path.write_text(minimal_spec_content(), encoding="utf-8")
            out_path = Path(tmpdir) / "out.h5"

            converter = Hdf5Converter(spec_path, out_path, verbose=True)
            converter.run()
            converter.run()
            out = capsys.readouterr()
            assert "Skipping" in out.out or "already complete" in out.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
