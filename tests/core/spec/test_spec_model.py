#!/usr/bin/python
# ----------------------------------------------------------------------------------
# Project: pds
# File: tests/core/spec/test_spec_model.py
# ----------------------------------------------------------------------------------
# Purpose:
# Tests for pds.model.spec_model and MainModel.spec (SPEC model facade).
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
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pds.model.main_model import MainModel
from pds.model.spec_model import SpecModel


class TestSpecModel:
    """Test SPEC model (parser + converter facade)."""

    def test_summarize_returns_scan_list(self, minimal_spec_lines):
        """SpecModel.summarize(lines) returns list of scan dicts."""
        model = SpecModel()
        summary = model.summarize(minimal_spec_lines)
        assert len(summary) == 1
        assert summary[0]["index"] == 1
        assert summary[0]["labels"] == ["mu", "counts"]
        assert summary[0]["nl_dat"] == 2
        assert len(summary[0]["point_data"]) == 2

    def test_spec_to_hdf5_success(self, minimal_spec_content):
        """SpecModel.spec_to_hdf5 writes HDF5 and returns True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / "sample.spec"
            spec_path.write_text(minimal_spec_content, encoding="utf-8")
            out_path = Path(tmpdir) / "out.h5"

            model = SpecModel()
            result = model.spec_to_hdf5(spec_path, out_path, verbose=False)
            assert result is True
            assert out_path.exists()
            with h5py.File(out_path, "r") as f:
                assert spec_path.name in f
                assert "1" in f[spec_path.name]
                scan = f[spec_path.name]["1"]
                assert "point_data" in scan
                assert len(scan["point_data"]) == 2

    def test_spec_to_hdf5_missing_file_returns_false(self):
        """SpecModel.spec_to_hdf5 returns False when SPEC file does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / "nonexistent.spec"
            out_path = Path(tmpdir) / "out.h5"
            model = SpecModel()
            result = model.spec_to_hdf5(spec_path, out_path, verbose=False)
            assert result is False


class TestMainModelSpec:
    """Test MainModel exposes SpecModel."""

    def test_main_model_has_spec(self):
        """MainModel.spec is a SpecModel with summarize and spec_to_hdf5."""
        main = MainModel()
        assert hasattr(main.spec, "summarize")
        assert hasattr(main.spec, "spec_to_hdf5")
        assert hasattr(main.spec, "parser")

    def test_main_model_spec_summarize(self, minimal_spec_lines):
        """MainModel().spec.summarize(lines) matches SpecModel.summarize."""
        main = MainModel()
        summary = main.spec.summarize(minimal_spec_lines)
        assert len(summary) == 1
        assert summary[0]["index"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
