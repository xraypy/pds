#!/usr/bin/python
# ----------------------------------------------------------------------------------
# Project: pds
# File: tests/core/spec/test_spec_image_reader.py
# ----------------------------------------------------------------------------------
# Purpose:
# Tests for pds.core.spec.image_reader (SPEC scan image reader).
# ----------------------------------------------------------------------------------
# Author: Christofanis Skordas
#
# Copyright (C) 2025-2026 GSECARS, The University of Chicago, USA
# Copyright (C) 2025-2026 NSF SEES, USA
# ----------------------------------------------------------------------------------

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pds.core.spec.image_reader import ImageReader


class TestImageReader:
    """Test SPEC scan image reader."""

    def test_read_supported_mode_I(self):
        """Mode 'I' (32-bit int) returns array with correct shape and dtype."""
        reader = ImageReader()
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = Path(f.name)
        try:
            # 3x2 image, mode I -> int32
            pil_im = Image.new("I", (3, 2), color=100)
            pil_im.save(path)
            arr = reader.read(path)
            assert arr is not None
            assert arr.dtype == np.int32
            assert arr.shape == (2, 3)
            np.testing.assert_array_equal(arr, 100)
        finally:
            path.unlink(missing_ok=True)

    def test_read_supported_mode_I16(self):
        """Mode 'I;16' (16-bit int) returns array with correct shape and dtype."""
        reader = ImageReader()
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = Path(f.name)
        try:
            pil_im = Image.new("I;16", (2, 4), color=500)
            pil_im.save(path)
            arr = reader.read(path)
            assert arr is not None
            assert arr.dtype == np.int16
            assert arr.shape == (4, 2)
        finally:
            path.unlink(missing_ok=True)

    def test_read_unsupported_mode_returns_none(self, capsys):
        """Unsupported PIL mode (e.g. RGB) returns None and prints message."""
        reader = ImageReader()
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
            path = Path(f.name)
        try:
            pil_im = Image.new("RGB", (2, 2), color=(1, 2, 3))
            pil_im.save(path)
            arr = reader.read(path)
            assert arr is None
            out = capsys.readouterr()
            assert "unsupported" in out.out.lower() or "RGB" in out.out
        finally:
            path.unlink(missing_ok=True)

    def test_read_missing_file_returns_none(self, capsys):
        """Missing file returns None and prints error."""
        reader = ImageReader()
        path = Path("/nonexistent/path/to/image.tif")
        arr = reader.read(path)
        assert arr is None
        out = capsys.readouterr()
        assert "failed" in out.out.lower() or "error" in out.out.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
