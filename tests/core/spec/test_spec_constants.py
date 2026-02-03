#!/usr/bin/python
# ----------------------------------------------------------------------------------
# Project: pds
# File: tests/core/spec/test_spec_constants.py
# ----------------------------------------------------------------------------------
# Purpose:
# Tests for pds.core.spec.constants (G_LABS for SPEC 6.03.03+).
# ----------------------------------------------------------------------------------
# Author: Christofanis Skordas
#
# Copyright (C) 2025-2026 GSECARS, The University of Chicago, USA
# Copyright (C) 2025-2026 NSF SEES, USA
# ----------------------------------------------------------------------------------

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pds.core.spec.constants import G_LABS


class TestSpecConstants:
    """Test SPEC constants (G_LABS for SPEC 6.03.03+)."""

    def test_g_labs_is_tuple_of_strings(self):
        """G_LABS is a non-empty tuple of strings."""
        assert isinstance(G_LABS, tuple)
        assert len(G_LABS) > 0
        assert all(isinstance(s, str) for s in G_LABS)

    def test_g_labs_no_duplicates(self):
        """G_LABS has no duplicate labels."""
        assert len(G_LABS) == len(set(G_LABS))

    def test_g_labs_first_and_last(self):
        """G_LABS starts with g_prefer and ends with g_111."""
        assert G_LABS[0] == "g_prefer"
        assert G_LABS[-1] == "g_111"

    def test_g_labs_contains_lattice_and_rodscan_labels(self):
        """G_LABS includes labels used by rodscan (g_aa_s, g_bb_s, g_ga_s) and common g_*."""
        required = ("g_aa", "g_bb", "g_cc", "g_H", "g_K", "g_L", "g_aa_s", "g_bb_s", "g_ga_s")
        for label in required:
            assert label in G_LABS, f"expected {label!r} in G_LABS"

    def test_g_labs_expected_length(self):
        """G_LABS has the expected number of entries for SPEC 6.03.03+."""
        # Current constant has 115 entries (g_prefer through g_111, CUT_*, etc.)
        assert len(G_LABS) == 115


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
