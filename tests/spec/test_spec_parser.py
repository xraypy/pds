#!/usr/bin/python
# ----------------------------------------------------------------------------------
# Project: pds
# File: tests/spec/test_spec_parser.py
# ----------------------------------------------------------------------------------
# Purpose:
# Tests for pds.core.spec.spec_parser (SPEC file parser).
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

from pds.core.spec.spec_parser import SpecParser
from tests.spec import minimal_spec_lines_one_scan


class TestSpecParser:
    """Test SPEC file parser."""

    def test_parse_minimal_one_scan(self):
        """One scan with #F, #E, #S, #D, #L and two data rows."""
        parser = SpecParser()
        lines = minimal_spec_lines_one_scan()
        summary = parser.parse(lines)
        assert len(summary) == 1
        scan = summary[0]
        assert scan["index"] == 1
        assert scan["spec_name"] == "sample.spec"
        assert scan["init_epoch"] == 1
        assert "ascan" in scan["cmd"]
        assert scan["labels"] == ["H", "K", "L", "counts"]
        assert scan["nl_dat"] == 2
        assert len(scan["point_data"]) == 2
        assert scan["point_data"][0] == [1.0, 0.0, 0.0, 100.0]
        assert "_next_index" not in scan

    def test_parse_two_scans(self):
        """Two scans: correct indices and row counts."""
        parser = SpecParser()
        lines = minimal_spec_lines_one_scan() + ["#S 2 ascan mu 0 20 3 0.5\n", "#L mu  counts\n", "0.0 10\n", "10.0 20\n", "20.0 30\n"]
        summary = parser.parse(lines)
        assert len(summary) == 2
        assert summary[0]["index"] == 1 and summary[0]["nl_dat"] == 2
        assert summary[1]["index"] == 2 and summary[1]["nl_dat"] == 3

    def test_parse_aborted_scan(self):
        """Comment line with 'aborted' sets aborted True."""
        parser = SpecParser()
        lines = ["#S 1 ascan mu 0 10 2 1\n", "#L mu  counts\n", "0.0 10\n", "#C aborted\n"]
        summary = parser.parse(lines)
        assert len(summary) == 1
        assert summary[0]["aborted"] is True

    def test_parse_L_column_derives_real_L_start_stop(self):
        """Labels include L; real_L_start/real_L_stop from first/last row."""
        parser = SpecParser()
        lines = ["#S 1 ascan L 0 2 3 1\n", "#L L  counts\n", "0.0 100\n", "1.0 200\n", "2.0 300\n"]
        summary = parser.parse(lines)
        assert len(summary) == 1
        assert summary[0]["real_L_start"] == 0.0
        assert summary[0]["real_L_stop"] == 2.0

    def test_parse_empty_lines_no_scan(self):
        """Empty input returns no scans."""
        parser = SpecParser()
        summary = parser.parse([])
        assert summary == []

    def test_parse_no_L_returns_no_scan(self):
        """Only #F/#S without #L never completes a scan."""
        parser = SpecParser()
        lines = ["#F /a/b.spec\n", "#S 1 ascan mu 0 10 5 1\n"]
        summary = parser.parse(lines)
        assert summary == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
