#!/usr/bin/python
# ----------------------------------------------------------------------------------
# Project: pds
# File: tests/core/spec/test_spec_scan_attributes.py
# ----------------------------------------------------------------------------------
# Purpose:
# Tests for pds.core.spec.scan_attributes (scan-type-specific HDF5 attribute handlers).
# ----------------------------------------------------------------------------------
# Author: Christofanis Skordas
#
# Copyright (C) 2025-2026 GSECARS, The University of Chicago, USA
# Copyright (C) 2025-2026 NSF SEES, USA
# ----------------------------------------------------------------------------------

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pds.core.spec.scan_attributes import process_scan_attributes


def _make_scan_group(h5_file, param_labs=None, param_data=None):
    """Create a group with optional param_labs/param_data datasets."""
    grp = h5_file.create_group("scan")
    if param_labs is not None:
        grp.create_dataset("param_labs", data=param_labs, dtype=h5py.string_dtype(encoding="utf-8"))
    if param_data is not None:
        grp.create_dataset("param_data", data=np.array(param_data, dtype=float))
    return grp


class TestProcessScanAttributes:
    """Test scan-type-specific attribute handlers."""

    def test_ascan(self):
        """ascan sets motor1, m1_start, m1_stop, num_points, count_time."""
        with h5py.File("test.h5", "w", driver="core", backing_store=False) as f:
            grp = f.create_group("scan")
            scan = {"cmd": "ascan mu 0 10 5 1"}
            process_scan_attributes(grp, scan)
            assert grp.attrs["s_type"] == "ascan"
            assert grp.attrs["cmd"] == "ascan mu 0 10 5 1"
            assert grp.attrs["motor1"] == "mu"
            assert grp.attrs["m1_start"] == 0.0
            assert grp.attrs["m1_stop"] == 10.0
            assert grp.attrs["num_points"] == 5.0
            assert grp.attrs["count_time"] == 1.0

    def test_a2scan(self):
        """a2scan sets two motors and shared num_points, count_time."""
        with h5py.File("test.h5", "w", driver="core", backing_store=False) as f:
            grp = f.create_group("scan")
            scan = {"cmd": "a2scan mu 0 10 chi 5 15 4 0.5"}
            process_scan_attributes(grp, scan)
            assert grp.attrs["s_type"] == "a2scan"
            assert grp.attrs["motor1"] == "mu"
            assert grp.attrs["m1_start"] == 0.0
            assert grp.attrs["m1_stop"] == 10.0
            assert grp.attrs["motor2"] == "chi"
            assert grp.attrs["m2_start"] == 5.0
            assert grp.attrs["m2_stop"] == 15.0
            assert grp.attrs["num_points"] == 4.0
            assert grp.attrs["count_time"] == 0.5

    def test_a4scan(self):
        """a4scan sets four motors and num_points, count_time."""
        with h5py.File("test.h5", "w", driver="core", backing_store=False) as f:
            grp = f.create_group("scan")
            cmd = "a4scan m1 0 1 m2 0 2 m3 0 3 m4 0 4 10 1.0"
            scan = {"cmd": cmd}
            process_scan_attributes(grp, scan)
            assert grp.attrs["s_type"] == "a4scan"
            assert grp.attrs["motor1"] == "m1"
            assert grp.attrs["m4_stop"] == 4.0
            assert grp.attrs["num_points"] == 10.0
            assert grp.attrs["count_time"] == 1.0

    def test_Escan_without_Q(self):
        """Escan sets energy_start/stop, num_points, count_time."""
        with h5py.File("test.h5", "w", driver="core", backing_store=False) as f:
            grp = f.create_group("scan")
            scan = {"cmd": "Escan 8.0 9.0 20 0.5"}
            process_scan_attributes(grp, scan)
            assert grp.attrs["s_type"] == "Escan"
            assert grp.attrs["energy_start"] == 8.0
            assert grp.attrs["energy_stop"] == 9.0
            assert grp.attrs["num_points"] == 20.0
            assert grp.attrs["count_time"] == 0.5
            assert "h_val" not in grp.attrs

    def test_Escan_with_Q(self):
        """Escan with Q sets h_val and k_val from Q string."""
        with h5py.File("test.h5", "w", driver="core", backing_store=False) as f:
            grp = f.create_group("scan")
            scan = {"cmd": "Escan 8.0 9.0 20 0.5", "Q": "1.0 2.0"}
            process_scan_attributes(grp, scan)
            assert grp.attrs["h_val"] == 1.0
            assert grp.attrs["k_val"] == 2.0

    def test_hklscan(self):
        """hklscan sets h/k/L start/stop, midpoints h_val/k_val, num_points, count_time."""
        with h5py.File("test.h5", "w", driver="core", backing_store=False) as f:
            grp = f.create_group("scan")
            scan = {"cmd": "hklscan 1.0 2.0 0.0 1.0 0 2 5 1"}
            process_scan_attributes(grp, scan)
            assert grp.attrs["s_type"] == "hklscan"
            assert grp.attrs["h_start"] == 1.0
            assert grp.attrs["h_stop"] == 2.0
            assert grp.attrs["k_start"] == 0.0
            assert grp.attrs["k_stop"] == 1.0
            assert grp.attrs["h_val"] == 1.5
            assert grp.attrs["k_val"] == 0.5
            assert grp.attrs["L_start"] == 0.0
            assert grp.attrs["L_stop"] == 2.0
            assert grp.attrs["num_points"] == 5.0
            assert grp.attrs["count_time"] == 1.0

    def test_rodscan_without_param_data(self):
        """rodscan sets h_val, k_val, L_*, and does not set hk_dist without param_data."""
        with h5py.File("test.h5", "w", driver="core", backing_store=False) as f:
            grp = f.create_group("scan")
            scan = {"cmd": "rodscan 1 0 0 2 0.1 60 0.5 0.05"}
            process_scan_attributes(grp, scan)
            assert grp.attrs["s_type"] == "rodscan"
            assert grp.attrs["h_val"] == 1.0
            assert grp.attrs["k_val"] == 0.0
            assert grp.attrs["L_start"] == 0.0
            assert grp.attrs["L_stop"] == 2.0
            assert "hk_dist" not in grp.attrs

    def test_rodscan_with_param_data_sets_hk_dist(self):
        """rodscan with param_labs/param_data (g_aa_s, g_bb_s, g_ga_s) sets hk_dist."""
        with h5py.File("test.h5", "w", driver="core", backing_store=False) as f:
            grp = _make_scan_group(f, param_labs=["g_aa_s", "g_bb_s", "g_ga_s"], param_data=[0.1, 0.1, 90.0])
            scan = {"cmd": "rodscan 1 0 0 2 0.1 60 0.5 0.05"}
            process_scan_attributes(grp, scan)
            assert grp.attrs["s_type"] == "rodscan"
            assert "hk_dist" in grp.attrs
            assert isinstance(grp.attrs["hk_dist"], (int, float))

    def test_timescan(self):
        """timescan sets count_time and time_space."""
        with h5py.File("test.h5", "w", driver="core", backing_store=False) as f:
            grp = f.create_group("scan")
            scan = {"cmd": "timescan 2.0 0.1"}
            process_scan_attributes(grp, scan)
            assert grp.attrs["s_type"] == "timescan"
            assert grp.attrs["count_time"] == 2.0
            assert grp.attrs["time_space"] == 0.1

    def test_unknown_scan_type_default_handler(self):
        """Unknown scan type sets only cmd and s_type."""
        with h5py.File("test.h5", "w", driver="core", backing_store=False) as f:
            grp = f.create_group("scan")
            scan = {"cmd": "unknown_scan 1 2 3"}
            process_scan_attributes(grp, scan)
            assert grp.attrs["s_type"] == "unknown_scan"
            assert grp.attrs["cmd"] == "unknown_scan 1 2 3"
            assert list(grp.attrs.keys()) == ["cmd", "s_type"]

    def test_empty_cmd_does_nothing(self):
        """Empty cmd does not set attributes."""
        with h5py.File("test.h5", "w", driver="core", backing_store=False) as f:
            grp = f.create_group("scan")
            scan = {"cmd": ""}
            process_scan_attributes(grp, scan)
            assert len(grp.attrs) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
