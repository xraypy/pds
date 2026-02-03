#!/usr/bin/python
# ----------------------------------------------------------------------------------
# Project: pds
# File: pds/core/spec/scan_attributes.py
# ----------------------------------------------------------------------------------
# Purpose:
# This file contains the SPEC scan-type attribute handlers for HDF5 conversion. It
# parses the SPEC scan commands and writes the scan-type-specific attributes onto
# the HDF5 groups. It supports the following scan types: ascan, a2scan, a4scan,
# Escan, hklscan, rodscan, timescan, and a default handler for unknown scan types.
# ----------------------------------------------------------------------------------
# Author: Christofanis Skordas
#
# Copyright (C) 2025-2026 GSECARS, The University of Chicago, USA
# Copyright (C) 2025-2026 NSF SEES, USA
# ----------------------------------------------------------------------------------

import math
from contextlib import suppress
from typing import Any, Protocol

import h5py

from pds.utils.converters import dataset_as_str_list

__all__ = ["process_scan_attributes"]


class ScanAttributeHandler(Protocol):
    """Protocol for setting scan-type-specific HDF5 attributes."""

    def apply(self, scan_group: h5py.Group, scan: dict[str, Any]) -> None: ...


def _set_base_scan_attrs(scan_group: h5py.Group, scan: dict[str, Any], scan_type: str) -> None:
    """Write common attributes for all scan types: full command and s_type."""
    scan_group.attrs["cmd"] = scan["cmd"]
    scan_group.attrs["s_type"] = scan_type


class _A2ScanHandler:
    """Handler for SPEC a2scan: two motors, start/stop, num_points, count_time."""

    def apply(self, scan_group: h5py.Group, scan: dict[str, Any]) -> None:
        cmd_parts = scan.get("cmd", "").split()
        if len(cmd_parts) < 9:
            return
        _set_base_scan_attrs(scan_group, scan, "a2scan")
        scan_group.attrs["motor1"] = cmd_parts[1]
        scan_group.attrs["m1_start"] = float(cmd_parts[2])
        scan_group.attrs["m1_stop"] = float(cmd_parts[3])
        scan_group.attrs["motor2"] = cmd_parts[4]
        scan_group.attrs["m2_start"] = float(cmd_parts[5])
        scan_group.attrs["m2_stop"] = float(cmd_parts[6])
        scan_group.attrs["num_points"] = float(cmd_parts[7])
        scan_group.attrs["count_time"] = float(cmd_parts[8])


class _A4ScanHandler:
    """Handler for SPEC a4scan: four motors, start/stop each, num_points, count_time."""

    def apply(self, scan_group: h5py.Group, scan: dict[str, Any]) -> None:
        cmd_parts = scan.get("cmd", "").split()
        if len(cmd_parts) < 15:
            return
        _set_base_scan_attrs(scan_group, scan, "a4scan")
        scan_group.attrs["motor1"] = cmd_parts[1]
        scan_group.attrs["m1_start"] = float(cmd_parts[2])
        scan_group.attrs["m1_stop"] = float(cmd_parts[3])
        scan_group.attrs["motor2"] = cmd_parts[4]
        scan_group.attrs["m2_start"] = float(cmd_parts[5])
        scan_group.attrs["m2_stop"] = float(cmd_parts[6])
        scan_group.attrs["motor3"] = cmd_parts[7]
        scan_group.attrs["m3_start"] = float(cmd_parts[8])
        scan_group.attrs["m3_stop"] = float(cmd_parts[9])
        scan_group.attrs["motor4"] = cmd_parts[10]
        scan_group.attrs["m4_start"] = float(cmd_parts[11])
        scan_group.attrs["m4_stop"] = float(cmd_parts[12])
        scan_group.attrs["num_points"] = float(cmd_parts[13])
        scan_group.attrs["count_time"] = float(cmd_parts[14])


class _AscanHandler:
    """Handler for SPEC ascan: one motor, start/stop, num_points, count_time."""

    def apply(self, scan_group: h5py.Group, scan: dict[str, Any]) -> None:
        cmd_parts = scan.get("cmd", "").split()
        if len(cmd_parts) < 6:
            return
        _set_base_scan_attrs(scan_group, scan, "ascan")
        scan_group.attrs["motor1"] = cmd_parts[1]
        scan_group.attrs["m1_start"] = float(cmd_parts[2])
        scan_group.attrs["m1_stop"] = float(cmd_parts[3])
        scan_group.attrs["num_points"] = float(cmd_parts[4])
        scan_group.attrs["count_time"] = float(cmd_parts[5])


class _EscanHandler:
    """Handler for SPEC Escan: energy start/stop, num_points, count_time; optional Q (h,k)."""

    def apply(self, scan_group: h5py.Group, scan: dict[str, Any]) -> None:
        cmd_parts = scan.get("cmd", "").split()
        if len(cmd_parts) < 5:
            return
        _set_base_scan_attrs(scan_group, scan, "Escan")
        scan_group.attrs["energy_start"] = float(cmd_parts[1])
        scan_group.attrs["energy_stop"] = float(cmd_parts[2])
        scan_group.attrs["num_points"] = float(cmd_parts[3])
        scan_group.attrs["count_time"] = float(cmd_parts[4])
        # Optional Q line: h,k for reciprocal-space position
        if scan.get("Q"):
            q_parts = scan["Q"].split()
            if len(q_parts) >= 2:
                scan_group.attrs["h_val"] = round(float(q_parts[0]), 1)
                scan_group.attrs["k_val"] = round(float(q_parts[1]), 1)


class _HklscanHandler:
    """Handler for SPEC hklscan: h,k,L ranges, num_points, count_time. h_val/k_val are midpoints."""

    def apply(self, scan_group: h5py.Group, scan: dict[str, Any]) -> None:
        cmd_parts = scan.get("cmd", "").split()
        if len(cmd_parts) < 9:
            return
        _set_base_scan_attrs(scan_group, scan, "hklscan")
        h_start = float(cmd_parts[1])
        h_stop = float(cmd_parts[2])
        k_start = float(cmd_parts[3])
        k_stop = float(cmd_parts[4])
        scan_group.attrs["h_start"] = h_start
        scan_group.attrs["h_stop"] = h_stop
        scan_group.attrs["k_start"] = k_start
        scan_group.attrs["k_stop"] = k_stop
        scan_group.attrs["h_val"] = round((h_start + h_stop) / 2, 1)
        scan_group.attrs["k_val"] = round((k_start + k_stop) / 2, 1)
        scan_group.attrs["L_start"] = float(cmd_parts[5])
        scan_group.attrs["L_stop"] = float(cmd_parts[6])
        scan_group.attrs["num_points"] = float(cmd_parts[7])
        scan_group.attrs["count_time"] = float(cmd_parts[8])


class _RodscanHandler:
    """Handler for SPEC rodscan: h,k fixed, L range, spacing, timing; computes hk_dist from param_data."""

    def apply(self, scan_group: h5py.Group, scan: dict[str, Any]) -> None:
        cmd_parts = scan.get("cmd", "").split()
        if len(cmd_parts) < 9:
            return
        _set_base_scan_attrs(scan_group, scan, "rodscan")
        scan_group.attrs["h_val"] = float(cmd_parts[1])
        scan_group.attrs["k_val"] = float(cmd_parts[2])
        scan_group.attrs["L_start"] = float(cmd_parts[3])
        scan_group.attrs["L_stop"] = float(cmd_parts[4])
        scan_group.attrs["L_space"] = float(cmd_parts[5])
        scan_group.attrs["max_time"] = float(cmd_parts[6])
        scan_group.attrs["peak_pos"] = float(cmd_parts[7])
        scan_group.attrs["peak_space"] = float(cmd_parts[8])

        # Compute in-plane distance hk_dist from lattice params (g_aa_s, g_bb_s, g_ga_s) in param_data
        with suppress(ValueError, IndexError, KeyError):
            param_labs_list = dataset_as_str_list(scan_group["param_labs"])
            param_data = list(scan_group["param_data"])
            h_val_d = float(param_data[param_labs_list.index("g_aa_s")])
            k_val_d = float(param_data[param_labs_list.index("g_bb_s")])
            t_val_d = math.radians(180 - float(param_data[param_labs_list.index("g_ga_s")]))
            h_val = scan_group.attrs["h_val"]
            k_val = scan_group.attrs["k_val"]
            # Law of cosines in reciprocal space for in-plane distance
            hk_dist = ((h_val * h_val_d) ** 2 + (k_val * k_val_d) ** 2 - 2 * h_val * h_val_d * k_val * k_val_d * math.cos(t_val_d)) ** 0.5
            scan_group.attrs["hk_dist"] = round(hk_dist, 8)


class _TimescanHandler:
    """Handler for SPEC timescan: count_time and time_space only."""

    def apply(self, scan_group: h5py.Group, scan: dict[str, Any]) -> None:
        cmd_parts = scan.get("cmd", "").split()
        if len(cmd_parts) < 3:
            return
        _set_base_scan_attrs(scan_group, scan, "timescan")
        scan_group.attrs["count_time"] = float(cmd_parts[1])
        scan_group.attrs["time_space"] = float(cmd_parts[2])


class _DefaultScanHandler:
    """Fallback handler for unknown SPEC scan types; sets only cmd and s_type."""

    def apply(self, scan_group: h5py.Group, scan: dict[str, Any]) -> None:
        cmd_parts = scan.get("cmd", "").split()
        if not cmd_parts:
            return
        _set_base_scan_attrs(scan_group, scan, cmd_parts[0])


_DEFAULT_HANDLER = _DefaultScanHandler()

# Registry mapping SPEC scan command name -> handler instance
_SCAN_ATTR_HANDLERS: dict[str, ScanAttributeHandler] = {
    "a2scan": _A2ScanHandler(),
    "a4scan": _A4ScanHandler(),
    "ascan": _AscanHandler(),
    "Escan": _EscanHandler(),
    "hklscan": _HklscanHandler(),
    "rodscan": _RodscanHandler(),
    "timescan": _TimescanHandler(),
}


def process_scan_attributes(scan_group: h5py.Group, scan: dict[str, Any]) -> None:
    """Apply scan-type-specific attributes using the handler registry.

    Dispatches to the handler for the scan type (first token of scan["cmd"]);
    unknown types use the default handler (cmd + s_type only).
    """
    cmd_parts = scan.get("cmd", "").split()
    if not cmd_parts:
        return
    scan_type = cmd_parts[0]
    handler = _SCAN_ATTR_HANDLERS.get(scan_type, _DEFAULT_HANDLER)
    handler.apply(scan_group, scan)
