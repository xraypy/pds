#!/usr/bin/python
# ----------------------------------------------------------------------------------
# Project: pds
# File: tests/spec/__init__.py
# ----------------------------------------------------------------------------------
# Purpose:
# SPEC test package: tests for pds.core.spec and pds.model (SPEC parsing, HDF5
# conversion, scan attributes, image reader, constants). Exposes shared minimal SPEC
# fixtures for use by test modules.
# ----------------------------------------------------------------------------------
# Author: Christofanis Skordas
#
# Copyright (C) 2025-2026 GSECARS, The University of Chicago, USA
# Copyright (C) 2025-2026 NSF SEES, USA
# ----------------------------------------------------------------------------------

__all__ = ["minimal_spec_lines", "minimal_spec_content", "minimal_spec_lines_one_scan"]


def minimal_spec_lines():
    """Minimal SPEC lines (one scan, mu/counts, 2 points). For model and HDF5 tests."""
    return ["#F /some/dir/sample.spec\n", "#E 1\n", "#S 1 ascan mu 0 10 2 1\n", "#D 2025-01-01\n", "#L mu  counts\n", "0.0 100\n", "10.0 200\n"]


def minimal_spec_content():
    """Minimal SPEC file content as string. For writing to a file (HDF5 converter tests)."""
    return "\n".join(line.rstrip() for line in minimal_spec_lines())


def minimal_spec_lines_one_scan():
    """Minimal SPEC lines with H,K,L,counts columns (one scan, 2 points). For parser tests."""
    return [
        "#F /some/dir/sample.spec\n",
        "#E 1\n",
        "#S 1 ascan mu 0 10 5 1\n",
        "#D 2025-01-01\n",
        "#L H  K  L  counts\n",
        "1.0 0.0 0.0 100\n",
        "1.0 0.0 1.0 200\n",
    ]
