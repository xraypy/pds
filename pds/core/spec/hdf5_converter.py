#!/usr/bin/python
# ----------------------------------------------------------------------------------
# Project: pds
# File: pds/core/spec/hdf5_converter.py
# ----------------------------------------------------------------------------------
# Purpose:
# This file contains the HDF5 converter model. It is used to convert a SPEC file to
# HDF5 format using a file lock and optional image directory.
# ----------------------------------------------------------------------------------
# Author: Christofanis Skordas
#
# Copyright (C) 2025-2026 GSECARS, The University of Chicago, USA
# Copyright (C) 2025-2026 NSF SEES, USA
# ----------------------------------------------------------------------------------

import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pds.core.spec.image_reader import ImageReader
from pds.core.spec.scan_attributes import process_scan_attributes
from pds.core.spec.spec_parser import SpecParser
from pds.utils.file_locker import FileLock, FileLockException

__all__ = ["Hdf5Converter"]

# Keys stored as HDF5 datasets; do not store these as group attributes.
_KEYS_STORED_AS_DATASETS = frozenset(("labels", "point_data", "g_labs", "mnames", "G", "P", "_next_index"))


class Hdf5Converter:
    """Converts a SPEC file to HDF5 using a file lock and optional image directory."""

    def __init__(
        self, spec_file: Path, output_file: Path, *, mode: str = "a", image_dir: Path | None = None, verbose: bool = True, lock_timeout: float = 10
    ) -> None:
        self.spec_file = spec_file
        self.output_file = output_file
        self.mode = mode
        self.image_dir = image_dir or self._default_image_dir()
        self.verbose = verbose
        self.lock_timeout = lock_timeout
        self._parser = SpecParser()
        self._image_reader = ImageReader()

    def _default_image_dir(self) -> Path:
        spec_dir = self.spec_file.parent
        base = self.spec_file.stem
        if base.endswith(".spc"):
            base = base[:-4]
        return spec_dir / "images" / base

    def run(self) -> bool:
        """Perform the conversion; returns True on success."""
        if not self.spec_file.exists():
            print(f"Error: SPEC file not found: {self.spec_file}")
            return False
        start = time.time()
        lines = self._read_spec_lines()
        if lines is None:
            return False
        if self.verbose:
            print(f"Parsing SPEC file: {self.spec_file}")
        summary = self._parser.parse(lines)
        if not summary:
            print("Warning: No scans found in SPEC file")
            return False
        lock = FileLock(str(self.output_file), timeout=self.lock_timeout)
        try:
            if self.verbose:
                print("Acquiring file lock...")
            # Blocks until lock is held or timeout occurs
            lock.acquire()
            if self.verbose:
                print("Lock acquired")
        except FileLockException as e:
            print(f"Error acquiring lock: {e}")
            return False
        try:
            success = self._write_hdf5(summary)
        finally:
            lock.release()
            if self.verbose:
                print("Lock released")
        if self.verbose and success:
            print(f"Conversion completed in {(time.time() - start) / 60:.2f} minutes")
        return success

    def _read_spec_lines(self) -> list[str] | None:
        try:
            # SPEC files may contain legacy non-UTF-8 bytes; ignore invalid chars
            with open(self.spec_file, encoding="utf-8", errors="ignore") as f:
                return f.readlines()
        except OSError as e:
            print(f"Error reading SPEC file: {e}")
            return None

    def _write_hdf5(self, summary: list[dict[str, Any]]) -> bool:
        try:
            with h5py.File(self.output_file, self.mode) as master_file:
                spec_group = master_file.require_group(self.spec_file.name)
                for scan in summary:
                    # One group per scan (index)
                    self._write_scan(spec_group, scan)
            return True
        except OSError as e:
            print(f"Error during HDF5 conversion: {e}")
            return False
        except ValueError as e:
            print(f"Error during HDF5 conversion: {e}")
            return False

    def _write_scan(self, spec_group: h5py.Group, scan: dict[str, Any]) -> None:
        scan_index = str(scan["index"])
        scan_group = spec_group.require_group(scan_index)
        # Skip re-write if point_data already exists with same row count
        if "point_data" in scan_group and len(scan_group["point_data"]) == scan["nl_dat"]:
            if self.verbose:
                print(f"Skipping scan {scan_index} (already complete)")
            return
        if self.verbose:
            print(f"Processing scan {scan_index}")
        for name in ("point_data", "point_labs", "param_labs", "param_data"):
            if name in scan_group:
                del scan_group[name]
        scan_group.create_dataset("point_data", data=scan["point_data"])
        scan_group.create_dataset("point_labs", data=scan["labels"], dtype=h5py.string_dtype(encoding="utf-8"))
        param_labs = list(scan["g_labs"]) + list(scan["mnames"])
        scan_group.create_dataset("param_labs", data=param_labs, dtype=h5py.string_dtype(encoding="utf-8"))
        scan_group.create_dataset("param_data", data=list(scan["G"]) + list(scan["P"]))
        for key, value in scan.items():
            if key in _KEYS_STORED_AS_DATASETS:
                continue
            if value is None:
                continue
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            scan_group.attrs[key] = value
        process_scan_attributes(scan_group, scan)
        self._write_scan_images(scan_group, scan)

    def _write_scan_images(self, scan_group: h5py.Group, scan: dict[str, Any]) -> None:
        """Read .tif images from scan image dir and write as image_data dataset; failed reads become -1 placeholder."""
        scan_image_dir = self.image_dir / f"S{scan['index']:03d}"
        if not scan_image_dir.exists():
            return
        if self.verbose:
            print(f"Processing images in: {scan_image_dir}")
        image_data: list[np.ndarray] = []
        for image_file in sorted(scan_image_dir.iterdir(), key=lambda p: p.name):
            if image_file.suffix.lower() != ".tif":
                continue
            arr = self._image_reader.read(image_file)
            if arr is not None:
                image_data.append(arr)
            elif image_data:
                image_data.append(np.full_like(image_data[-1], -1))
        if image_data:
            if "image_data" in scan_group:
                del scan_group["image_data"]
            scan_group.create_dataset("image_data", data=image_data, compression="gzip")
