#!/usr/bin/python
# ----------------------------------------------------------------------------------
# Project: pds
# File: pds/model/spec_model.py
# ----------------------------------------------------------------------------------
# Purpose:
# This is the model for SPEC. It contains the parser, image reader, and hdf5
# converter.
# ----------------------------------------------------------------------------------
# Author: Christofanis Skordas
#
# Copyright (C) 2025-2026 GSECARS, The University of Chicago, USA
# Copyright (C) 2025-2026 NSF SEES, USA
# ----------------------------------------------------------------------------------

from pathlib import Path
from typing import Any

from pds.core.spec import Hdf5Converter, ImageReader, SpecParser


class SpecModel:
    """SPEC model for parser, image reader, and hdf5 converter."""

    def __init__(self) -> None:
        self.parser = SpecParser()
        self.image_reader = ImageReader()

    def summarize(self, lines: list[str]) -> list[dict[str, Any]]:
        """Parse SPEC file lines and return a list of scan summary dicts."""
        return self.parser.parse(lines)

    def spec_to_hdf5(
        self, spec_file: Path, output_file: Path, *, mode: str = "a", image_dir: Path | None = None, verbose: bool = True, lock_timeout: float = 10
    ) -> bool:
        """Convert a SPEC file to HDF5 format."""
        converter = Hdf5Converter(spec_file, output_file, mode=mode, image_dir=image_dir, verbose=verbose, lock_timeout=lock_timeout)
        return converter.run()
