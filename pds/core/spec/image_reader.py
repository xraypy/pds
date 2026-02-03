#!/usr/bin/python
# ----------------------------------------------------------------------------------
# Project: pds
# File: pds/core/spec/image_reader.py
# ----------------------------------------------------------------------------------
# Purpose:
# Image reader for SPEC scan images. Reads image files into numpy arrays with
# supported mode -> dtype mapping (e.g. PIL "I" -> int32, "I;16" -> int16).
# ----------------------------------------------------------------------------------
# Author: Christofanis Skordas
#
# Copyright (C) 2025-2026 GSECARS, The University of Chicago, USA
# Copyright (C) 2025-2026 NSF SEES, USA
# ----------------------------------------------------------------------------------

from pathlib import Path

import numpy as np
from numpy.typing import DTypeLike
from PIL import Image

__all__ = ["ImageReader"]

_IMAGE_MODE_DTYPE: dict[str, DTypeLike] = {"I": np.int32, "I;16": np.int16}


class ImageReader:
    """Reads SPEC scan image files into numpy arrays with supported mode -> dtype mapping."""

    def __init__(self) -> None:
        self._mode_dtype = dict(_IMAGE_MODE_DTYPE)

    def read(self, image_path: Path) -> np.ndarray | None:
        """Read SPEC scan image at path; returns array or None on failure/unsupported mode.

        On success, returns a 2D array with shape (height, width) and dtype one of
        the supported types (e.g. int32 for mode \"I\", int16 for \"I;16\").
        """
        try:
            with Image.open(image_path) as im:
                dtype = self._mode_dtype.get(im.mode)
                if dtype is None:
                    supported = ", ".join(sorted(self._mode_dtype))
                    print(f"SPEC image reader: unsupported PIL mode '{im.mode}' for {image_path} (supported: {supported})")
                    return None
                arr = np.array(im, dtype=dtype)
                return arr
        except OSError as e:
            print(f"SPEC image reader: failed to read {image_path}: {e}")
            return None
        except ValueError as e:
            print(f"SPEC image reader: failed to read {image_path}: {e}")
            return None
