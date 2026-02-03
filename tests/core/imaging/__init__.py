#!/usr/bin/python
# ----------------------------------------------------------------------------------
# Project: pds
# File: tests/core/imaging/__init__.py
# ----------------------------------------------------------------------------------
# Purpose:
# Imaging test package: tests for pds.core.imaging (background, image_data).
# Exposes shared pytest fixtures (test_data_simple, test_data_complex,
# test_image_simple) for use by test modules.
# ----------------------------------------------------------------------------------
# Author: Christofanis Skordas
#
# Copyright (C) 2025-2026 GSECARS, The University of Chicago, USA
# Copyright (C) 2025-2026 NSF SEES, USA
# ----------------------------------------------------------------------------------

import numpy as np
import pytest

__all__ = ["test_data_simple", "test_data_complex", "test_image_simple"]


@pytest.fixture
def test_data_simple():
    """Simple 1D curve data for background tests (gaussian + linear + noise, 100 pts)."""
    np.random.seed(42)

    def gauss(x, cen, sigma):
        return np.exp(-((x - cen) ** 2) / (2 * sigma**2))

    npts = 100
    x = 1.0 * np.arange(npts)
    g1 = 40 * gauss(x, 0.6 * npts, 8.0)
    g2 = 20 * gauss(x, 0.5 * npts, 15.0)
    r = 2 * np.random.normal(size=npts)
    y = r + x / 25 + g1 + g2

    return x, y


@pytest.fixture
def test_data_complex():
    """Complex 1D curve data for background tests (2000 pts, original Python 2 example)."""
    np.random.seed(42)

    def gauss(x, cen, sigma):
        return np.exp(-((x - cen) ** 2) / (2 * sigma**2))

    npts = 2000
    x = 1.0 * np.arange(npts)
    g1 = 40 * gauss(x, 0.6 * npts, 8.0)
    g2 = 20 * gauss(x, 0.5 * npts, 270)
    r = 2 * np.random.normal(size=npts)
    y = r + x / 25 + g1 + g2

    return x, y


@pytest.fixture
def test_image_simple():
    """Simple 2D int32 image (50x60) with peaks and background for image_data tests."""
    np.random.seed(42)

    image = np.zeros((50, 60), dtype=np.int32)

    for i in range(50):
        for j in range(60):
            image[i, j] = int(10 + 5 * np.sin(i / 10.0) + 3 * np.cos(j / 8.0))

    image[20:25, 25:30] += 100
    image[35:38, 45:48] += 50

    noise = np.random.randint(-5, 6, size=(50, 60))
    image = image + noise
    image = np.maximum(image, 0)

    return image.astype(np.int32)
