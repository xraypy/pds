#!/usr/bin/python
# ----------------------------------------------------------------------------------
# Project: pds
# File: pds/model/__init__.py
# ----------------------------------------------------------------------------------
# Purpose:
# This is the model package. It contains all models for the PDS application.
# ----------------------------------------------------------------------------------
# Author: Christofanis Skordas
#
# Copyright (C) 2025-2026 GSECARS, The University of Chicago, USA
# Copyright (C) 2025-2026 NSF SEES, USA
# ----------------------------------------------------------------------------------

from pds.model.main_model import MainModel
from pds.model.spec_model import SpecModel

__all__ = ["MainModel", "SpecModel"]
