#!/usr/bin/python
# ----------------------------------------------------------------------------------
# Project: pds
# File: pds/model/main_model.py
# ----------------------------------------------------------------------------------
# Purpose:
# This file initializes and exposes all models. The controller creates one MainModel
# instance (e.g. self.model = MainModel()) and accesses sub-models via
# self.model.spec.parser, self.model.spec.summarize(...), etc.
# ----------------------------------------------------------------------------------
# Author: Christofanis Skordas
#
# Copyright (C) 2025-2026 GSECARS, The University of Chicago, USA
# Copyright (C) 2025-2026 NSF SEES, USA
# ----------------------------------------------------------------------------------

from pds.model.spec_model import SpecModel

__all__ = ["MainModel"]


class MainModel:
    """Initializes and exposes sub-models."""

    def __init__(self) -> None:
        self.spec = SpecModel()
