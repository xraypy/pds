#!/usr/bin/python
# ----------------------------------------------------------------------------------
# Project: pds
# File: pds/core/spec/constants.py
# ----------------------------------------------------------------------------------
# Purpose:
# This file contains the SPEC constants for the SPEC 6.03.03+ version.
# ----------------------------------------------------------------------------------
# Author: Christofanis Skordas
#
# Copyright (C) 2025-2026 GSECARS, The University of Chicago, USA
# Copyright (C) 2025-2026 NSF SEES, USA
# ----------------------------------------------------------------------------------

__all__ = ["G_LABS"]

# fmt: off
G_LABS: tuple[str, ...] = (
    "g_prefer", "g_sect", "g_frz", "g_haz", "g_kaz", "g_laz", "g_zh0", "g_zk0", "g_z10", "g_zh1", "g_zk1", "g_zl1",
    "g_kappa", "g_13", "g_14", "g_sigtau", "g_mode1", "g_mode2", "g_mode3", "g_mode4", "g_mode5", "g_21", "g_new1",
    "g_aa", "g_bb", "g_cc", "g_al", "g_be", "g_ga", "g_aa_s", "g_bb_s", "g_cc_s", "g_al_s", "g_be_s", "g_ga_s",
    "g_h0", "g_k0", "g_l0", "g_h1", "g_k1", "g_l1", "g_u00", "g_u01", "g_u02", "g_u03", "g_u04", "g_u05",
    "g_u10", "g_u11", "g_u12", "g_u13", "g_u14", "g_u15", "g_lambda0", "g_lambda1", "g_new2", "g_new3",
    "g_54", "g_55", "g_56", "g_57", "g_58", "g_59", "g_60", "g_61", "g_62",
    "g_H", "g_K", "g_L", "g_LAMBDA", "g_ALPHA", "g_BETA", "g_OMEGA", "g_TTH", "g_PSI", "g_TAU", "g_QAZ", "g_NAZ",
    "g_SIGMA_AZ", "g_TAU_AZ", "g_F_ALPHA", "g_F_BETA", "g_F_OMEGA", "g_F_PSI", "g_F_NAZ", "g_F_QAZ", "g_F_DEL",
    "g_F_ETA", "g_F_CHI", "g_F_PHI", "g_F_NU", "g_F_MU", "g_F_CHI_Z", "g_F_PHI_Z",
    "CUT_DEL", "CUT_ETA", "CUT_CHI", "CUT_PHI", "CUT_NU", "CUT_MU", "CUT_KETA", "CUT_KAP", "CUT_KPHI",
    "g_100", "g_101", "g_102", "g_103", "g_104", "g_105", "g_106", "g_107", "g_108", "g_109", "g_110", "g_111",
)
# fmt: on
