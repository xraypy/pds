#!/usr/bin/python
# ----------------------------------------------------------------------------------
# Project: pds
# File: pds/core/spec/spec_parser.py
# ----------------------------------------------------------------------------------
# Purpose:
# This file is used for the SPEC file parser. It reads SPEC 6.x control file lines
# and produces a list of scan summary dictionaries (one per scan), including header
# metadata (#F, #E, #O, #S, #D, #T, #G, #Q, #P, #N, #AT, #EN, #L) and point data
# following #L.
# ----------------------------------------------------------------------------------
# Author: Christofanis Skordas
#
# Copyright (C) 2025-2026 GSECARS, The University of Chicago, USA
# Copyright (C) 2025-2026 NSF SEES, USA
# ----------------------------------------------------------------------------------

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pds.core.spec.constants import G_LABS

__all__ = ["SpecParser"]


@dataclass
class _ParserState:
    """Mutable state accumulated while parsing a SPEC file until a scan is complete."""

    # File-level fields (persist across scans)
    spec_name: str | None = None
    epoch: int | None = None
    mnames: str = ""
    index: int = 0
    cmnd: str = ""
    n_sline: int = 0
    date: str | None = None
    xtime: str | None = None
    g_vals: str = ""
    q: str | None = None
    p_vals: str = ""
    ncols: int = 0
    atten: str | None = None
    energy: float | None = None
    lab: list[str] = field(default_factory=list)
    point_data: list[list[float]] = field(default_factory=list)
    aborted: bool = False
    nl_dat: int = 0

    def reset_scan(self) -> None:
        """Reset only scan-specific fields so the next scan starts from clean state."""
        self.cmnd = ""
        self.date = None
        self.xtime = None
        self.g_vals = ""
        self.q = None
        self.p_vals = ""
        self.atten = None
        self.energy = None
        self.lab = []
        self.point_data = []
        self.aborted = False
        self.nl_dat = 0
        self.index = 0
        self.ncols = 0
        self.n_sline = 0


class SpecParser:
    """Parses SPEC file lines into a list of scan dictionaries."""

    def __init__(self) -> None:
        self._state = _ParserState()
        # Dispatch by line prefix. Order matters: longer prefixes first
        self._handlers: list[tuple[str, str]] = [
            ("#F", "_on_file"),
            ("#E ", "_on_epoch"),
            ("#O0", "_on_mnames_reset"),
            ("#O", "_on_mnames"),
            ("#S ", "_on_scan"),
            ("#D ", "_on_date"),
            ("#T ", "_on_time"),
            ("#G0", "_on_g_reset"),
            ("#G", "_on_g"),
            ("#Q ", "_on_q"),
            ("#P0", "_on_p_reset"),
            ("#P", "_on_p"),
            ("#N ", "_on_ncols"),
            ("#AT", "_on_atten"),
            ("#EN", "_on_energy"),
            ("#L ", "_on_labels"),
        ]

    def parse(self, lines: list[str]) -> list[dict[str, Any]]:
        """Parse SPEC lines and return a list of scan summary dicts (one per scan)."""
        summary: list[dict[str, Any]] = []
        i = 0
        while i < len(lines):
            line = lines[i].rstrip("\n\r")
            lineno = i + 1
            consumed = self._dispatch(line, lineno, lines, i)
            # Only _on_labels returns a scan dict; it also sets _next_index past the data block.
            if consumed is not None:
                next_i = consumed.pop("_next_index", i + 1)
                summary.append(consumed)
                i = next_i
                self._state.reset_scan()
            else:
                i += 1
        return summary

    def _dispatch(self, line: str, lineno: int, lines: list[str], start: int) -> dict[str, Any] | None:
        """Route line to the appropriate handler. Returns a scan dict only when #L and its data block are processed."""
        for prefix, method_name in self._handlers:
            if line.startswith(prefix):
                handler = getattr(self, method_name)
                return handler(line, lineno, lines, start)
        return None

    def _on_file(self, line: str, _lineno: int, _lines: list[str], _start: int) -> dict[str, Any] | None:
        """#F path -> store file base name (e.g. spec file path)."""
        self._state.spec_name = Path(line[3:]).name
        return None

    def _on_epoch(self, line: str, _lineno: int, _lines: list[str], _start: int) -> dict[str, Any] | None:
        """#E N -> run epoch number."""
        self._state.epoch = int(line[3:])
        return None

    def _on_mnames_reset(self, line: str, _lineno: int, _lines: list[str], _start: int) -> dict[str, Any] | None:
        """#O0 ... -> replace motor/setting names (fresh list)."""
        self._state.mnames = line[3:]
        return None

    def _on_mnames(self, line: str, _lineno: int, _lines: list[str], _start: int) -> dict[str, Any] | None:
        """#O ... -> append to motor/setting names (continuation line)."""
        self._state.mnames += line[3:]
        return None

    def _on_scan(self, line: str, _lineno: int, _lines: list[str], _start: int) -> dict[str, Any] | None:
        """#S N command -> scan index N and full command string."""
        parts = line[3:].split()
        self._state.index = int(parts[0])
        self._state.cmnd = line[4 + len(parts[0]) :]
        self._state.n_sline = _lineno
        return None

    def _on_date(self, line: str, _lineno: int, _lines: list[str], _start: int) -> dict[str, Any] | None:
        """#D date -> scan date string."""
        self._state.date = line[3:]
        return None

    def _on_time(self, line: str, _lineno: int, _lines: list[str], _start: int) -> dict[str, Any] | None:
        """#T time -> scan time string."""
        self._state.xtime = line[3:]
        return None

    def _on_g_reset(self, line: str, _lineno: int, _lines: list[str], _start: int) -> dict[str, Any] | None:
        """#G0 ... -> replace global (gonio) values."""
        self._state.g_vals = line[3:]
        return None

    def _on_g(self, line: str, _lineno: int, _lines: list[str], _start: int) -> dict[str, Any] | None:
        """#G ... -> append to global values (continuation line)."""
        self._state.g_vals += line[3:]
        return None

    def _on_q(self, line: str, _lineno: int, _lines: list[str], _start: int) -> dict[str, Any] | None:
        """#Q ... -> Q or similar value string."""
        self._state.q = line[3:]
        return None

    def _on_p_reset(self, line: str, _lineno: int, _lines: list[str], _start: int) -> dict[str, Any] | None:
        """#P0 ... -> replace position values."""
        self._state.p_vals = line[3:]
        return None

    def _on_p(self, line: str, _lineno: int, _lines: list[str], _start: int) -> dict[str, Any] | None:
        """#P ... -> append to position values (continuation line)."""
        self._state.p_vals += line[3:]
        return None

    def _on_ncols(self, line: str, _lineno: int, _lines: list[str], _start: int) -> dict[str, Any] | None:
        """#N N -> number of data columns."""
        self._state.ncols = int(line[3:])
        return None

    def _on_atten(self, line: str, _lineno: int, _lines: list[str], _start: int) -> dict[str, Any] | None:
        """#AT ... -> attenuator info (strip 6 chars for '#AT  ' or similar)."""
        self._state.atten = line[6:].strip()
        return None

    def _on_energy(self, line: str, _lineno: int, _lines: list[str], _start: int) -> dict[str, Any] | None:
        """#EN value -> energy (strip 8 chars for '#EN ' and parse float)."""
        self._state.energy = float(line[8:])
        return None

    def _on_labels(self, line: str, lineno: int, lines: list[str], start: int) -> dict[str, Any]:
        """#L label1 label2 ... -> column labels; then read data rows until next #S or EOF."""
        self._state.lab = line[3:].split()
        s = self._state
        point_data: list[list[float]] = []
        aborted = False
        j = start + 1
        # Consume lines after #L: data rows (space-separated floats) until next #S or comment-only.
        while j < len(lines):
            data_line = lines[j]
            if data_line.startswith("#S "):
                break
            if data_line.startswith("#"):
                if "aborted" in data_line:
                    aborted = True
                j += 1
                continue
            if data_line.strip():
                try:
                    point_data.append([float(x) for x in data_line.split()])
                except ValueError:
                    aborted = True
                    break
            j += 1
        s.point_data = point_data
        s.aborted = aborted
        s.nl_dat = len(point_data)
        # Derive L start/stop from column "L" if present (for scan range display).
        L_start = L_stop = "--"
        try:
            L_pos = s.lab.index("L")
            if L_pos >= 0 and point_data:
                L_start = point_data[0][L_pos]
                L_stop = point_data[-1][L_pos]
        except (ValueError, IndexError):
            pass
        mnames_list = s.mnames.split() if s.mnames else []
        G = [float(x) for x in s.g_vals.split()] if s.g_vals else []
        P = [float(x) for x in s.p_vals.split()] if s.p_vals else []
        # Build one scan summary dict; _next_index tells parse() where to resume.
        scan_data: dict[str, Any] = {
            "index": s.index,
            "spec_name": s.spec_name,
            "init_epoch": s.epoch,
            "nl_start": s.n_sline,
            "cmd": s.cmnd,
            "date": s.date,
            "time": s.xtime,
            "mnames": mnames_list,
            "P": P,
            "g_labs": list(G_LABS),
            "G": G,
            "Q": s.q,
            "ncols": s.ncols,
            "labels": s.lab,
            "atten": s.atten,
            "energy": s.energy,
            "lineno": lineno,
            "aborted": s.aborted,
            "point_data": s.point_data,
            "real_L_start": L_start,
            "real_L_stop": L_stop,
            "nl_dat": s.nl_dat,
            "_next_index": j,
        }
        return scan_data
