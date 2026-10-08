#!/usr/bin/env python3
"""The one palette. Every figure generator imports from here.

Two rules the manuscript depends on:

  1. Whenever latency, rate and delta appear as categories they are blue,
     orange and green, in that order, on every figure. A reader who learns the
     mapping on one plot must not have to relearn it on the next.
  2. Colour is never the only channel. Each encoding also carries a marker
     shape, a line style and a hatch, so the figures survive greyscale printing
     and colour-vision deficiency.

Centralised rather than copied because the mapping already drifted once: the
slopegraph used grey for delta while the Pareto plot used green, which is
exactly the inconsistency a shared constant makes impossible.
"""
from __future__ import annotations

# --- master palette ---------------------------------------------------------
LATENCY = "#0072B2"
RATE = "#D55E00"
DELTA = "#009E73"
DERIVED = "#CC79A7"      # derived/SOP metrics; deliberately not an encoding hue
TEXT = "#333333"
ANNOT = "#595959"
NEUTRAL = "#B3B3B3"
NEUTRAL_DK = "#8C8C8C"
GRID = "#E6E6E6"
BG_LIGHT = "#F2F2F2"
ZERO = "#222222"

#: Encoding -> every channel that carries it. `ls` is the line style, `m` the
#: marker, `h` the hatch for bar/area marks.
ENC = {
    "latency": {"c": LATENCY, "m": "o", "ls": "-",    "h": "///"},
    "rate":    {"c": RATE,    "m": "s", "ls": "--",   "h": "..."},
    "delta":   {"c": DELTA,   "m": "^", "ls": "-.",   "h": "\\\\\\"},
}
#: Fixed order wherever the three appear together, so legends never reshuffle.
ENC_ORDER = ("latency", "rate", "delta")

# --- sizes ------------------------------------------------------------------
COL_W, TEXT_W = 3.487, 7.16          # IEEEtran \columnwidth, \textwidth (in)
FS_AXIS, FS_TICK, FS_LEGEND = 8.5, 7.5, 7.5
FS_ANNOT, FS_LABEL = 7.5, 8.0        # 7.0 pt is the absolute floor
MS_ORDINARY, MS_LEADER = 28, 42      # scatter areas

RC = {
    "figure.dpi": 200, "savefig.dpi": 600,
    "savefig.bbox": None,            # exact width; see make_figures docstring
    "savefig.pad_inches": 0.0,
    "savefig.facecolor": "white", "figure.facecolor": "white",
    "font.family": "serif", "font.size": FS_ANNOT,
    "axes.titlesize": FS_LABEL, "axes.labelsize": FS_AXIS,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEGEND, "legend.frameon": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "grid.linestyle": "-", "grid.alpha": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": TEXT, "axes.linewidth": 0.6,
    "axes.labelcolor": TEXT, "text.color": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "hatch.linewidth": 0.45,
}
