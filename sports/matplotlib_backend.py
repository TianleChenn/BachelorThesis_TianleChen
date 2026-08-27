"""Headless Matplotlib configuration for protected backend analyses."""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

# This module is imported by plotting modules before pyplot.  Agg keeps batch
# evaluation and Windows services from allocating GUI window bitmaps.
matplotlib.use("Agg", force=True)

