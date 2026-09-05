#!/usr/bin/env python3
"""Run generate_paper_fig14.py's compute path to reproduce the vision/automobile
AVG DECREASE numbers, WITHOUT requiring seaborn (only used for cosmetic barplot).
We stub seaborn so the module imports; the GA computation + the printed
"Mozart AVG DECREASE" lines are unaffected. Figure goes to a scratch path.
"""
import os, sys, types, importlib.util

# --- stub seaborn (cosmetic only) ---
_sns = types.ModuleType("seaborn")
_sns.set_palette = lambda *a, **k: None
_sns.barplot = lambda *a, **k: None
_sns.set_theme = lambda *a, **k: None
sys.modules["seaborn"] = _sns

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "generate_paper_fig14", os.path.join(HERE, "generate_paper_fig14.py"))
m = importlib.util.module_from_spec(spec)
sys.modules["generate_paper_fig14"] = m
spec.loader.exec_module(m)

m.plot_figure14("/tmp/fig14_repro.pdf")
