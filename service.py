"""
service.py - bridge between the web app and the analyzer.

Takes a recording CSV, runs the analyzer we already built and returns a plain
dictionary (JSON-safe) that the web pages can display and plot.
The analyzer itself is NOT modified.
"""
import importlib
import math
from pathlib import Path

import numpy as np


def _load_analyzer():
    # your file name first, then the name I gave it
    for name in ("analyz_tremor", "tremor_analysis_v2"):
        try:
            m = importlib.import_module(name)
            if hasattr(m, "analyze") and hasattr(m, "longest_segment"):
                return m
        except ImportError:
            continue
    raise ImportError("Analyzer not found. Put analyz_tremor.py next to app.py.")


T = _load_analyzer()

MAX_WAVE_POINTS = 600       # keep the response small
MAX_SPECTRUM_HZ = 15.0


def _num(v, digits=4):
    """float -> JSON-safe (NaN/inf become None)."""
    if v is None:
        return None
    v = float(v)
    return round(v, digits) if math.isfinite(v) else None


def _downsample(t, y, n=MAX_WAVE_POINTS):
    if len(t) <= n:
        idx = np.arange(len(t))
    else:
        idx = np.linspace(0, len(t) - 1, n).astype(int)
    return [_num(t[i] - t[0], 3) for i in idx], [_num(y[i], 5) for i in idx]


def run_analysis(csv_path, condition="rest"):
    """Analyze one recording file and return everything the pages need."""
    seg = T.longest_segment(T.load_data(Path(csv_path)))
    r = T.analyze(seg)
    title, meaning, step = T.interpret(r, condition)

    if r["reliable"]:
        status = "tremor"
    elif T.is_steady(r):
        status = "steady"
    else:
        status = "inconclusive"

    wave_t, wave_y = _downsample(r["t"], r["s"])
    keep = r["f"] <= MAX_SPECTRUM_HZ
    spec_f = [_num(v, 3) for v in r["f"][keep]]
    spec_p = [_num(v, 12) for v in r["p"][keep]]

    return {
        "condition": condition,
        "status": status,
        "reliable": bool(r["reliable"]),
        # only expose a frequency when it is a trustworthy peak
        "frequency_hz": _num(r["freq"], 2) if r["reliable"] else None,
        "strongest_bin_hz": _num(r["freq"], 2),
        "concentration": _num(r["concentration"], 2),
        "snr": _num(r["snr"], 1),
        "duration_s": _num(r["duration"], 1),
        "sample_rate_hz": _num(r["fs"], 1),
        "rms": _num(r["rms"], 5),
        "rms_cm": _num(r["rms_cm"], 3),
        "warnings": list(r["warnings"]),
        "result_title": title,
        "result_meaning": meaning,
        "result_next_step": step,
        "waveform": {"t": wave_t, "y": wave_y},
        "spectrum": {"f": spec_f, "p": spec_p},
        "disclaimer": "Research prototype - not a medical diagnosis.",
    }