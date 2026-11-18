"""
TremorVision v2 - tremor frequency analyzer (research prototype, NOT a diagnostic tool)

Usage:
    python tremor_analysis_v2.py --condition rest            # analyze tremor_data.csv
    python tremor_analysis_v2.py --condition postural --no-plot
    python tremor_analysis_v2.py --selftest                  # validate on synthetic signals

Reference frequency ranges used (literature, verify against your cited guide):
    Parkinsonian rest tremor      : ~4-6 Hz (borderline 3-7 Hz), present at REST
    Essential tremor              : ~4-12 Hz, postural / kinetic
    (Enhanced) physiological      : ~6-12 Hz, low amplitude, postural
    Orthostatic tremor            : 13-18 Hz (cannot be measured with a 30 fps camera)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

CSV_FILE = Path(__file__).resolve().parent / "tremor_data.csv"

# ---------------- configuration ----------------
ASPECT_RATIO = 1280 / 720        # MediaPipe x is normalized by width, y by height -> make them comparable
HAND_SIZE_CM = 9.5               # assumed wrist -> middle-finger MCP length (ESTIMATE, calibrate per person)
SEARCH_BAND = (2.0, 12.0)        # where we look for the peak (wide so a true <3 Hz peak is not clipped at the edge)
FILTER_BAND = (2.0, 12.0)        # band-pass to remove voluntary drift and jitter
MIN_DURATION_S = 8.0             # minimum clean segment length
MIN_SAMPLES = 32
# Reliability thresholds (heuristic - must be tuned on real labeled data)
MIN_CONCENTRATION = 0.40         # share of in-band power within +/-0.5 Hz of the peak
MIN_PEAK_SNR = 6.0               # peak power / median in-band power
# "Steady hand" verdict (provisional - calibrate with recordings of steady hands)
STEADY_MAX_RMS_CM = 0.12         # band-limited movement below this = no significant tremor
STEADY_MIN_DURATION_S = 8.0
STEADY_MIN_FS_HZ = 20.0


# ---------------- 1. load ----------------
def load_data(path):
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    def pick(names):
        for n in names:
            if n in df.columns:
                return n
        raise ValueError(f"Missing column {names}. Found: {list(df.columns)}")

    out = pd.DataFrame({
        "t": pd.to_numeric(df[pick(["time_seconds", "time", "timestamp"])], errors="coerce"),
        "x": pd.to_numeric(df[pick(["wrist_x", "x"])], errors="coerce"),
        "y": pd.to_numeric(df[pick(["wrist_y", "y"])], errors="coerce"),
    })
    for extra in ("hand_size", "aspect_ratio"):   # written by tremor_recorder_v2.py
        if extra in df.columns:
            out[extra] = pd.to_numeric(df[extra], errors="coerce")
    if "hand_detected" in df.columns:
        out = out[pd.to_numeric(df["hand_detected"], errors="coerce").fillna(0) == 1]
    out = (out.replace([np.inf, -np.inf], np.nan)
              .dropna(subset=["t", "x", "y"])
              .sort_values("t").drop_duplicates("t").reset_index(drop=True))
    if len(out) < MIN_SAMPLES:
        raise ValueError(f"Only {len(out)} valid samples. Record longer with the hand visible.")
    return out


# ---------------- 2. longest continuous segment ----------------
def longest_segment(df):
    dt = np.diff(df["t"].to_numpy())
    med = float(np.median(dt[dt > 0]))
    gap = max(0.15, 2.5 * med)
    parts = np.split(np.arange(len(df)), np.where(dt > gap)[0] + 1)
    parts = [p for p in parts if len(p) >= MIN_SAMPLES]
    if not parts:
        raise ValueError("No continuous segment long enough (hand lost too often).")
    seg = df.iloc[max(parts, key=len)].reset_index(drop=True)
    t0, t1 = seg["t"].iloc[0], seg["t"].iloc[-1]
    trimmed = seg[(seg["t"] >= t0 + 1.0) & (seg["t"] <= t1 - 1.0)]
    return (trimmed if len(trimmed) >= MIN_SAMPLES else seg).reset_index(drop=True)


# ---------------- 3. analysis ----------------
def analyze(seg):
    t = seg["t"].to_numpy(float)
    dt = np.diff(t)
    med = float(np.median(dt))
    fs = 1.0 / med
    duration = float(t[-1] - t[0])
    aspect = float(np.nanmedian(seg["aspect_ratio"])) if "aspect_ratio" in seg.columns else ASPECT_RATIO
    if not np.isfinite(aspect):
        aspect = ASPECT_RATIO
    dropped = float(np.mean(dt > 1.5 * med))          # share of late frames (timing jitter)

    tu = np.arange(t[0], t[-1], 1.0 / fs)
    x = np.interp(tu, t, seg["x"].to_numpy(float) * aspect)
    y = np.interp(tu, t, seg["y"].to_numpy(float))

    # band-pass (zero-phase) instead of only linear detrend
    hi = min(FILTER_BAND[1], 0.45 * fs)
    sos = signal.butter(4, [FILTER_BAND[0], hi], btype="bandpass", fs=fs, output="sos")
    xf = signal.sosfiltfilt(sos, x - x.mean())
    yf = signal.sosfiltfilt(sos, y - y.mean())

    # project on the principal axis of motion (tremor is not always vertical)
    m = np.vstack([xf, yf])
    _, vecs = np.linalg.eigh(np.cov(m))
    s = vecs[:, -1] @ m

    nperseg = min(len(s), int(5 * fs))
    f, p = signal.welch(s, fs=fs, window="hann", nperseg=nperseg,
                        noverlap=nperseg // 2, nfft=4096)
    band = (f >= SEARCH_BAND[0]) & (f <= min(SEARCH_BAND[1], hi))
    fb, pb = f[band], p[band]
    i = int(np.argmax(pb))
    peak_f = float(fb[i])
    near = np.abs(fb - peak_f) <= 0.5
    concentration = float(pb[near].sum() / pb.sum())
    snr = float(pb[i] / max(np.median(pb), 1e-18))

    rms = float(np.sqrt(np.mean(s ** 2)))
    hs = float(np.nanmedian(seg["hand_size"])) if "hand_size" in seg.columns else None
    if hs is not None and not np.isfinite(hs):
        hs = None
    rms_cm = rms / hs * HAND_SIZE_CM if hs else None

    warnings = []
    if duration < MIN_DURATION_S:
        warnings.append(f"Segment only {duration:.1f}s (< {MIN_DURATION_S}s).")
    if fs < 24:
        warnings.append(f"Sample rate {fs:.1f} Hz is low; upper band is unreliable.")
    if dropped > 0.10:
        warnings.append(f"{dropped*100:.0f}% of frames arrived late (timing jitter).")
    at_edge = (peak_f - fb[0] < 0.3) or (fb[-1] - peak_f < 0.3)
    if at_edge:
        warnings.append(f"Peak at {peak_f:.2f} Hz sits on the edge of the search band; true frequency may be outside it.")
    reliable = (concentration >= MIN_CONCENTRATION and snr >= MIN_PEAK_SNR
                and duration >= MIN_DURATION_S and not at_edge)

    return dict(fs=fs, duration=duration, freq=peak_f, concentration=concentration, snr=snr,
                rms=rms, rms_cm=rms_cm, reliable=reliable, warnings=warnings,
                t=tu, x=x, y=y, s=s, f=f, p=p)


# ---------------- 4. interpretation ----------------
def is_steady(r):
    """Valid recording + tiny 2-12 Hz movement + no rhythmic peak = no significant tremor."""
    return (not r["reliable"] and r["rms_cm"] is not None
            and r["rms_cm"] < STEADY_MAX_RMS_CM
            and r["duration"] >= STEADY_MIN_DURATION_S
            and r["fs"] >= STEADY_MIN_FS_HZ)


def interpret(r, condition):
    if is_steady(r):
        return ("NO SIGNIFICANT TREMOR DETECTED",
                f"Hand movement in the 2-12 Hz tremor band was very small (about {r['rms_cm']:.2f} cm RMS) "
                "and showed no rhythmic peak. This is consistent with a steady hand during this test.",
                "No action needed from this result. One short test cannot rule out a movement disorder; "
                "if you have symptoms, see a doctor.")
    if not r["reliable"]:
        return ("INCONCLUSIVE",
                "No clear, stable tremor peak was found (or the recording is too short/noisy).",
                "Repeat: hand fully visible, good light, 10-20 s, same posture.")
    f = r["freq"]
    if condition == "rest":
        if 4.0 <= f <= 6.0:
            return ("REST TREMOR IN PARKINSONIAN FREQUENCY RANGE (4-6 Hz)",
                    "Frequency is typical of parkinsonian rest tremor. Frequency alone cannot confirm it.",
                    "Clinical neurological assessment is recommended.")
        if 3.0 <= f < 4.0 or 6.0 < f <= 7.0:
            return ("BORDERLINE REST TREMOR FREQUENCY",
                    "Slightly outside the classic 4-6 Hz range; other causes are possible.",
                    "Repeat the test; consider clinical assessment if symptoms persist.")
        return ("REST TREMOR OUTSIDE PARKINSONIAN RANGE",
                "Frequency is atypical for parkinsonian rest tremor (consider physiological/essential tremor, "
                "medication, caffeine, anxiety, thyroid...).",
                "Repeat the test; discuss with a doctor if persistent.")
    # postural / kinetic
    if 4.0 <= f < 8.0:
        return ("POSTURAL/KINETIC TREMOR 4-8 Hz",
                "Compatible with essential tremor (also seen as re-emergent tremor in Parkinson's).",
                "Compare with a REST recording; clinical assessment if persistent.")
    if 8.0 <= f <= 12.0:
        return ("POSTURAL TREMOR 8-12 Hz",
                "Typical of physiological / enhanced physiological tremor, especially if amplitude is small.",
                "Review caffeine, stress, sleep, medication; repeat later.")
    return ("FREQUENCY OUTSIDE EXPECTED RANGE",
            "Peak is outside the usual tremor ranges for this posture.",
            "Check recording quality and repeat.")


def print_report(r, condition):
    title, meaning, step = interpret(r, condition)
    print("\n" + "=" * 58)
    print("TREMOR VISION - REPORT   (condition:", condition + ")")
    print("=" * 58)
    print(f"Sample rate       : {r['fs']:.1f} Hz   Duration: {r['duration']:.1f} s")
    if r["reliable"]:
        print(f"Dominant freq     : {r['freq']:.2f} Hz")
    else:
        print(f"Dominant freq     : --  (strongest bin {r['freq']:.2f} Hz, NOT a reliable peak)")
    print(f"Peak concentration: {r['concentration']:.2f}   Peak SNR: {r['snr']:.1f}   "
          f"Reliable: {r['reliable']}")
    amp = f"{r['rms']:.5f} (image-height units, uncalibrated)"
    if r["rms_cm"] is not None:
        amp += f"  ~ {r['rms_cm']:.3f} cm RMS (estimated)"
    print(f"Amplitude (RMS)   : {amp}")
    for w in r["warnings"]:
        print("WARNING:", w)
    print(f"\nRESULT : {title}\nMEANING: {meaning}\nNEXT   : {step}")
    print("-" * 58)
    print("Research prototype - not a medical diagnosis.")


def show_plots(r):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(11, 9))
    ax[0].plot(r["t"], r["x"], label="x (aspect-corrected)")
    ax[0].plot(r["t"], r["y"], label="y")
    ax[0].set_title("Raw wrist coordinates"); ax[0].legend()
    ax[1].plot(r["t"], r["s"])
    ax[1].set_title("Band-passed motion on principal axis")
    ax[2].semilogy(r["f"], r["p"])
    ax[2].axvspan(*SEARCH_BAND, alpha=0.15)
    ax[2].axvline(r["freq"], color="r", ls="--")
    ax[2].set_xlim(0, 15); ax[2].set_title("Welch PSD")
    for a in ax:
        a.grid(True)
    plt.tight_layout(); plt.show()


# ---------------- 5. self-test ----------------
def synth(freq, amp=0.01, noise=0.001, seconds=15, fps=30, seed=0):
    rng = np.random.default_rng(seed)
    t = np.cumsum(np.full(int(seconds * fps), 1 / fps) + rng.normal(0, 0.003, int(seconds * fps)))
    drift = 0.05 * np.sin(2 * np.pi * 0.3 * t)                       # slow voluntary drift
    tremor = amp * np.sin(2 * np.pi * freq * t) if freq else 0 * t
    y = 0.5 + drift + tremor + rng.normal(0, noise, t.size)
    x = 0.5 + 0.3 * drift + rng.normal(0, noise, t.size)
    return pd.DataFrame({"t": t, "x": x, "y": y})


def selftest():
    ok = True
    for f in (4.0, 5.0, 6.0, 8.0, 10.0):
        r = analyze(synth(f))
        good = abs(r["freq"] - f) <= 0.3 and r["reliable"]
        ok &= good
        print(f"true {f:5.1f} Hz -> measured {r['freq']:.2f} Hz  reliable={r['reliable']}  "
              f"{'PASS' if good else 'FAIL'}")
    r = analyze(synth(None, noise=0.003))
    good = not r["reliable"]
    ok &= good
    print(f"noise only       -> reliable={r['reliable']} (expected False)  {'PASS' if good else 'FAIL'}")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return ok


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=CSV_FILE)
    ap.add_argument("--condition", choices=["rest", "postural", "kinetic"], default=None,
                    help="default: read from the CSV (written by the recorder), else rest")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if selftest() else 1)
    raw = pd.read_csv(a.csv, usecols=lambda c: c.strip().lower() == "condition") if a.csv.exists() else None
    from_csv = str(raw.iloc[0, 0]).strip().lower() if raw is not None and len(raw) else None
    condition = a.condition or (from_csv if from_csv in ("rest", "postural", "kinetic") else "rest")
    seg = longest_segment(load_data(a.csv))
    r = analyze(seg)
    print_report(r, condition)
    if not a.no_plot:
        show_plots(r)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print("\nTREMOR VISION ERROR:", e)