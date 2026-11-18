"""Compare wrist X and Y movement from tremor_data.csv.

Run this file from any working directory. It reads the CSV beside this script
and only displays a plot; it does not modify or create data files.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lombscargle


CSV_PATH = Path(__file__).resolve().parent / "tremor_data.csv"
REQUIRED_COLUMNS = ["time_seconds", "wrist_x", "wrist_y"]
FREQUENCY_LOW_HZ = 3.0
FREQUENCY_HIGH_HZ = 12.0


def stop(message):
    print(f"ERROR: {message}")
    raise SystemExit(1)


def remove_linear_trend(times, values):
    """Remove gradual drift by fitting and subtracting a straight line."""
    centered_time = times - times.mean()
    slope, intercept = np.polyfit(centered_time, values, 1)
    return values - (slope * centered_time + intercept)


def estimate_peak_frequency(times, detrended_values, max_frequency):
    """Estimate a spectral peak directly from possibly uneven timestamps."""
    duration = times[-1] - times[0]
    if duration <= 0:
        return None, None

    low = FREQUENCY_LOW_HZ
    high = min(FREQUENCY_HIGH_HZ, max_frequency)
    if high <= low:
        return None, None

    # Frequency spacing is finer than the recording's basic 1/duration scale.
    frequency_step = 1.0 / (4.0 * duration)
    frequencies = np.arange(low, high + frequency_step / 2, frequency_step)
    angular_frequencies = 2.0 * np.pi * frequencies
    power = lombscargle(
        times - times[0],
        detrended_values,
        angular_frequencies,
        normalize=True,
    )
    peak_index = int(np.argmax(power))
    return float(frequencies[peak_index]), float(power[peak_index])


def main():
    if not CSV_PATH.exists():
        stop(f"Could not find {CSV_PATH.name} beside this script.")

    try:
        data = pd.read_csv(CSV_PATH)
    except (OSError, pd.errors.ParserError) as error:
        stop(f"Could not read {CSV_PATH.name}: {error}")

    missing_columns = [name for name in REQUIRED_COLUMNS if name not in data.columns]
    if missing_columns:
        stop(
            "The CSV is missing required column(s): "
            + ", ".join(missing_columns)
            + ". Required columns are time_seconds, wrist_x, wrist_y."
        )
    if data.empty:
        stop("The CSV has a header but no data rows.")

    original_count = len(data)
    data = data[REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=REQUIRED_COLUMNS)
    removed_count = original_count - len(data)
    if len(data) < 10:
        stop(
            f"Only {len(data)} usable samples remain after removing invalid rows; "
            "at least 10 are needed for this exploratory comparison."
        )

    data = data.sort_values("time_seconds", kind="stable")
    duplicate_count = int(data.duplicated(subset="time_seconds").sum())
    if duplicate_count:
        # Average simultaneous detections rather than arbitrarily choosing one.
        data = data.groupby("time_seconds", as_index=False)[["wrist_x", "wrist_y"]].mean()
    if len(data) < 10:
        stop("Fewer than 10 distinct timestamps remain after handling duplicates.")

    times = data["time_seconds"].to_numpy(dtype=float)
    intervals = np.diff(times)
    if np.any(intervals <= 0):
        stop("Timestamps must increase after duplicate timestamps are combined.")

    duration = float(times[-1] - times[0])
    if duration <= 0:
        stop("The recording duration must be greater than zero.")

    median_interval = float(np.median(intervals))
    mean_interval = float(np.mean(intervals))
    sampling_rate = 1.0 / median_interval
    mean_rate = 1.0 / mean_interval
    max_gap = float(np.max(intervals))
    gap_threshold = 2.0 * median_interval
    large_gaps = int(np.sum(intervals > gap_threshold))
    # This is only a cautious upper search bound for irregularly sampled data.
    search_high = min(FREQUENCY_HIGH_HZ, sampling_rate / 2.0)

    print("TremorVision — X/Y comparison (exploratory)")
    print("=" * 47)
    print(f"File: {CSV_PATH.name}")
    print(f"Usable samples: {len(data)} (from {original_count} rows)")
    print(f"Invalid rows removed: {removed_count}")
    print(f"Duplicate timestamps combined: {duplicate_count}")
    print(f"Recording duration: {duration:.2f} seconds")
    print(f"Median frame interval: {median_interval:.4f} seconds")
    print(f"Estimated sampling rate (median interval): {sampling_rate:.2f} samples/second")
    print(f"Mean-interval rate (for comparison): {mean_rate:.2f} samples/second")
    print(f"Largest interval: {max_gap:.4f} seconds")
    print(f"Intervals over twice the median: {large_gaps}")
    if large_gaps:
        print("WARNING: Missing frames or tracking interruptions may affect frequency estimates.")
    if sampling_rate / 2.0 < FREQUENCY_HIGH_HZ:
        print(
            f"WARNING: The estimated Nyquist limit is about {sampling_rate / 2.0:.2f} Hz; "
            f"the search band is limited to that value."
        )

    results = {}
    for axis in ("wrist_x", "wrist_y"):
        values = data[axis].to_numpy(dtype=float)
        detrended = remove_linear_trend(times, values)
        rms = float(np.sqrt(np.mean(np.square(detrended))))
        peak_hz, peak_power = estimate_peak_frequency(times, detrended, search_high)
        results[axis] = {"values": values, "detrended": detrended, "rms": rms}
        label = "X (horizontal)" if axis == "wrist_x" else "Y (vertical)"
        print(f"\n{label} position (normalized image coordinates):")
        print(f"  Mean: {np.mean(values):.5f}")
        print(f"  Range: {np.ptp(values):.5f}")
        print(f"  Detrended movement RMS: {rms:.5f} coordinate units")
        if peak_hz is None:
            print("  Dominant frequency: not estimated (insufficient frequency range)")
        else:
            print(f"  Strongest peak in searched band: {peak_hz:.2f} Hz")
            print(f"  Normalized Lomb–Scargle peak strength: {peak_power:.3f}")

    x_rms = results["wrist_x"]["rms"]
    y_rms = results["wrist_y"]["rms"]
    print("\nHow to read this comparison:")
    print("- RMS compares the size of the detrended image-coordinate variation on each axis.")
    print("- It does not establish that either axis is a true physical hand displacement.")
    print("- Different peak frequencies can reflect noise, mixed movement, or tracking errors.")
    if x_rms > y_rms:
        print("- X varies more than Y in this recording; this alone does not mean X is more meaningful.")
    elif y_rms > x_rms:
        print("- Y varies more than X in this recording; this alone does not mean Y is more meaningful.")
    else:
        print("- X and Y have equal detrended RMS in this recording.")
    print("- Coordinates are normalized to the image, not calibrated physical units such as millimeters.")
    print("- A webcam-only estimate is exploratory and is not a medical diagnosis.")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for axis, key, title, color in (
        (axes[0], "wrist_x", "Wrist X — horizontal image position", "tab:blue"),
        (axes[1], "wrist_y", "Wrist Y — vertical image position", "tab:orange"),
    ):
        axis.plot(times, results[key]["values"], color=color, linewidth=1.0, label="Recorded position")
        axis.plot(times, results[key]["detrended"], color="tab:green", alpha=0.8,
                  linewidth=0.9, label="After removing linear drift")
        axis.set_title(title)
        axis.set_ylabel("Normalized image coordinate")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle("TremorVision — Wrist movement by axis", fontsize=14)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
