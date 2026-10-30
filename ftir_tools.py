"""Small utilities for reading and processing the Oct. 22 Agilent FTIR scans."""

from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

# These are labels from the file names, not independently verified compositions.
SAMPLE_LABELS = (
    "Dry silica 2 LHS",
    "Dry silica 2 LMS",
    "Silica Gel 5",
    "Silica Gel 8",
)


def label_from_filename(filename):
    name = Path(filename).name
    for prefix, label in (
        ("Silla Gel 2 LHS_", SAMPLE_LABELS[0]),
        ("Silla Gel 2 LMS_", SAMPLE_LABELS[1]),
        ("Silla Gel 5_", SAMPLE_LABELS[2]),
        ("Silla Gel 8_", SAMPLE_LABELS[3]),
    ):
        if name.startswith(prefix):
            return label
    raise ValueError(f"Unrecognized sample label in {name!r}")


def read_asp(path):
    """Read an Agilent .asp text export with a six-line numeric header.

    The first line is point count, the next two are starting and ending
    wavenumbers, and transmittance starts at line seven. Header fields
    four through six are retained by the source file but not used here.
    """
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 7:
        raise ValueError(f"Not enough lines in {path.name}")

    try:
        count = int(lines[0].strip())
        start = float(lines[1].strip())
        stop = float(lines[2].strip())
        values = np.array([float(line.strip()) for line in lines[6:] if line.strip()])
    except ValueError as exc:
        raise ValueError(f"Invalid numeric ASP content in {path.name}") from exc

    if count != len(values):
        raise ValueError(f"{path.name}: header says {count} points, found {len(values)}")
    if count < 15 or not start > stop:
        raise ValueError(f"{path.name}: unexpected wavenumber range or point count")
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"{path.name}: values must be finite and positive")

    wavenumber = np.linspace(start, stop, count)
    return wavenumber, values


def load_scans(folder):
    """Return shared wavenumber grid and list of scan dictionaries."""
    paths = sorted(Path(folder).glob("*.asp"))
    if not paths:
        raise FileNotFoundError(f"No .asp files found in {folder}")

    reference = None
    scans = []
    for path in paths:
        x, transmittance = read_asp(path)
        if reference is None:
            reference = x
        elif len(x) != len(reference) or not np.allclose(x, reference, atol=0.001, rtol=0):
            raise ValueError(f"Wavenumber grid mismatch in {path.name}; do not average without alignment")
        scans.append({
            "filename": path.name,
            "label": label_from_filename(path),
            "raw": transmittance,
        })
    return reference, scans


def smooth_transmittance(values, window=15, degree=3):
    """Apply light smoothing without doing baseline or atmospheric correction."""
    if window % 2 == 0 or window <= degree:
        raise ValueError("Savitzky-Golay window must be odd and larger than degree")
    if window > len(values):
        raise ValueError("Smoothing window is longer than the scan")
    return savgol_filter(values, window_length=window, polyorder=degree, mode="interp")


def prepare_scans(scans, window=15, degree=3):
    for scan in scans:
        scan["smoothed"] = smooth_transmittance(scan["raw"], window, degree)
    return scans


def averages(scans):
    """Average scans equally *within each label*, plus one five-scan dry mean."""
    answer = {}
    for label in SAMPLE_LABELS:
        group = [scan for scan in scans if scan["label"] == label]
        if not group:
            raise ValueError(f"Missing scans for {label}")
        answer[label] = {
            "n": len(group),
            "raw": np.mean([scan["raw"] for scan in group], axis=0),
            "smoothed": np.mean([scan["smoothed"] for scan in group], axis=0),
            "std": np.std([scan["smoothed"] for scan in group], axis=0, ddof=1)
            if len(group) > 1 else np.zeros_like(group[0]["raw"]),
        }

    dry = [scan for scan in scans if scan["label"] in SAMPLE_LABELS[:2]]
    answer["Dry silica 2 pooled"] = {
        "n": len(dry),
        "raw": np.mean([scan["raw"] for scan in dry], axis=0),
        "smoothed": np.mean([scan["smoothed"] for scan in dry], axis=0),
        "std": np.std([scan["smoothed"] for scan in dry], axis=0, ddof=1),
    }
    return answer


def absorbance(transmittance_percent):
    """A = -log10(T/100); negative values are preserved when %T > 100."""
    values = np.asarray(transmittance_percent, dtype=float)
    if np.any(values <= 0):
        raise ValueError("Cannot calculate absorbance from nonpositive %T")
    return -np.log10(values / 100.0)


def noise_proxy(wavenumber, spectrum, low, high):
    """Point-to-point variation proxy, not calibrated instrument noise."""
    selected = spectrum[(wavenumber >= low) & (wavenumber <= high)]
    return float(np.std(np.diff(selected), ddof=1) / np.sqrt(2))


def minimum_in_band(wavenumber, spectrum, low, high):
    mask = (wavenumber >= low) & (wavenumber <= high)
    if not mask.any():
        raise ValueError(f"Band {low}-{high} cm-1 falls outside the scan")
    positions = np.where(mask)[0]
    best = positions[np.argmin(spectrum[mask])]
    return float(wavenumber[best]), float(spectrum[best])


def spectrum_difference(a, b):
    """Difference in percentage points of transmittance (a minus b)."""
    return np.asarray(a) - np.asarray(b)
