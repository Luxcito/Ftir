"""Rebuild the FTIR comparison and noise-correction figures from raw .asp files.

Run from this folder:
    python plot_ftir.py
"""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("Agg")  # Save files without opening a plot window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ftir_tools import (
    SAMPLE_LABELS, absorbance, averages, load_scans, minimum_in_band,
    noise_proxy, prepare_scans, spectrum_difference,
)

DRY = "Dry silica 2 pooled"
COLORS = {
    "Dry silica 2 LHS": "#325e98",
    "Dry silica 2 LMS": "#d08a3c",
    "Silica Gel 5": "#288674",
    "Silica Gel 8": "#b35b76",
    DRY: "#343e50",
}


def format_axes(ax, x_min=4000, x_max=650, ylabel="Transmittance (%)"):
    ax.set_xlim(x_min, x_max)  # FTIR convention: higher wavenumber on left.
    ax.set_xlabel("Wavenumber (cm$^{-1}$)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2, linewidth=0.5)


def save_figure(fig, folder, name):
    fig.savefig(folder / name, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_four_means(x, means, folder):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for label in SAMPLE_LABELS:
        mean = means[label]
        color = COLORS[label]
        ax.plot(x, mean["smoothed"], color=color, lw=1.8,
                label=f"{label} (n={mean['n']})")
    format_axes(ax)
    ax.set_title("Averaged FTIR spectra — October 22, 2026")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    save_figure(fig, folder, "01_averaged_spectra.png")


def plot_dry_comparison(x, means, folder):
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 1]})
    top, bottom = axes
    lhs, lms = SAMPLE_LABELS[:2]
    for label in (lhs, lms):
        data = means[label]
        color = COLORS[label]
        y = data["smoothed"]
        top.plot(x, y, color=color, lw=1.8, label=f"{label} (n={data['n']})")
        top.fill_between(x, y - data["std"], y + data["std"],
                         color=color, alpha=0.13, linewidth=0)
    top.set_ylabel("Transmittance (%)")
    top.set_title("Dry silica 2 — LHS versus LMS (mean ± 1 sample SD)")
    top.legend(loc="lower left")
    top.grid(alpha=0.2)

    difference = spectrum_difference(means[lhs]["smoothed"], means[lms]["smoothed"])
    bottom.plot(x, difference, color="#555c66", lw=1.3)
    bottom.axhline(0, color="black", lw=0.7)
    format_axes(bottom, ylabel="LHS − LMS (%T points)")
    fig.tight_layout()
    save_figure(fig, folder, "02_dry_LHS_vs_LMS.png")


def plot_dry_vs_gels(x, means, folder):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for ax, gel in zip(axes, SAMPLE_LABELS[2:]):
        ax.plot(x, means[DRY]["smoothed"], color=COLORS[DRY], lw=1.7,
                label=f"Dry silica 2 pooled (n={means[DRY]['n']})")
        ax.plot(x, means[gel]["smoothed"], color=COLORS[gel], lw=1.7,
                label=f"{gel} (n={means[gel]['n']})")
        ax.legend(loc="lower left", fontsize=9)
        ax.set_title(f"{gel} compared with the pooled dry reference")
        ax.set_ylabel("Transmittance (%)")
        ax.grid(alpha=0.2)
    format_axes(axes[-1])
    fig.tight_layout()
    save_figure(fig, folder, "03_dry_vs_gels.png")


def plot_differences(x, means, folder):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for gel in SAMPLE_LABELS[2:]:
        delta = spectrum_difference(means[gel]["smoothed"], means[DRY]["smoothed"])
        rmse = float(np.sqrt(np.mean(delta ** 2)))
        ax.plot(x, delta, lw=1.6, color=COLORS[gel],
                label=f"{gel} − dry (RMS {rmse:.2f} %T points)")
    ax.axhline(0, color="black", lw=0.7)
    format_axes(ax, ylabel="Difference (%T points)")
    ax.set_title("Spectral differences relative to dry silica 2")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    save_figure(fig, folder, "04_differences_vs_dry.png")


def plot_absorbance(x, means, folder):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for label in (DRY,) + SAMPLE_LABELS[2:]:
        axes[0].plot(x, means[label]["smoothed"], lw=1.5,
                     color=COLORS[label], label=label)
        axes[1].plot(x, absorbance(means[label]["smoothed"]), lw=1.5,
                     color=COLORS[label], label=label)
    axes[0].set_title("Corrected spectra — transmittance and absorbance")
    axes[0].set_ylabel("Transmittance (%)")
    axes[1].set_ylabel("Absorbance (AU)")
    axes[1].axhline(0, color="black", lw=0.7)
    axes[0].legend(loc="lower left", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.2)
    format_axes(axes[-1], ylabel="Absorbance (AU)")
    fig.tight_layout()
    save_figure(fig, folder, "05_transmittance_and_absorbance.png")


def plot_noise_zooms(x, scans, means, folder):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2))
    regions = [
        ("Dry silica 2 LMS", 3800, 3600, "High-wavenumber fluctuations"),
        ("Silica Gel 8", 2150, 1850, "Mid-range fluctuations"),
        ("Silica Gel 8", 2410, 2280, "Possible atmospheric CO$_2$ feature"),
        ("Silica Gel 5", 1200, 850, "Silica fingerprint band"),
    ]
    for ax, (label, xmax, xmin, title) in zip(axes.flat, regions):
        # Raw and corrected means are enough to show what smoothing did;
        # both are calculated from the original scans in this project.
        ax.plot(x, means[label]["raw"], color="#8e9198", lw=1.0,
                alpha=0.8, label="Original mean")
        ax.plot(x, means[label]["smoothed"], color=COLORS[label], lw=1.7,
                label="Smoothed mean")
        ax.set_xlim(xmax, xmin)
        # Zoom the vertical scale to the selected region, not the whole scan.
        # Otherwise the small random fluctuations are almost invisible.
        inside = (x >= xmin) & (x <= xmax)
        local = np.r_[means[label]["raw"][inside], means[label]["smoothed"][inside]]
        spread = max(float(np.ptp(local)), 0.6)
        ax.set_ylim(float(local.min() - 0.12 * spread),
                    float(local.max() + 0.12 * spread))
        ax.set_title(f"{label}: {title}", fontsize=10)
        ax.set_xlabel("Wavenumber (cm$^{-1}$)")
        ax.set_ylabel("Transmittance (%)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Noise correction: original versus 15-point Savitzky–Golay", y=1.01)
    fig.tight_layout()
    save_figure(fig, folder, "06_noise_before_after.png")


def plot_noise_metrics(x, scans, folder):
    rows = []
    for label in SAMPLE_LABELS:
        group = [s for s in scans if s["label"] == label]
        raw = np.mean([noise_proxy(x, s["raw"], 3600, 3800) for s in group])
        smooth = np.mean([noise_proxy(x, s["smoothed"], 3600, 3800) for s in group])
        rows.append((label, raw, smooth))

    fig, ax = plt.subplots(figsize=(10, 4.6))
    centers = np.arange(len(rows))
    ax.bar(centers - 0.18, [r[1] for r in rows], 0.36, label="Original")
    ax.bar(centers + 0.18, [r[2] for r in rows], 0.36, label="Smoothed")
    ax.set_xticks(centers, [r[0] for r in rows], rotation=12, ha="right")
    ax.set_ylabel("Point-to-point variation (%T)")
    ax.set_title("Change in short-scale variation, 3600–3800 cm$^{-1}$")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, folder, "07_noise_metrics.png")


def plot_diagnostic_bands(x, means, folder):
    bands = [
        ("O–H region", 3200, 3550),
        ("H$_2$O region", 1600, 1680),
        ("2478 region", 2400, 2550),
        ("Silica fingerprint", 1000, 1120),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    positions = np.arange(len(bands))
    width = 0.2
    for i, label in enumerate(SAMPLE_LABELS):
        values = []
        for _, lo, hi in bands:
            _, minimum = minimum_in_band(x, means[label]["smoothed"], lo, hi)
            values.append(float(absorbance(minimum)))
        ax.bar(positions + (i - 1.5) * width, values, width,
               label=label, color=COLORS[label])
    ax.set_xticks(positions, [b[0] for b in bands])
    ax.set_ylabel("Absorbance at local minimum (AU)")
    ax.set_title("Selected spectral regions (not elemental concentrations)")
    ax.legend(fontsize=9)
    ax.axhline(0, color="black", lw=0.65)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, folder, "08_selected_bands.png")


def export_tables(x, scans, means, folder):
    mean_table = pd.DataFrame({"wavenumber_cm-1": x})
    for label in (*SAMPLE_LABELS, DRY):
        for key in ("raw", "smoothed"):
            mean_table[f"{label} {key} %T"] = means[label][key]
        mean_table[f"{label} absorbance AU"] = absorbance(means[label]["smoothed"])
    mean_table.to_csv(folder / "averaged_spectra.csv", index=False, float_format="%.9f")

    qc_rows = []
    for label in SAMPLE_LABELS:
        group = [s for s in scans if s["label"] == label]
        before = np.mean([noise_proxy(x, s["raw"], 3600, 3800) for s in group])
        after = np.mean([noise_proxy(x, s["smoothed"], 3600, 3800) for s in group])
        peak_x, peak_t = minimum_in_band(x, means[label]["smoothed"], 800, 1150)
        qc_rows.append({
            "sample": label,
            "scans": len(group),
            "main_band_cm-1": round(peak_x, 3),
            "main_band_percent_T": round(peak_t, 3),
            "noise_proxy_original_percent_T": round(before, 5),
            "noise_proxy_smoothed_percent_T": round(after, 5),
            "variation_reduction_percent": round(100 * (1 - after / before), 2),
            "fraction_of_avg_values_over_100_percent_T": round(
                np.mean(means[label]["smoothed"] > 100), 4),
        })
    pd.DataFrame(qc_rows).to_csv(folder / "noise_and_peak_summary.csv", index=False)

    dry = means[DRY]["smoothed"]
    comparisons = []
    for label in SAMPLE_LABELS[2:]:
        diff = spectrum_difference(means[label]["smoothed"], dry)
        comparisons.append({"sample": label,
                            "rms_difference_percent_T_points": float(np.sqrt(np.mean(diff**2)))})
    pd.DataFrame(comparisons).to_csv(folder / "gel_vs_dry_differences.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})

    x, scans = load_scans(args.input_dir)
    prepare_scans(scans)
    means = averages(scans)

    plot_four_means(x, means, args.figures_dir)
    plot_dry_comparison(x, means, args.figures_dir)
    plot_dry_vs_gels(x, means, args.figures_dir)
    plot_differences(x, means, args.figures_dir)
    plot_absorbance(x, means, args.figures_dir)
    plot_noise_zooms(x, scans, means, args.figures_dir)
    plot_noise_metrics(x, scans, args.figures_dir)
    plot_diagnostic_bands(x, means, args.figures_dir)
    export_tables(x, scans, means, args.results_dir)
    print(f"Processed {len(scans)} scans: " + ", ".join(
        f"{label} (n={means[label]['n']})" for label in SAMPLE_LABELS))
    print(f"Saved 8 graphs to {args.figures_dir} and CSV results to {args.results_dir}")


if __name__ == "__main__":
    main()
