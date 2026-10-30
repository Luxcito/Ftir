"""Exploratory FTIR classifier using nine easy-to-inspect spectral regions.

Run after making the plots:
    python classify_ftir.py

The leave-one-scan-out score is *not* an independent-sample accuracy estimate.
"""

from pathlib import Path
import argparse
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ftir_tools import SAMPLE_LABELS, load_scans, prepare_scans

# Fixed before looking at the classification results. Each feature is the
# mean percent transmittance in a spectral region, NOT a measured element.
BANDS = [
    ("O-H broad", 3200, 3550),
    ("Water bending region", 1600, 1680),
    ("2478 feature", 2400, 2550),
    ("CO2 region", 2300, 2370),
    ("Silica 1000-1120", 1000, 1120),
    ("Dry silica 900-950", 900, 950),
    ("Silica 780-820", 780, 820),
    ("1250-1400", 1250, 1400),
    ("1140-1230", 1140, 1230),
]


def band_features(x, spectrum, bands=BANDS):
    features = []
    for _name, low, high in bands:
        mask = (x >= low) & (x <= high)
        if not mask.any():
            raise ValueError(f"Spectrum does not cover {low}-{high} cm-1")
        features.append(float(np.mean(spectrum[mask])))
    return features


def make_model():
    # The scaler is inside the pipeline so it is fitted on the training
    # folds only; otherwise cross-validation would leak information.
    return make_pipeline(StandardScaler(), NearestCentroid())


def show_confusion(actual, predicted, filename, title):
    matrix = confusion_matrix(actual, predicted, labels=SAMPLE_LABELS)
    fig, ax = plt.subplots(figsize=(8.3, 6.6))
    display = ConfusionMatrixDisplay(matrix, display_labels=[
        "Dry 2 LHS", "Dry 2 LMS", "Gel 5", "Gel 8"
    ])
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return matrix


def show_pca(x_features, labels, filename):
    # PCA is for visualization only. It uses all 12 scans and must not
    # be presented as a separate classification validation.
    standardized = StandardScaler().fit_transform(x_features)
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(standardized)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for label in SAMPLE_LABELS:
        positions = [i for i, observed in enumerate(labels) if observed == label]
        ax.scatter(coordinates[positions, 0], coordinates[positions, 1],
                   s=65, alpha=0.85, label=label)
    ax.set_xlabel(f"Principal component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"Principal component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA of smoothed FTIR band features (descriptive)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(filename, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return pca.explained_variance_ratio_.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    wavenumber, scans = load_scans(args.input_dir)
    prepare_scans(scans, window=15, degree=3)
    labels = np.array([scan["label"] for scan in scans])
    names = [scan["filename"] for scan in scans]
    comparisons = []
    records = []
    for version in ("raw", "smoothed"):
        features = np.array([band_features(wavenumber, scan[version]) for scan in scans])
        predictions = cross_val_predict(make_model(), features, labels, cv=LeaveOneOut())
        score = accuracy_score(labels, predictions)
        matrix = show_confusion(
            labels, predictions,
            args.figures_dir / f"09_confusion_{version}.png",
            f"Leave-one-scan-out classification — {version} ({score*100:.1f}%)",
        )
        comparisons.append({
            "processing": version, "correct": int(np.sum(predictions == labels)),
            "total": len(labels), "accuracy": float(score),
            "confusion_matrix": matrix.tolist(),
        })
        for filename, actual, predicted in zip(names, labels, predictions):
            records.append({"processing": version, "filename": filename,
                            "actual": actual, "predicted": predicted,
                            "correct": bool(actual == predicted)})

        if version == "smoothed":
            explained = show_pca(features, labels, args.figures_dir / "10_pca.png")
            model = make_model().fit(features, labels)
            # The saved model is useful for testing NEW scans from comparable
            # acquisition settings. Loading a .joblib file executes pickle
            # operations, so never load one from an untrusted source.
            joblib.dump({"model": model, "wavenumber": wavenumber,
                         "bands": BANDS, "window": 15, "degree": 3,
                         "labels": list(SAMPLE_LABELS)},
                        args.results_dir / "ftir_exploratory_model.joblib")

    pd.DataFrame(records).to_csv(args.results_dir / "classification_predictions.csv", index=False)
    summary = {"n_scans": len(scans), "model": "StandardScaler + NearestCentroid",
               "validation": "leave-one-scan-out (not independent specimens)",
               "band_windows_cm-1": BANDS, "comparisons": comparisons,
               "pca_explained_variance_ratio": explained,
               "note": "Four recorded sample labels; not elemental identification."}
    (args.results_dir / "classification_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for c in comparisons:
        print(f"{c['processing']}: {c['correct']}/{c['total']} scans correct ({c['accuracy']:.1%})")
    print("Important: repeated same-session scans are not independent specimens.")


if __name__ == "__main__":
    main()
