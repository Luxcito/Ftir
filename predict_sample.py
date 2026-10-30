"""Use the exploratory model to label an additional .asp scan.

    python predict_sample.py path/to/new_scan.asp

The prediction is among four file labels only, not chemical identification.
"""

from pathlib import Path
import argparse

import joblib
import numpy as np

from classify_ftir import band_features
from ftir_tools import read_asp, smooth_transmittance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spectrum", type=Path)
    parser.add_argument("--model", type=Path, default=Path("results/ftir_exploratory_model.joblib"))
    args = parser.parse_args()
    # Do not load a joblib/pickle file unless you trust where it came from.
    saved = joblib.load(args.model)
    x, transmittance = read_asp(args.spectrum)
    reference = saved["wavenumber"]
    if x.min() > reference.min() or x.max() < reference.max():
        raise ValueError("New scan doesn't cover the saved model's full spectral range")
    if len(x) != len(reference) or not np.allclose(x, reference, atol=0.001, rtol=0):
        # Interpolation uses increasing x, but the FTIR convention plots it backwards.
        transmittance = np.interp(reference, x[::-1], transmittance[::-1])
        x = reference
    corrected = smooth_transmittance(transmittance, saved["window"], saved["degree"])
    features = np.array(band_features(x, corrected, saved["bands"])).reshape(1, -1)
    print("Predicted recorded sample label:", saved["model"].predict(features)[0])
    print("This doesn't establish chemical composition or accuracy on new specimens.")


if __name__ == "__main__":
    main()
