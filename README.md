# FTIR analysis — silica samples, October 22, 2026

This repository reproduces the FTIR plots from the October 22 Agilent data. It
starts with the original `.asp` transmittance files and builds the averages,
noise-corrected spectra, comparisons, absorption plots, and an **exploratory**
classification model using Python.

I kept the scripts straightforward so the analysis is easy to review and adjust.
The raw measurements are stored separately and are never overwritten.

## Quick start

You'll need Python 3.10 or newer. In Terminal, from this folder:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run_all.py
```

That will regenerate all graphs in `figures/` and all tables/model outputs in
`results/`. To test the data reader and calculations:

```bash
python -m pytest -q
```

The individual steps also work separately:

```bash
python plot_ftir.py           # Figures 01–08 and averaged-data CSVs
python classify_ftir.py       # Confusion matrices, PCA, classifier and predictions
```

## Figures

| File | What it shows |
| --- | --- |
| [01_averaged_spectra.png](figures/01_averaged_spectra.png) | Average of each sample group, after smoothing |
| [02_dry_LHS_vs_LMS.png](figures/02_dry_LHS_vs_LMS.png) | LHS and LMS average spectra, with ±1 sample standard deviation and their difference |
| [03_dry_vs_gels.png](figures/03_dry_vs_gels.png) | Gel 5 versus pooled dry; Gel 8 versus pooled dry |
| [04_differences_vs_dry.png](figures/04_differences_vs_dry.png) | Transmittance difference (gel minus dry) |
| [05_transmittance_and_absorbance.png](figures/05_transmittance_and_absorbance.png) | Corrected transmittance and calculated absorbance |
| [06_noise_before_after.png](figures/06_noise_before_after.png) | Zoomed before/after noise comparisons |
| [07_noise_metrics.png](figures/07_noise_metrics.png) | Short-scale variation before and after smoothing |
| [08_selected_bands.png](figures/08_selected_bands.png) | Absorbance in selected bands; **not** elemental percentages |
| [09_confusion_raw.png](figures/09_confusion_raw.png) | Exploratory classifier using original scans |
| [09_confusion_smoothed.png](figures/09_confusion_smoothed.png) | Same model approach using smoothed scans |
| [10_pca.png](figures/10_pca.png) | Descriptive PCA plot of the smoothed feature vectors |

![Averaged FTIR spectra](figures/01_averaged_spectra.png)

![Noise correction, before and after](figures/06_noise_before_after.png)

## Data and methods

The available raw data consist of **12 Agilent `.asp` files**, each containing
3,596 transmittance measurements across approximately 4000–650 cm⁻¹:

| Sample label in this analysis | Numeric scans |
| --- | ---: |
| Dry silica 2 LHS | 3 |
| Dry silica 2 LMS | 2 |
| Silica Gel 5 | 3 |
| Silica Gel 8 | 4 |

The original file names spell silica as `Silla`; those file names are unchanged.
There is one additional LMS **PDF** for 14:15:34, but no corresponding numeric
`.asp` file was provided. It is not included in the calculations.

The program reads each spectrum's point count and wavenumber endpoints from the
file header. All numeric scans have the same grid. It then:

1. Applies **Savitzky–Golay smoothing** to *each scan* (`window_length=15`,
   `polyorder=3`). This reduces point-to-point fluctuations but is **not** a
   calibrated atmospheric, baseline, or CO₂ correction.
2. Calculates a mean **percent transmittance** spectrum for each sample group.
   The pooled dry reference is the arithmetic mean of all five available LHS
   and LMS scans (3 + 2), **not** the simple 50/50 mean of two group averages.
3. Converts the mean transmittance to absorbance using
   `A = -log10(%T / 100)`. Measurements above 100%T remain in the data and can
   produce negative apparent absorbance. The code does not silently clip them.
4. Calculates differences as **percentage points of %T**, not percent water or
   percent composition. It also saves root-mean-square spectral differences.
5. Calculates a *point-to-point variation proxy* from each scan in the
   3600–3800 cm⁻¹ region: `std(diff(%T)) / sqrt(2)`. This is useful for comparing
   before/after smoothing, but atmospheric features could contribute; it isn't
   a calibrated detector noise or signal-to-noise measurement.

The following features **are not automatically removed**: the narrow ~2350 cm⁻¹
feature (possibly atmospheric CO₂), signals that may include water vapor, and
the broad Gel 8 feature around ~2478 cm⁻¹. Any chemical interpretation of that
last feature should be treated as provisional until the instrument/background
and sample preparation are checked.

## Classification experiment

`classify_ftir.py` uses the mean transmittance in nine specified spectral
windows as features. It trains `StandardScaler` and `NearestCentroid` inside a
pipeline. **Leave-one-scan-out** cross-validation is used for the plotted
confusion matrices, so the scaler isn't fitted on a held-out scan.

On the provided dataset:

- Original scans: **10/12 correct (83.3%)**.
- Smoothed scans: **10/12 correct (83.3%)**.
- The two errors are between the LHS and LMS labels.

This score **does not** validate prediction on new physical specimens. The
measurements were collected during a single session, and scans from the same
sample could be strongly dependent. PCA is descriptive, not validation. The
model only predicts among the four **recorded labels**; FTIR alone does not
provide the elemental composition or a percent water-saturation measurement.

If you have another comparable `.asp` scan, you can run:

```bash
python predict_sample.py "path/to/new_scan.asp"
```

The trained model is created in `results/ftir_exploratory_model.joblib`.
**Only load `.joblib` files that you trust** (the file format uses pickle).

## Project layout

```text
ftir-silica-analysis/
├── data/
│   ├── README.md               # Data provenance and missing LMS file
│   └── raw/                    # 12 unmodified Agilent .asp exports
├── figures/                    # PNG graphs, generated by the scripts
├── results/                    # Averaged CSVs, model summary and predictions
├── tests/test_ftir.py          # Data and calculation checks
├── ftir_tools.py               # File reader, smoothing, averaging, absorbance
├── plot_ftir.py                # Graphs and exported measurement tables
├── classify_ftir.py            # Exploratory classification and PCA
├── predict_sample.py           # Optional prediction for another scan
├── run_all.py                  # Run both analysis scripts
└── requirements.txt
```

## Notes for research reporting

- These filenames don't independently establish the composition or processing
  history of the samples. Please confirm what LHS/LMS mean and whether Gel 5/8
  are distinct preparation conditions.
- These October 22 measurements have **not** been verified as formal project
  measurements versus additional training data. Keep that distinction clear in
  abstracts or presentations.
- For a scientifically stronger classifier, collect independent specimens with
  documented preparation, and split validation by specimen or measurement
  session instead of treating repeat scans as fully independent.
- This repository shares an analysis workflow, not a definitive chemical
  identification or instrument-calibration procedure.

## Putting this on GitHub

Create an empty repository on GitHub, then in this project folder run:

```bash
git init
git add .
git commit -m "Add FTIR silica spectrum analysis"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin main
```

Replace the example URL with your own. Review the raw files before publishing
if your lab or mentor has data-sharing restrictions.
