# Excessive Recruitment of Ripple-Rich States in Schizophrenia
## Reproducible Analysis Repository

This repository contains the analysis code and derived (anonymized) datasets supporting the manuscript:

Takei Y, Ohki T, Sunaga M, Kato Y, Fujihara K, Sekiya T, Jinde S.  
**Excessive Recruitment of Ripple-Rich States in Schizophrenia Links State Engagement to Positive Symptoms.**  
Biological Psychiatry. DOI: [to be updated upon publication]

The goal of this repository is to enable transparent inspection, computational reproduction, and extension of all reported analyses while respecting data privacy and ethical constraints associated with human neurophysiological recordings.

All statistical results and figures are recomputed directly from the included derived tables.

---

# Software Dependencies

This repository requires a standard scientific Python environment.

Tested on:
- Python 3.11
- Ubuntu 22.04 (Code Ocean compute capsule)

Core dependencies:
- NumPy
- SciPy
- pandas
- matplotlib
- statsmodels
- MNE-Python
- nibabel

Exact package versions are specified in `requirements.txt`.

---

# Scope of the Repository

This repository supports the following components of the manuscript:

- Detection and characterization of ripple events (80–240 Hz)
- Identification of temporally clustered ripple-rich neural states
- Group comparisons between healthy controls (HC) and schizophrenia (SZ)
- Surrogate-based validation of clustering
- Ripple counts, event durations, and event-controlled spectral power analyses
- Network distribution of ripple events
- Clinical symptom association analyses

The repository is analysis-complete: all numerical results, statistical inferences, and figures can be regenerated using the provided scripts and data.

---

# Repository Structure

```
.
├── data/
│   ├── subject-level and group-level analysis tables (derived)
│   ├── surrogate-aware clustering outputs
│   ├── anonymized clinical association tables
│   └── group-averaged neuroimaging maps (NIfTI)
│
├── scripts/
│   ├── figure-specific analysis scripts (Figs. 1–4)
│   ├── supplementary table builders
│   └── utility scripts
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── logs/
│
├── run_all.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Reproducibility

## One-Command Reproduction

From the repository root:

```
python run_all.py
```

Optional modes:

```
python run_all.py --dry-run
python run_all.py --only fig3 fig4
python run_all.py --continue-on-error
```

All outputs are written to:

- `outputs/figures/`
- `outputs/tables/`
- `outputs/logs/`

---

# Data Privacy and Ethical Compliance

- Only derived, anonymized, and non-identifiable data are shared.
- No raw MEG recordings are included.
- Internal linkage between derived tables uses anonymized identifiers only.
- No information included in this repository allows re-identification of participants.

---

# Data Dictionary (Summary)

Core tables include:

- `df_clean_expanded.csv` — merged subject-level derived variables
- `epoch_metrics_subject_freq.csv` — clustered epoch metrics
- `epoch_metrics_epoch_level.csv` — epoch-level durations
- `high_rate_epoch_debug_merged.csv` — surrogate metrics
- `public_clinical_subject_level.csv` — anonymized clinical data
- `events_by_network_subject_level.csv` — network-based ripple counts
- `df_PSD.csv` — spectral power summaries

Group-level NIfTI files support visualization of Fig. 2a and contain only averaged source maps.

---

# Citation

If you use this code or derived data, please cite:

Takei Y, Ohki T, Sunaga M, Kato Y, Fujihara K, Sekiya T, Jinde S.  
Excessive Recruitment of Ripple-Rich States in Schizophrenia Links State Engagement to Positive Symptoms.  
Biological Psychiatry. DOI: [to be updated]

---

# License

Code in this repository is released under the MIT License.

Derived data tables are provided for research use under CC-BY 4.0.  
See LICENSE for details.
