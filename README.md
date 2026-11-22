# Bearing Fault Severity Classification Using FastICA-CSD Separation and Random Forest

Standalone FAIR supplementary material and reproducibility repository for the manuscript **"Bearing Fault Severity Classification Using FastICA-CSD Separation and Random Forest"**.

> **Note for Peer Review (Double-Blind Safety):**  
> Core scripts (`scripts/`), datasets (`data/`), documentation (`docs/`), and benchmark outputs (`results/`) are kept author-neutral. If uploading this repository as anonymous supplementary material for a double-blind journal submission, remove `CITATION.cff` and update the copyright header in `LICENSE` before archiving.

---

## 1. Repository Structure

```text
supplementary_fair_repo/
├── README.md                                  # Quickstart, setup, and reproduction guide
├── CITATION.cff                               # Machine-readable citation metadata (GitHub/Zenodo)
├── LICENSE                                    # MIT (code) + CC BY 4.0 (data & docs)
├── requirements.txt                           # Python dependencies
├── docs/
│   ├── DATA_DICTIONARY.md                     # 5-class severity dictionary, sensor wiring, 12-feature specs
│   └── SUPPLEMENTARY_NOTES.md                 # Complete leak-free benchmark tables & field-test notes
├── data/
│   ├── processed/
│   │   ├── 3result25-500-5_nfica_stat.csv     # Primary 3,300 x 13 feature dataset (W=500, step=25)
│   │   └── datates_mentah_500_5_fica_stat.csv # Unified 10,500 x 13 field-test features
│   └── raw_fieldtest/
│       └── *.csv                              # 15 raw 4-channel MPU6050 streams (18,000 rows each)
├── scripts/
│   ├── common.py                              # Shared FastICA-CSD, Welch PSD, and classifier utilities
│   ├── 01_leakfree_5fold_benchmark.py         # Reproduces 5-fold leak-free block CV (Paper Table IV)
│   ├── 02_unified_fieldtest_cv.py             # Reproduces unified 10,500-window field-test benchmark
│   ├── 03_split_first_raw_ablation.py         # Reproduces split-first FastICA ablation on raw streams
│   ├── 04_verify_dataset_and_rf.py            # Verifies dataset geometry (300/500/700) & Gini importances
│   └── run_all.py                             # Executes scripts 04 -> 01 -> 02 -> 03 sequentially
└── results/
    ├── results_leakfree_5fold.csv             # Out-of-fold benchmark across 8 classifier configs
    ├── results_unified_mentah_leakfree.csv    # Unified field-test benchmark (10,500 windows)
    └── results_split_first_raw.csv            # Split-first FastICA ablation + 8-model benchmark
```

Total repository footprint is ~24.1 MB (all individual files under 2 MB; no Git LFS required).

---

## 2. Quickstart & End-to-End Reproduction

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run All Verification and Benchmark Scripts

```bash
python scripts/run_all.py
```

Or run individual stages cleanly with optional CLI paths:

```bash
python scripts/04_verify_dataset_and_rf.py
python scripts/01_leakfree_5fold_benchmark.py
python scripts/02_unified_fieldtest_cv.py
python scripts/03_split_first_raw_ablation.py
```

---

## 3. Mapping Scripts to Manuscript Sections

| Script | Paper Reference | Verified Key Output |
| :--- | :--- | :--- |
| `04_verify_dataset_and_rf.py` | Section II-G, II-H, Table II, Fig. 13 | Verifies `3,420` / `3,300` / `3,180` window counts, 15 raw CSVs, and top-4 Gini features (`68.3%` native refit; `68.5%` original Colab binary) |
| `01_leakfree_5fold_benchmark.py` | Section II-E, II-H, III-D, Table IV | Reproduces 8-classifier 5-fold leak-free CV (`GAP = 20`): Regularized RF `80.00% ± 3.06%`, Nested-CV RF `80.27% ± 3.67%` |
| `02_unified_fieldtest_cv.py` | Section II-H, III-D (`datates/mentah`) | Reproduces 10,500-window unified leak-free CV: Nested-CV RF macro ROC-AUC `0.889` (`63.31% ± 3.01%`), Regularized RF `60.78% ± 3.11%` |
| `03_split_first_raw_ablation.py` | Section II-H, III-D (`datates/mentah`) | Reproduces split-first per-fold preprocessing FastICA gain (`50.23% → 61.53%` Regularized RF; `55.97% → 64.61%` Nested-CV RF) |

*(CSV naming convention note: within output CSVs, `Random Forest (Paper)` corresponds to **Random Forest (Regularized)** and `Random Forest (Tuned)` corresponds to **Random Forest (Nested-CV)** in the manuscript.)*

---

## 4. Hardware & Acquisition Summary

- **Sensors:** 4x MPU6050 triaxial MEMS accelerometers ($\pm 2g$ default range via `MPU6050_tockn`).
- **Multiplexing & Streaming:** TCA9548A I2C multiplexer (`0x70`, channels `0, 2, 4, 7`) polled sequentially by an Arduino Mega 2560 under a $>1\text{ ms}$ timer gate, streamed via USB Type-B at `115,200 bps`.
- **Operating Speeds:** 242 rpm (`posisi_1`), 490 rpm (`posisi_2`), and 662 rpm (`posisi_3`).
- **Fault Classes:** 5 severity levels representing 0, 1, 2, 3, and 4 rolling balls removed out of 7 (`0%`, `14.3%`, `28.6%`, `42.9%`, `57.1%` rolling-element loss).
