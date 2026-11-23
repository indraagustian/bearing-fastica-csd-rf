# Data Dictionary

This document describes all datasets bundled under `data/` for reproducible verification of the accompanying manuscript *"Bearing Fault Severity Classification Using FastICA-CSD Separation and Random Forest"*.

## 1. Bearing Fault Severity Classes (5 Classes, 7 Total Rolling Balls)

| Class Label (`label`) | Bearing State | Balls Removed | Rolling-Element Loss (%) | Field-Test Prefix & Target Sensor |
| :---: | :--- | :---: | :---: | :--- |
| `0` | Normal (Healthy) | 0 of 7 | 0.0% | `Bearing_1_Bola_Lepas_posisi_*.csv` -> `Sensor 1` |
| `1` | Fault Level 1 | 1 of 7 | 14.3% | `Bearing_1_Bola_Lepas_posisi_*.csv` -> `Sensor 3` |
| `2` | Fault Level 2 | 2 of 7 | 28.6% | `Bearing_2_Bola_Lepas_posisi_*.csv` -> `Sensor 3` |
| `3` | Fault Level 3 | 3 of 7 | 42.9% | `Bearing_3_Bola_Lepas_posisi_*.csv` -> `Sensor 3` |
| `4` | Fault Level 4 | 4 of 7 | 57.1% | `Bearing_4_Bola_Lepas_posisi_*.csv` -> `Sensor 3` |

*Dry-lubrication follow-up recordings:* `Tanpa_Pelumas_posisi_{1,2,3}.csv` (3 files, 18,000 samples each) are provided for exploratory follow-up diagnostics of dry-lubrication bearing behavior.

## 2. Speed Positions

- `posisi_1` / Speed 1: **242 rpm**
- `posisi_2` / Speed 2: **490 rpm**
- `posisi_3` / Speed 3: **662 rpm**

## 3. Raw Multi-Channel Field Recordings (`data/raw_fieldtest/*.csv`)

- **Files:** 15 CSV recordings (12 labeled fault conditions + 3 `Tanpa_Pelumas` recordings).
- **Length:** 18,000 contiguous time-series rows per recording (0 missing values across sensor channels).
- **Columns (16 core signals):**
  - `Sensor 1`, `Sensor 2`, `Sensor 3`, `Sensor 4`: Resultant acceleration magnitudes $a = \sqrt{a_x^2 + a_y^2 + a_z^2}$ mounted on **Bearing 1**, **DC Motor**, **Bearing 2**, and **Differential Gear**, respectively.
  - `Ax1`, `Ay1`, `Az1` through `Ax4`, `Ay4`, `Az4`: Raw triaxial MPU6050 readings ($\pm 2g$ default range via `MPU6050_tockn`, polled sequentially via a TCA9548A I2C multiplexer at `0x70`, streamed over USB at `115,200 bps`).
  - *Schema note:* `Bearing_3_Bola_Lepas_posisi_3.csv` contains a trailing delimiter producing an empty `Unnamed: 16` column; all scripts select explicitly by `["Sensor 1", "Sensor 2", "Sensor 3", "Sensor 4"]`.

## 4. Processed Window-Feature Datasets (`data/processed/*.csv`)

- `3result25-500-5_nfica_stat.csv`: **3,300 rows x 13 columns** (`660` windows per class across 3 speeds = `220` contiguous windows per `(class, speed)` from 6,000 samples at `W = 500`, `step = 25`).
- `datates_mentah_500_5_fica_stat.csv`: **10,500 rows x 13 columns** (`700` windows per `(class, speed)` stream across 18,000 raw samples at `W = 500`, `step = 25`), generated deterministically by `scripts/02_unified_fieldtest_cv.py`.

**Feature Columns (12 Features + `label`):**

| Index | Column | Domain | Mathematical / Numerical Definition |
| :---: | :--- | :--- | :--- |
| 1 | `mean` | Time | Window sample mean |
| 2 | `std` | Time | Sample standard deviation (`ddof=1`) |
| 3 | `max` | Time | Maximum absolute amplitude $\lvert\max(\text{seg})\rvert$ |
| 4 | `min` | Time | Minimum absolute amplitude $\lvert\min(\text{seg})\rvert$ |
| 5 | `kurtosis` | Time | Unbiased Fisher excess kurtosis |
| 6 | `skew` | Time | Unbiased sample skewness |
| 7 | `psd_mean` | PSD | Mean of Welch PSD (`nperseg=100`, `average="median"`) |
| 8 | `psd_std` | PSD | Standard deviation of Welch PSD |
| 9 | `psd_kurtosis` | PSD | Kurtosis of Welch PSD |
| 10 | `psd_skew` | PSD | Skewness of Welch PSD |
| 11 | `psd_max` | PSD | Square root of maximum Welch PSD amplitude ($\sqrt{\max(\text{PSD})}$) |
| 12 | `psd_min` | PSD | Square root of minimum Welch PSD amplitude ($\sqrt{\min(\text{PSD})}$) |
| 13 | `label` | Target | Integer class label in `{0, 1, 2, 3, 4}` |
