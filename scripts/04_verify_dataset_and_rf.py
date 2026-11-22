"""Dataset Integrity, Sliding-Window Arithmetic, and Regularized RF Audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    FEATURE_COLS,
    FIELD_CLASS_SPECS,
    REPO_ROOT,
    SENSOR_COLS,
    make_regularized_rf,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify dataset integrity, window counts, and RF importances")
    parser.add_argument(
        "--processed-csv",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "3result25-500-5_nfica_stat.csv",
    )
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw_fieldtest")
    args = parser.parse_args()

    # 1. Verify sliding-window arithmetic reported in Section II.H & Table II
    print("=== 1. SLIDING-WINDOW COUNT VERIFICATION (6,000 samples x 3 speeds x 5 classes, step=25) ===")
    for w in [300, 500, 700]:
        per_speed = len([i for i in range(0, 6000, 25) if i + w < 6000])
        total = per_speed * 3 * 5
        print(f"  Window W={w}: {per_speed} per speed x 3 speeds x 5 classes = {total:,} instances")

    # 2. Verify primary processed dataset
    df = pd.read_csv(args.processed_csv)
    assert df.shape == (3300, 13), f"Unexpected processed shape: {df.shape}"
    assert list(df.columns) == FEATURE_COLS + ["label"]
    assert int(df.isna().sum().sum()) == 0
    counts = df["label"].value_counts().sort_index().to_dict()
    print(f"\n=== 2. PROCESSED DATASET AUDIT ({args.processed_csv.name}) ===")
    print(f"  Shape: {df.shape} | Missing values: 0 | Class counts: {counts}")

    # 3. Verify all 15 raw field-test CSVs (12 labeled + 3 unlubricated Tanpa_Pelumas)
    print(f"\n=== 3. RAW FIELD RECORDINGS AUDIT ({args.raw_dir.name}) ===")
    for cls, (prefix, target_idx) in FIELD_CLASS_SPECS.items():
        for pos in [1, 2, 3]:
            p = args.raw_dir / f"{prefix}_posisi_{pos}.csv"
            raw_df = pd.read_csv(p)
            assert raw_df.shape[0] == 18000, f"{p.name}: unexpected rows {raw_df.shape[0]}"
            assert set(SENSOR_COLS).issubset(raw_df.columns)
            assert int(raw_df[SENSOR_COLS].isna().sum().sum()) == 0
    for pos in [1, 2, 3]:
        p = args.raw_dir / f"Tanpa_Pelumas_posisi_{pos}.csv"
        dry_df = pd.read_csv(p)
        assert dry_df.shape[0] == 18000, f"{p.name}: unexpected rows {dry_df.shape[0]}"
        assert set(SENSOR_COLS).issubset(dry_df.columns)
        assert int(dry_df[SENSOR_COLS].isna().sum().sum()) == 0
    print("  All 12 labeled field recordings + 3 Tanpa_Pelumas recordings verified (18,000 rows, 4 resultant channels, 0 missing).")

    # 4. Fit regularized RF natively on modern scikit-learn and inspect Gini feature importance
    rf = make_regularized_rf(random_state=123)
    rf.fit(df[FEATURE_COLS].values, df["label"].values)
    imp = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n=== 4. REGULARIZED RF GINI FEATURE IMPORTANCES (native scikit-learn refit) ===")
    for feat, val in imp.items():
        print(f"  {feat:<14}: {val:.4f}")
    top4_sum = float(imp.iloc[:4].sum()) * 100.0
    print(f"  Top-4 features ({', '.join(imp.index[:4])}) cumulative importance: {top4_sum:.1f}%")


if __name__ == "__main__":
    main()
