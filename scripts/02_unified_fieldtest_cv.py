"""Unified Field-Test Pipeline & 5-Fold Leak-Free Block CV on Raw Recordings."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

from common import (
    FEATURE_COLS,
    FIELD_CLASS_SPECS,
    MODEL_NAMES,
    REPO_ROOT,
    SENSOR_COLS,
    align_ica_via_csd,
    build_classifier_zoo,
    evaluate_fold_models,
    extract_12_features,
    summarize_metrics,
    tune_nested_rf,
)

GAP = 20


def build_unified_features(raw_dir: Path, out_csv: Path) -> pd.DataFrame:
    """Apply FastICA -> CSD -> StandardScaler -> W=500, step=25 across all 15 labeled streams."""
    rows = []
    per_pos_counts = {}
    for cls, (prefix, target_idx) in FIELD_CLASS_SPECS.items():
        for pos in [1, 2, 3]:
            path = raw_dir / f"{prefix}_posisi_{pos}.csv"
            X = pd.read_csv(path)[SENSOR_COLS].values[:18000]
            ica = FastICA(n_components=4, random_state=123, max_iter=1000)
            S = ica.fit_transform(X)
            best_ic = align_ica_via_csd(X, S, target_idx)
            sig_norm = StandardScaler().fit_transform(S[:, best_ic : best_ic + 1]).ravel()
            feats = extract_12_features(sig_norm, window=500, step=25, strict_less=True)
            per_pos_counts[(cls, pos)] = len(feats)
            for row in feats:
                rows.append(list(row) + [cls])

    df_feat = pd.DataFrame(rows, columns=FEATURE_COLS + ["label"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_feat.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}: shape={df_feat.shape}, per_stream_windows={set(per_pos_counts.values())}")
    return df_feat


def assign_field_blocks(n_rows: int, n_per_pos: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Assign 5 contiguous temporal blocks per (class, speed) recording stream."""
    block_len = n_per_pos // 5
    fold_ids = np.zeros(n_rows, dtype=int)
    group_ids = np.zeros(n_rows, dtype=int)
    idx_cursor = 0
    for cls in range(5):
        for pos in [1, 2, 3]:
            for j in range(n_per_pos):
                b = min(j // block_len, 4)
                fold_ids[idx_cursor + j] = b
                group_ids[idx_cursor + j] = cls * 15 + (pos - 1) * 5 + b
            idx_cursor += n_per_pos
    return fold_ids, group_ids, block_len


def get_fold_indices(
    n_rows: int, fold_ids: np.ndarray, n_per_pos: int, block_len: int, test_fold: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exclude windows within GAP of train-test boundaries per stream."""
    test_mask = fold_ids == test_fold
    train_mask = np.zeros(n_rows, dtype=bool)
    idx_cursor = 0
    t_start = test_fold * block_len
    t_end = (test_fold + 1) * block_len - 1
    for _cls in range(5):
        for _pos in [1, 2, 3]:
            for j in range(n_per_pos):
                r_idx = idx_cursor + j
                if fold_ids[r_idx] != test_fold and (j < t_start - GAP or j > t_end + GAP):
                    train_mask[r_idx] = True
            idx_cursor += n_per_pos
    return np.where(train_mask)[0], np.where(test_mask)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Field-Test Feature Extraction & 5-Fold Leak-Free CV")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw_fieldtest")
    parser.add_argument(
        "--feat-csv",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "datates_mentah_500_5_fica_stat.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "results_unified_mentah_leakfree.csv",
    )
    args = parser.parse_args()

    df_feat = build_unified_features(args.raw_dir, args.feat_csv)
    n_per_pos = len(df_feat) // 15
    fold_ids, group_ids, block_len = assign_field_blocks(len(df_feat), n_per_pos)

    X_all = df_feat.drop(columns=["label"]).values
    y_all = df_feat["label"].values
    param_grid = {
        "n_estimators": [200],
        "max_depth": [5, 8, 12],
        "min_samples_split": [10, 30],
        "min_samples_leaf": [2, 5],
    }

    metrics_store = {m: {"acc": [], "prec": [], "rec": [], "f1": [], "auc": []} for m in MODEL_NAMES}
    for fold in range(5):
        tr_idx, te_idx = get_fold_indices(len(df_feat), fold_ids, n_per_pos, block_len, fold)
        X_tr, y_tr, g_tr = X_all[tr_idx], y_all[tr_idx], group_ids[tr_idx]
        X_te, y_te = X_all[te_idx], y_all[te_idx]

        rf_grid = tune_nested_rf(X_tr, y_tr, g_tr, param_grid=param_grid, n_splits=4)
        fold_models = build_classifier_zoo(rf_grid.best_estimator_)
        evaluate_fold_models(fold_models, X_tr, y_tr, X_te, y_te, metrics_store)
        print(f"Fold {fold + 1} done (train={len(tr_idx)}, test={len(te_idx)})")

    summary_df = summarize_metrics(metrics_store)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output, index=False)

    print("\n=== UNIFIED TREATMENT ON FIELD RECORDINGS + 5-FOLD LEAK-FREE BLOCK CV ===")
    for _, r in summary_df.iterrows():
        print(
            f"  {r['model']:<24} | Acc: {r['acc_mean']:5.2f} +/- {r['acc_std']:4.2f}% "
            f"| Prec: {r['macro_precision']:.3f} | Rec: {r['macro_recall']:.3f} "
            f"| F1: {r['macro_f1']:.3f} | AUC: {r['macro_roc_auc']:.3f}"
        )


if __name__ == "__main__":
    main()
