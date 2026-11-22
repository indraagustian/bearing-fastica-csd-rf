"""5-Fold Leak-Free Block Cross-Validation Benchmark (Reproduces Table IV)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    MODEL_NAMES,
    REPO_ROOT,
    build_classifier_zoo,
    evaluate_fold_models,
    summarize_metrics,
    tune_nested_rf,
)

GAP = 20  # 500-sample window / 25-sample step -> zero raw sample overlap
BLOCK_LEN = 44  # 220 windows per (class, speed) / 5 blocks
SEG_LEN = 220


def build_block_assignments(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign each window to a contiguous block fold (0..4) and GroupKFold ID."""
    fold_ids = np.zeros(len(labels), dtype=int)
    group_ids = np.zeros(len(labels), dtype=int)
    for cls in range(5):
        cls_idx = np.where(labels == cls)[0]
        assert len(cls_idx) == 660, f"Expected 660 windows for class {cls}, got {len(cls_idx)}"
        for spd in range(3):
            seg = cls_idx[spd * SEG_LEN : (spd + 1) * SEG_LEN]
            for j, r_idx in enumerate(seg):
                block = min(j // BLOCK_LEN, 4)
                fold_ids[r_idx] = block
                group_ids[r_idx] = cls * 15 + spd * 5 + block
    return fold_ids, group_ids


def get_fold_indices(labels: np.ndarray, fold_ids: np.ndarray, test_fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Return non-overlapping training and testing indices with a GAP-window guard band."""
    test_mask = fold_ids == test_fold
    train_mask = np.zeros(len(labels), dtype=bool)
    for cls in range(5):
        cls_idx = np.where(labels == cls)[0]
        for spd in range(3):
            seg = cls_idx[spd * SEG_LEN : (spd + 1) * SEG_LEN]
            t_start = test_fold * BLOCK_LEN
            t_end = (test_fold + 1) * BLOCK_LEN - 1
            for j, r_idx in enumerate(seg):
                if fold_ids[r_idx] != test_fold and (j < t_start - GAP or j > t_end + GAP):
                    train_mask[r_idx] = True
    return np.where(train_mask)[0], np.where(test_mask)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="5-Fold Leak-Free Block CV Benchmark (Paper Table IV)")
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "3result25-500-5_nfica_stat.csv",
        help="Path to the 3,300-row 12-feature CSV dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "results_leakfree_5fold.csv",
        help="Path to write the benchmark summary CSV.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    X_all = df.drop(columns=["label"]).values
    y_all = df["label"].values
    fold_ids, group_ids = build_block_assignments(y_all)

    param_grid = {
        "n_estimators": [200],
        "max_depth": [5, 7, 10],
        "min_samples_split": [10, 25, 50],
        "min_samples_leaf": [2, 5],
        "max_features": ["sqrt", 0.5],
    }

    metrics_store = {m: {"acc": [], "prec": [], "rec": [], "f1": [], "auc": []} for m in MODEL_NAMES}
    for fold in range(5):
        tr_idx, te_idx = get_fold_indices(y_all, fold_ids, fold)
        X_tr, y_tr, g_tr = X_all[tr_idx], y_all[tr_idx], group_ids[tr_idx]
        X_te, y_te = X_all[te_idx], y_all[te_idx]

        rf_grid = tune_nested_rf(X_tr, y_tr, g_tr, param_grid=param_grid, n_splits=4)
        fold_models = build_classifier_zoo(rf_grid.best_estimator_)
        evaluate_fold_models(fold_models, X_tr, y_tr, X_te, y_te, metrics_store)
        print(f"Fold {fold + 1} done (train={len(tr_idx)}, test={len(te_idx)}, best_rf={rf_grid.best_params_})")

    summary_df = summarize_metrics(metrics_store)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.output, index=False)

    print("\n=== 5-FOLD LEAK-FREE BLOCK CV SUMMARY ===")
    for _, r in summary_df.iterrows():
        print(
            f"{r['model']:<24} | Acc: {r['acc_mean']:5.2f} +/- {r['acc_std']:4.2f}% "
            f"| Prec: {r['macro_precision']:.3f} | Rec: {r['macro_recall']:.3f} "
            f"| F1: {r['macro_f1']:.3f} | AUC: {r['macro_roc_auc']:.3f}"
        )


if __name__ == "__main__":
    main()
