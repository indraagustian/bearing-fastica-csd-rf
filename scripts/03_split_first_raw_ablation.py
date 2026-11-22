"""Split-First Per-Fold Preprocessing FastICA Ablation & 8-Model Benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from common import (
    FIELD_CLASS_SPECS,
    MODEL_NAMES,
    REPO_ROOT,
    SENSOR_COLS,
    align_ica_via_csd,
    build_classifier_zoo,
    evaluate_fold_models,
    extract_12_features,
    make_regularized_rf,
    summarize_metrics,
    tune_nested_rf,
)


def load_raw_streams(raw_dir: Path) -> dict[tuple[int, int], np.ndarray]:
    """Load the 15 labeled 18,000 x 4 raw field accelerometer streams."""
    raw_data = {}
    for cls, (prefix, _) in FIELD_CLASS_SPECS.items():
        for pos in [1, 2, 3]:
            path = raw_dir / f"{prefix}_posisi_{pos}.csv"
            raw_data[(cls, pos)] = pd.read_csv(path)[SENSOR_COLS].values[:18000]
    return raw_data


def build_fold_datasets(
    raw_data: dict[tuple[int, int], np.ndarray],
    test_fold: int,
    use_fastica: bool = True,
    window: int = 500,
    step: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split raw signal into 5 blocks FIRST, then fit FastICA/CSD/StandardScaler solely on train blocks."""
    X_tr_list, y_tr_list, g_tr_list = [], [], []
    X_te_list, y_te_list = [], []
    block_size = 3600  # 18,000 / 5

    for cls, (_, target_idx) in FIELD_CLASS_SPECS.items():
        for pos in [1, 2, 3]:
            arr = raw_data[(cls, pos)]
            blocks = [arr[b * block_size : (b + 1) * block_size] for b in range(5)]
            train_blocks = [blocks[b] for b in range(5) if b != test_fold]
            test_block = blocks[test_fold]

            X_tr_raw = np.vstack(train_blocks)
            if use_fastica:
                ica = FastICA(n_components=4, random_state=123, max_iter=1000)
                S_tr = ica.fit_transform(X_tr_raw)
                best_ic = align_ica_via_csd(X_tr_raw, S_tr, target_idx)
                tr_sig_all = S_tr[:, best_ic : best_ic + 1]
                te_sig_all = ica.transform(test_block)[:, best_ic : best_ic + 1]
            else:
                tr_sig_all = X_tr_raw[:, target_idx : target_idx + 1]
                te_sig_all = test_block[:, target_idx : target_idx + 1]

            scaler = StandardScaler()
            tr_sig_norm = scaler.fit_transform(tr_sig_all).ravel()
            te_sig_norm = scaler.transform(te_sig_all).ravel()

            tr_b_idx = 0
            for b in range(5):
                if b == test_fold:
                    continue
                seg_b = tr_sig_norm[tr_b_idx * block_size : (tr_b_idx + 1) * block_size]
                fb = extract_12_features(seg_b, window=window, step=step, strict_less=False)
                X_tr_list.append(fb)
                y_tr_list.append(np.full(len(fb), cls, dtype=int))
                g_tr_list.append(np.full(len(fb), cls * 15 + (pos - 1) * 5 + b, dtype=int))
                tr_b_idx += 1

            fte = extract_12_features(te_sig_norm, window=window, step=step, strict_less=False)
            X_te_list.append(fte)
            y_te_list.append(np.full(len(fte), cls, dtype=int))

    return (
        np.vstack(X_tr_list),
        np.concatenate(y_tr_list),
        np.concatenate(g_tr_list),
        np.vstack(X_te_list),
        np.concatenate(y_te_list),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Split-First Per-Fold Preprocessing FastICA Ablation")
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data" / "raw_fieldtest")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "results_split_first_raw.csv",
    )
    args = parser.parse_args()

    raw_data = load_raw_streams(args.raw_dir)
    param_grid = {
        "n_estimators": [200],
        "max_depth": [5, 8, 12],
        "min_samples_split": [10, 30],
        "min_samples_leaf": [2, 5],
    }

    # 1. Ablation: Without FastICA vs With FastICA (RF Regularized & Nested-CV)
    print("=== ABLATION: WITHOUT vs WITH FASTICA (5-Fold Split-First Blocked CV) ===")
    ablation_rows = []
    for use_fica in [False, True]:
        accs_paper, accs_tuned = [], []
        for f in range(5):
            X_tr, y_tr, g_tr, X_te, y_te = build_fold_datasets(raw_data, f, use_fastica=use_fica)
            rf_p = make_regularized_rf()
            rf_p.fit(X_tr, y_tr)
            accs_paper.append(accuracy_score(y_te, rf_p.predict(X_te)) * 100.0)

            rf_g = tune_nested_rf(X_tr, y_tr, g_tr, param_grid=param_grid, n_splits=4)
            accs_tuned.append(accuracy_score(y_te, rf_g.predict(X_te)) * 100.0)

        tag = "With FastICA" if use_fica else "Without FastICA"
        print(
            f"  {tag:<16} | RF Paper: {np.mean(accs_paper):.2f} +/- {np.std(accs_paper):.2f}% "
            f"| RF Tuned: {np.mean(accs_tuned):.2f} +/- {np.std(accs_tuned):.2f}%"
        )
        if not use_fica:
            ablation_rows.extend(
                [
                    {"scenario": tag, "model": "Random Forest (Paper)", "acc_mean": round(float(np.mean(accs_paper)), 2), "acc_std": round(float(np.std(accs_paper)), 2)},
                    {"scenario": tag, "model": "Random Forest (Tuned)", "acc_mean": round(float(np.mean(accs_tuned)), 2), "acc_std": round(float(np.std(accs_tuned)), 2)},
                ]
            )

    # 2. Full 8-classifier benchmark on With FastICA
    print("\n=== 8-MODEL BENCHMARK ON SPLIT-FIRST WITH-FASTICA (5-Fold Blocked CV) ===")
    metrics_store = {m: {"acc": [], "prec": [], "rec": [], "f1": [], "auc": []} for m in MODEL_NAMES}
    oof_true, oof_pred_rf = [], []
    for f in range(5):
        X_tr, y_tr, g_tr, X_te, y_te = build_fold_datasets(raw_data, f, use_fastica=True)
        rf_g = tune_nested_rf(X_tr, y_tr, g_tr, param_grid=param_grid, n_splits=4)
        fold_models = build_classifier_zoo(rf_g.best_estimator_)
        y_t, tuned_pred = evaluate_fold_models(fold_models, X_tr, y_tr, X_te, y_te, metrics_store)
        oof_true.append(y_t)
        oof_pred_rf.append(tuned_pred)

    with_df = summarize_metrics(metrics_store, scenario="With FastICA")
    for _, r in with_df.iterrows():
        print(
            f"  {r['model']:<24} | Acc: {r['acc_mean']:5.2f} +/- {r['acc_std']:4.2f}% "
            f"| Prec: {r['macro_precision']:.3f} | Rec: {r['macro_recall']:.3f} "
            f"| F1: {r['macro_f1']:.3f} | AUC: {r['macro_roc_auc']:.3f}"
        )

    out_df = pd.concat([pd.DataFrame(ablation_rows), with_df], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    cm = confusion_matrix(np.concatenate(oof_true), np.concatenate(oof_pred_rf), normalize="true")
    print("\nRF Tuned OOF Normalized Confusion Matrix:")
    print(np.round(cm, 2))


if __name__ == "__main__":
    main()
