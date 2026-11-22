"""Shared signal-processing, feature extraction, and evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import kurtosis, skew
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parent.parent

SENSOR_COLS: List[str] = ["Sensor 1", "Sensor 2", "Sensor 3", "Sensor 4"]

FEATURE_COLS: List[str] = [
    "mean",
    "std",
    "max",
    "min",
    "kurtosis",
    "skew",
    "psd_mean",
    "psd_std",
    "psd_kurtosis",
    "psd_skew",
    "psd_max",
    "psd_min",
]

# Maps health class label (0..4 removed balls out of 7) -> (CSV filename prefix, target sensor index)
FIELD_CLASS_SPECS: Dict[int, Tuple[str, int]] = {
    0: ("Bearing_1_Bola_Lepas", 0),  # Sensor 1 = Normal bearing
    1: ("Bearing_1_Bola_Lepas", 2),  # Sensor 3 = 1 ball removed
    2: ("Bearing_2_Bola_Lepas", 2),  # Sensor 3 = 2 balls removed
    3: ("Bearing_3_Bola_Lepas", 2),  # Sensor 3 = 3 balls removed
    4: ("Bearing_4_Bola_Lepas", 2),  # Sensor 3 = 4 balls removed
}

MODEL_NAMES: List[str] = [
    "Random Forest (Paper)",
    "Random Forest (Tuned)",
    "SVM (RBF)",
    "Gradient Boosting",
    "Logistic Regression",
    "KNN (k=5)",
    "MLP (ANN)",
    "Decision Tree",
]


def align_ica_via_csd(
    X: np.ndarray, S: np.ndarray, target_col_idx: int, fs: float = 100.0, nperseg: int = 50
) -> int:
    """Return index of the independent component maximizing mean Welch coherence with X[:, target_col_idx]."""
    best_ic, best_coh = 0, -1.0
    x = X[:, target_col_idx]
    for ic in range(S.shape[1]):
        _, coh = signal.coherence(x, S[:, ic], fs=fs, nperseg=nperseg)
        mean_coh = float(np.mean(coh))
        if mean_coh > best_coh:
            best_coh, best_ic = mean_coh, ic
    return best_ic


def extract_12_features(
    sig_1d: np.ndarray,
    window: int = 500,
    step: int = 25,
    strict_less: bool = True,
) -> np.ndarray:
    """Extract 6 time-domain and 6 Welch-PSD features per overlapping window."""
    feats = []
    n = len(sig_1d)
    limit = n - 1 if strict_less else n
    for i in range(0, n, step):
        end = i + window
        if (strict_less and end < n) or (not strict_less and end <= n):
            seg = sig_1d[i:end]
            _, psd = signal.welch(seg, nperseg=min(100, window), average="median")
            feats.append(
                [
                    float(np.mean(seg)),
                    float(np.std(seg, ddof=1)),
                    float(np.abs(np.max(seg))),
                    float(np.abs(np.min(seg))),
                    float(kurtosis(seg, fisher=True, bias=False)),
                    float(skew(seg, bias=False)),
                    float(np.mean(psd)),
                    float(np.std(psd)),
                    float(kurtosis(psd)),
                    float(skew(psd)),
                    float(np.sqrt(np.max(psd))),
                    float(np.sqrt(np.min(psd))),
                ]
            )
    return np.array(feats, dtype=np.float64)


def make_regularized_rf(random_state: int = 123) -> RandomForestClassifier:
    """Instantiate the fixed shallow-tree regularized Random Forest reported in the paper."""
    return RandomForestClassifier(
        n_jobs=-1,
        random_state=random_state,
        max_depth=5,
        min_samples_leaf=5,
        min_samples_split=100,
        n_estimators=100,
    )


def tune_nested_rf(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    groups: np.ndarray,
    param_grid: dict,
    n_splits: int = 4,
    random_state: int = 123,
) -> GridSearchCV:
    """Run inner GroupKFold GridSearchCV strictly within an outer training fold."""
    grid = GridSearchCV(
        RandomForestClassifier(n_jobs=-1, random_state=random_state),
        param_grid=param_grid,
        cv=GroupKFold(n_splits=n_splits),
        scoring="accuracy",
        n_jobs=-1,
    )
    grid.fit(X_tr, y_tr, groups=groups)
    return grid


def build_classifier_zoo(tuned_rf, random_state: int = 123) -> Dict[str, object]:
    """Return the eight classifier configurations benchmarked under leak-free validation."""
    return {
        "Random Forest (Paper)": make_regularized_rf(random_state=random_state),
        "Random Forest (Tuned)": tuned_rf,
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=random_state)),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_state)),
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "MLP (ANN)": make_pipeline(
            StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=random_state)
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
    }


def evaluate_fold_models(
    fold_models: Dict[str, object],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    metrics_store: Dict[str, Dict[str, List[float]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit baselines, predict on X_te, append fold metrics, and return (y_te, tuned_rf_pred)."""
    tuned_pred = None
    for name, clf in fold_models.items():
        if name != "Random Forest (Tuned)":
            clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)
        prob = clf.predict_proba(X_te)
        if name == "Random Forest (Tuned)":
            tuned_pred = pred
        metrics_store[name]["acc"].append(accuracy_score(y_te, pred) * 100.0)
        metrics_store[name]["prec"].append(precision_score(y_te, pred, average="macro"))
        metrics_store[name]["rec"].append(recall_score(y_te, pred, average="macro"))
        metrics_store[name]["f1"].append(f1_score(y_te, pred, average="macro"))
        metrics_store[name]["auc"].append(roc_auc_score(y_te, prob, multi_class="ovr", average="macro"))
    return y_te, tuned_pred


def summarize_metrics(
    metrics_store: Dict[str, Dict[str, List[float]]],
    names: List[str] | None = None,
    scenario: str | None = None,
) -> pd.DataFrame:
    """Aggregate per-fold metrics into a summary DataFrame matching repository CSV outputs."""
    rows = []
    for name in (names or MODEL_NAMES):
        r = metrics_store[name]
        row = {}
        if scenario is not None:
            row["scenario"] = scenario
        row.update(
            {
                "model": name,
                "acc_mean": round(float(np.mean(r["acc"])), 2),
                "acc_std": round(float(np.std(r["acc"])), 2),
                "macro_precision": round(float(np.mean(r["prec"])), 3),
                "macro_recall": round(float(np.mean(r["rec"])), 3),
                "macro_f1": round(float(np.mean(r["f1"])), 3),
                "macro_roc_auc": round(float(np.mean(r["auc"])), 3),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
