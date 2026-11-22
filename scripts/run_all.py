"""End-to-end reproduction runner for all supplementary experiments and audits."""

from __future__ import annotations

import importlib

MODULES = [
    "04_verify_dataset_and_rf",
    "01_leakfree_5fold_benchmark",
    "02_unified_fieldtest_cv",
    "03_split_first_raw_ablation",
]


def main() -> None:
    for mod_name in MODULES:
        print(f"\n{'=' * 72}\nRunning {mod_name}\n{'=' * 72}")
        mod = importlib.import_module(mod_name)
        mod.main()


if __name__ == "__main__":
    main()
