"""Command-line entry point for reproducible model training."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.model_training import (
    DEFAULT_BUNDLE_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_PLOTS_DIR,
    DEFAULT_REPORT_PATH,
    train_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the Student Performance Predictor.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="CSV with seven features and math_score.")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE_PATH, help="Output versioned model bundle.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH, help="Machine-readable JSON experiment report.")
    parser.add_argument("--plots-dir", type=Path, default=DEFAULT_PLOTS_DIR, help="Directory for holdout and importance plots.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the holdout split, CV, and estimators.")
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of shuffled CV folds on development data.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Final untouched holdout fraction.")
    parser.add_argument("--no-optional-models", action="store_true", help="Skip XGBoost and CatBoost explicitly.")
    parser.add_argument("--no-plots", action="store_true", help="Skip writing evaluation plots.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_model(
        data_path=args.data,
        bundle_path=args.bundle,
        report_path=args.report,
        plots_dir=args.plots_dir,
        seed=args.seed,
        cv_splits=args.cv_splits,
        test_size=args.test_size,
        include_optional=not args.no_optional_models,
        generate_plots=not args.no_plots,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
