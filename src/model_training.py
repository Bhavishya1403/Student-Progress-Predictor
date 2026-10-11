"""Leakage-safe, reproducible training and evaluation for the student score model."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import platform
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as package_metadata
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import permutation_importance

from src.schema import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    SUPPORTED_CATEGORIES,
    TARGET_COLUMN,
    reference_values,
    validate_training_frame,
)

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "notebook" / "data" / "stud.csv"
DEFAULT_BUNDLE_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model_bundle.pkl"
DEFAULT_REPORT_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model_report.json"
DEFAULT_PLOTS_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "evaluation"
SCHEMA_VERSION = 1
SCORING = {
    "r2": "r2",
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    param_grid: dict[str, list[Any]]


def make_preprocessor() -> ColumnTransformer:
    """Create transformations; fitting is delegated to each CV fold's Pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_model_specs(seed: int = 42, include_optional: bool = True) -> tuple[list[ModelSpec], dict[str, str]]:
    """Return the requested model families and explicit optional-dependency skips."""
    specs = [
        ModelSpec("DummyRegressor", DummyRegressor(strategy="mean"), {}),
        ModelSpec("LinearRegression", LinearRegression(), {}),
        ModelSpec(
            "DecisionTreeRegressor",
            DecisionTreeRegressor(random_state=seed),
            {"model__max_depth": [None, 5, 10], "model__min_samples_leaf": [1, 5]},
        ),
        ModelSpec(
            "RandomForestRegressor",
            RandomForestRegressor(random_state=seed, n_jobs=1),
            {
                "model__n_estimators": [100, 250],
                "model__max_depth": [None, 8],
                "model__min_samples_leaf": [1, 3],
            },
        ),
        ModelSpec(
            "GradientBoostingRegressor",
            GradientBoostingRegressor(random_state=seed),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
        ModelSpec(
            "AdaBoostRegressor",
            AdaBoostRegressor(random_state=seed),
            {
                "model__n_estimators": [50, 100, 200],
                "model__learning_rate": [0.05, 0.5],
            },
        ),
    ]
    skipped: dict[str, str] = {}

    if include_optional:
        try:
            from xgboost import XGBRegressor

            specs.append(
                ModelSpec(
                    "XGBRegressor",
                    XGBRegressor(random_state=seed, n_jobs=1, verbosity=0),
                    {
                        "model__n_estimators": [100, 200],
                        "model__learning_rate": [0.05, 0.1],
                        "model__max_depth": [2, 4],
                    },
                )
            )
        except Exception as exc:  # optional package may be absent or binary-incompatible
            skipped["XGBRegressor"] = f"Optional dependency unavailable: {exc}"

        try:
            from catboost import CatBoostRegressor

            specs.append(
                ModelSpec(
                    "CatBoostRegressor",
                    CatBoostRegressor(
                        random_seed=seed,
                        verbose=False,
                        allow_writing_files=False,
                        thread_count=1,
                    ),
                    {
                        "model__iterations": [100, 200],
                        "model__learning_rate": [0.03, 0.1],
                        "model__depth": [4, 6],
                    },
                )
            )
        except Exception as exc:
            skipped["CatBoostRegressor"] = f"Optional dependency unavailable: {exc}"

    return specs, skipped


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _metric_value(cv_results: dict[str, Any], key: str, index: int, metric: str) -> float:
    value = float(cv_results[f"{key}_{metric}"][index])
    return -value if metric in ("mae", "rmse") else value


def _candidate_records(cv_results: dict[str, Any], n_splits: int) -> list[dict[str, Any]]:
    records = []
    for candidate_index, params in enumerate(cv_results["params"]):
        record: dict[str, Any] = {"params": _json_safe(params), "folds": []}
        for split_index in range(n_splits):
            fold: dict[str, Any] = {"fold": split_index + 1, "train": {}, "validation": {}}
            for metric in SCORING:
                fold["train"][metric] = _metric_value(
                    cv_results, f"split{split_index}_train", candidate_index, metric
                )
                fold["validation"][metric] = _metric_value(
                    cv_results, f"split{split_index}_test", candidate_index, metric
                )
            record["folds"].append(fold)
        record["cv"] = {
            "train": {
                metric: {
                    "mean": _metric_value(cv_results, "mean_train", candidate_index, metric),
                    "std": float(cv_results[f"std_train_{metric}"][candidate_index]),
                }
                for metric in SCORING
            },
            "validation": {
                metric: {
                    "mean": _metric_value(cv_results, "mean_test", candidate_index, metric),
                    "std": float(cv_results[f"std_test_{metric}"][candidate_index]),
                }
                for metric in SCORING
            },
        }
        record["mean_fit_time_seconds"] = float(cv_results["mean_fit_time"][candidate_index])
        records.append(record)
    return _json_safe(records)


def _atomic_write_json(path: Path, content: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(_json_safe(content), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _atomic_write_bundle(path: Path, bundle: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False, suffix=".tmp") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def _save_evaluation_plots(
    plots_dir: Path,
    y_true: Sequence[float],
    y_pred: Sequence[float],
    importance: list[dict[str, float]],
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    residual = actual - predicted

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.scatter(actual, predicted, alpha=0.65, color="#2563eb", edgecolor="white", linewidth=0.4)
    low = min(float(actual.min()), float(predicted.min()))
    high = max(float(actual.max()), float(predicted.max()))
    ax.plot([low, high], [low, high], linestyle="--", color="#b45309", label="Ideal prediction")
    ax.set(xlabel="Actual math score", ylabel="Predicted math score", title="Holdout: actual vs predicted")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plots_dir / "actual_vs_predicted.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.scatter(predicted, residual, alpha=0.65, color="#0f766e", edgecolor="white", linewidth=0.4)
    ax.axhline(0, linestyle="--", color="#b45309")
    ax.set(xlabel="Predicted math score", ylabel="Residual (actual − predicted)", title="Holdout residuals")
    fig.tight_layout()
    fig.savefig(plots_dir / "residuals.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.hist(residual, bins=18, color="#7c3aed", alpha=0.82, edgecolor="white")
    ax.axvline(0, linestyle="--", color="#b45309")
    ax.set(xlabel="Residual (actual − predicted)", ylabel="Number of students", title="Holdout error distribution")
    fig.tight_layout()
    fig.savefig(plots_dir / "error_distribution.png", dpi=150)
    plt.close(fig)

    if importance:
        ranked = sorted(importance, key=lambda item: item["importance_mean"], reverse=True)
        labels = [item["feature"] for item in ranked]
        means = [item["importance_mean"] for item in ranked]
        errors = [item["importance_std"] for item in ranked]
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.barh(labels[::-1], means[::-1], xerr=errors[::-1], color="#0891b2", alpha=0.85)
        ax.axvline(0, color="#475569", linewidth=0.8)
        ax.set(xlabel="Decrease in training R² after permutation", title="Global permutation importance (training data)")
        fig.tight_layout()
        fig.savefig(plots_dir / "permutation_importance.png", dpi=150)
        plt.close(fig)


def train_model(
    data_path: str | Path = DEFAULT_DATA_PATH,
    bundle_path: str | Path = DEFAULT_BUNDLE_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    plots_dir: str | Path = DEFAULT_PLOTS_DIR,
    *,
    seed: int = 42,
    cv_splits: int = 5,
    test_size: float = 0.2,
    model_specs: list[ModelSpec] | None = None,
    include_optional: bool = True,
    generate_plots: bool = True,
) -> dict[str, Any]:
    """Tune on development data, evaluate the selected pipeline once on holdout, save bundle/report."""
    data_path = Path(data_path).resolve()
    bundle_path = Path(bundle_path).resolve()
    report_path = Path(report_path).resolve()
    plots_dir = Path(plots_dir).resolve()

    raw_bytes = data_path.read_bytes()
    data = validate_training_frame(pd.read_csv(data_path))
    if len(data) <= cv_splits:
        raise ValueError(f"Need more than {cv_splits} rows for {cv_splits}-fold CV.")

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    if model_specs is None:
        specs, unavailable = make_model_specs(seed=seed, include_optional=include_optional)
    else:
        specs, unavailable = model_specs, {}

    if not specs:
        raise RuntimeError("No estimator specifications are available.")

    searches: dict[str, Any] = {}
    candidate_results: dict[str, list[dict[str, Any]]] = {}
    comparison: list[dict[str, Any]] = []
    failed_models = dict(unavailable)

    for spec in specs:
        estimator_pipeline = Pipeline(
            steps=[("preprocessor", make_preprocessor()), ("model", spec.estimator)]
        )
        search = GridSearchCV(
            estimator=estimator_pipeline,
            param_grid=spec.param_grid,
            scoring=SCORING,
            refit="r2",
            cv=cv,
            n_jobs=1,
            return_train_score=True,
            error_score="raise",
        )
        try:
            search.fit(X_train, y_train)
        except Exception as exc:
            logger.exception("Model search failed for %s", spec.name)
            failed_models[spec.name] = f"Search failed: {type(exc).__name__}: {exc}"
            continue

        searches[spec.name] = search
        candidate_results[spec.name] = _candidate_records(search.cv_results_, cv_splits)
        index = int(search.best_index_)
        best_candidate = candidate_results[spec.name][index]
        comparison.append(
            {
                "model": spec.name,
                "selected_params": _json_safe(search.best_params_),
                "cv": best_candidate["cv"]["validation"],
                "cv_train": best_candidate["cv"]["train"],
                "candidate_count": len(search.cv_results_["params"]),
            }
        )

    if not searches:
        raise RuntimeError("Every model search failed; no artifact was written.")

    selected_name = max(
        searches,
        key=lambda name: float(searches[name].cv_results_["mean_test_r2"][searches[name].best_index_]),
    )
    selected_search = searches[selected_name]
    selected_pipeline = selected_search.best_estimator_

    # The holdout is not used for tuning or importance. Its predictions are made once here.
    holdout_predictions = selected_pipeline.predict(X_test)
    holdout_metrics = regression_metrics(y_test, holdout_predictions)
    train_predictions = selected_pipeline.predict(X_train)
    train_metrics = regression_metrics(y_train, train_predictions)

    permutation = permutation_importance(
        selected_pipeline,
        X_train,
        y_train,
        scoring="r2",
        n_repeats=5,
        random_state=seed,
        n_jobs=1,
    )
    global_importance = [
        {
            "feature": name,
            "importance_mean": float(permutation.importances_mean[index]),
            "importance_std": float(permutation.importances_std[index]),
        }
        for index, name in enumerate(FEATURE_COLUMNS)
    ]
    global_importance.sort(key=lambda item: item["importance_mean"], reverse=True)

    selected_candidate = candidate_results[selected_name][int(selected_search.best_index_)]
    training_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    dataset_hash = hashlib.sha256(raw_bytes).hexdigest()
    model_version = hashlib.sha256(
        f"{dataset_hash}:{seed}:{selected_name}:{selected_search.best_params_}".encode("utf-8")
    ).hexdigest()[:12]
    versions = {"python": platform.python_version(), "scikit-learn": sklearn.__version__}
    for distribution in ("Flask", "numpy", "pandas", "xgboost", "catboost", "matplotlib"):
        try:
            versions[distribution] = package_metadata.version(distribution)
        except package_metadata.PackageNotFoundError:
            continue

    metadata: dict[str, Any] = {
        "artifact_type": "student-performance-regression",
        "schema_version": SCHEMA_VERSION,
        "model_version": model_version,
        "trained_at_utc": training_time,
        "dataset_source": "notebook/data/stud.csv (notebook attributes source to Kaggle Students Performance in Exams)",
        "dataset_sha256": dataset_hash,
        "dataset_rows": int(len(data)),
        "train_rows": int(len(X_train)),
        "holdout_rows": int(len(X_test)),
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "categorical_features": list(CATEGORICAL_FEATURES),
        "numeric_features": list(NUMERIC_FEATURES),
        "supported_categories": SUPPORTED_CATEGORIES,
        "model_name": selected_name,
        "model_type": f"{type(selected_pipeline.named_steps['model']).__module__}.{type(selected_pipeline.named_steps['model']).__qualname__}",
        "selected_hyperparameters": _json_safe(selected_search.best_params_),
        "selection_metric": "mean cross-validated R²",
        "split": {"test_size": test_size, "random_state": seed, "shuffle": True},
        "cross_validation": {
            "splitter": "KFold",
            "n_splits": cv_splits,
            "shuffle": True,
            "random_state": seed,
            "scoring": SCORING,
            "refit": "r2",
        },
        "dependency_versions": versions,
        "python_version": platform.python_version(),
        "metrics": {
            "selected_cv_validation": selected_candidate["cv"]["validation"],
            "selected_cv_training": selected_candidate["cv"]["train"],
            "development_training": train_metrics,
            "untouched_holdout": holdout_metrics,
        },
        "model_comparison": sorted(comparison, key=lambda item: item["cv"]["r2"]["mean"], reverse=True),
        "skipped_models": failed_models,
        "global_permutation_importance": global_importance,
        "importance_data": "development training rows; descriptive global importance, not causal and not an individual explanation",
        "local_reference_values": reference_values(X_train),
        "holdout_predictions_available": True,
        "explanation_method": "single-feature sensitivity to training median/mode reference values; effects are not additive or causal",
    }
    report = {
        "metadata": metadata,
        "candidate_searches": candidate_results,
        "notes": [
            "The final holdout was not used in hyperparameter or model selection.",
            "Only the selected pipeline was evaluated on the final holdout.",
            "Cross-validation preprocessing is inside the estimator Pipeline and is refitted within each fold.",
        ],
    }
    bundle = {
        "artifact_type": metadata["artifact_type"],
        "schema_version": SCHEMA_VERSION,
        "pipeline": selected_pipeline,
        "metadata": metadata,
    }

    _atomic_write_bundle(bundle_path, bundle)
    _atomic_write_json(report_path, report)
    if generate_plots:
        _save_evaluation_plots(plots_dir, y_test, holdout_predictions, global_importance)

    print(f"Selected model: {selected_name}")
    print(f"CV R²: {metadata['metrics']['selected_cv_validation']['r2']['mean']:.4f} ± {metadata['metrics']['selected_cv_validation']['r2']['std']:.4f}")
    print(f"Holdout R²: {holdout_metrics['r2']:.4f}; MAE: {holdout_metrics['mae']:.3f}; RMSE: {holdout_metrics['rmse']:.3f}")
    print(f"Bundle: {bundle_path}")
    print(f"Report: {report_path}")
    if failed_models:
        print("Skipped models:")
        for name, reason in failed_models.items():
            print(f"  - {name}: {reason}")
    return report
