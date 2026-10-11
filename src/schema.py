"""Shared input schema and validation for training and prediction."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd


TARGET_COLUMN = "math_score"
FEATURE_COLUMNS = [
    "gender",
    "race_ethnicity",
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
    "reading_score",
    "writing_score",
]
CATEGORICAL_FEATURES = FEATURE_COLUMNS[:5]
NUMERIC_FEATURES = ["reading_score", "writing_score"]

# Domain values represented in the source dataset and prediction form.
SUPPORTED_CATEGORIES = {
    "gender": ["female", "male"],
    "race_ethnicity": ["group A", "group B", "group C", "group D", "group E"],
    "parental_level_of_education": [
        "associate's degree",
        "bachelor's degree",
        "high school",
        "master's degree",
        "some college",
        "some high school",
    ],
    "lunch": ["free/reduced", "standard"],
    "test_preparation_course": ["completed", "none"],
}


class DataValidationError(ValueError):
    """Raised when a training or prediction row violates the project schema."""


def _check_frame_columns(frame: pd.DataFrame, expected: list[str]) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError("Input must be a pandas DataFrame.")
    if frame.columns.duplicated().any():
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise DataValidationError(f"Duplicate input columns: {duplicates}.")

    missing = [name for name in expected if name not in frame.columns]
    extra = [name for name in frame.columns if name not in expected]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing columns: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected columns: {', '.join(extra)}")
        raise DataValidationError("Invalid input schema (" + "; ".join(details) + ").")


def _validate_numeric_series(
    series: pd.Series,
    name: str,
    *,
    allow_missing: bool,
) -> pd.Series:
    try:
        values = pd.to_numeric(series, errors="raise")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{name} must contain numeric values.") from exc

    missing = values.isna()
    if missing.any() and not allow_missing:
        raise DataValidationError(f"{name} is required.")

    present = values[~missing].astype(float)
    if not np.isfinite(present.to_numpy()).all():
        raise DataValidationError(f"{name} must contain finite values.")
    if ((present < 0) | (present > 100)).any():
        raise DataValidationError(f"{name} must be between 0 and 100.")
    return values


def _validate_categorical_series(
    series: pd.Series,
    name: str,
    *,
    allow_missing: bool,
) -> pd.Series:
    missing = series.isna()
    if missing.any() and not allow_missing:
        raise DataValidationError(f"{name} is required.")

    values = series.copy()
    present = values[~missing].astype(str)
    unsupported = sorted(set(present.unique()) - set(SUPPORTED_CATEGORIES[name]))
    if unsupported:
        raise DataValidationError(
            f"Unsupported value for {name}: {', '.join(unsupported)}."
        )
    values.loc[~missing] = present
    return values


def validate_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate a source dataset while allowing missing input features for imputation."""
    expected = [*FEATURE_COLUMNS, TARGET_COLUMN]
    _check_frame_columns(frame, expected)
    result = frame.loc[:, expected].copy()

    if result.empty:
        raise DataValidationError("Training dataset contains no rows.")
    result[TARGET_COLUMN] = _validate_numeric_series(
        result[TARGET_COLUMN], TARGET_COLUMN, allow_missing=False
    )
    for name in NUMERIC_FEATURES:
        result[name] = _validate_numeric_series(result[name], name, allow_missing=True)
    for name in CATEGORICAL_FEATURES:
        result[name] = _validate_categorical_series(
            result[name], name, allow_missing=True
        )
    return result


def validate_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and reorder one or more complete prediction rows."""
    _check_frame_columns(frame, FEATURE_COLUMNS)
    result = frame.loc[:, FEATURE_COLUMNS].copy()
    if result.empty:
        raise DataValidationError("At least one prediction row is required.")
    for name in NUMERIC_FEATURES:
        result[name] = _validate_numeric_series(result[name], name, allow_missing=False)
    for name in CATEGORICAL_FEATURES:
        result[name] = _validate_categorical_series(
            result[name], name, allow_missing=False
        )
    return result


def validate_form_values(values: Mapping[str, Any]) -> pd.DataFrame:
    """Build a validated single-row frame from Flask form values."""
    row = {name: values.get(name) for name in FEATURE_COLUMNS}
    missing = [name for name, value in row.items() if value is None or str(value).strip() == ""]
    if missing:
        raise DataValidationError(f"Please provide: {', '.join(missing)}.")
    return validate_prediction_frame(pd.DataFrame([row]))


def reference_values(frame: pd.DataFrame) -> dict[str, Any]:
    """Return training-only median/mode values for local sensitivity comparisons."""
    references: dict[str, Any] = {}
    for name in NUMERIC_FEATURES:
        value = pd.to_numeric(frame[name], errors="coerce").median()
        if pd.isna(value):
            raise DataValidationError(f"Cannot create a reference value for {name}.")
        references[name] = float(value)
    for name in CATEGORICAL_FEATURES:
        modes = frame[name].dropna().astype(str).mode()
        if modes.empty:
            raise DataValidationError(f"Cannot create a reference value for {name}.")
        references[name] = str(modes.iloc[0])
    return references


def is_finite_number(value: Any) -> bool:
    """Small helper for callers that receive untrusted scalar values."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
