"""Versioned model bundle loading and prediction helpers."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.schema import FEATURE_COLUMNS, validate_prediction_frame


class ModelBundleError(RuntimeError):
    """Raised when a model bundle is absent, malformed, or incompatible."""


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    bundle_path = Path(path)
    if not bundle_path.is_file():
        raise ModelBundleError("Model bundle is missing. Train the model with `python -m src.train`.")
    try:
        # This loader is only for artifacts produced by this project's training command.
        with bundle_path.open("rb") as handle:
            bundle = pickle.load(handle)
    except Exception as exc:
        raise ModelBundleError("Model bundle could not be loaded; retrain it to create a compatible bundle.") from exc

    if not isinstance(bundle, dict):
        raise ModelBundleError("Model bundle has an unsupported format.")
    if bundle.get("artifact_type") != "student-performance-regression":
        raise ModelBundleError("Model bundle has an unsupported artifact type.")
    if bundle.get("schema_version") != 1:
        raise ModelBundleError("Model bundle schema version is not supported.")

    metadata = bundle.get("metadata")
    pipeline = bundle.get("pipeline")
    if not isinstance(metadata, dict) or not callable(getattr(pipeline, "predict", None)):
        raise ModelBundleError("Model bundle is missing its metadata or prediction pipeline.")
    if metadata.get("schema_version") != bundle["schema_version"]:
        raise ModelBundleError("Model and metadata schema versions do not match.")
    if metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise ModelBundleError("Model bundle feature schema does not match the application.")
    if metadata.get("target_column") != "math_score":
        raise ModelBundleError("Model bundle target does not match the application.")

    fitted_features = getattr(pipeline, "feature_names_in_", None)
    if fitted_features is None or list(fitted_features) != FEATURE_COLUMNS:
        raise ModelBundleError("Fitted model feature order does not match the application schema.")
    return bundle


class ModelBundle:
    def __init__(self, bundle: dict[str, Any]):
        self.pipeline = bundle["pipeline"]
        self.metadata = bundle["metadata"]

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        return cls(load_model_bundle(path))

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        validated = validate_prediction_frame(features)
        return np.asarray(self.pipeline.predict(validated), dtype=float)

    def predict_with_explanation(self, features: pd.DataFrame) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
        """Return per-row single-feature sensitivity checks against training references."""
        validated = validate_prediction_frame(features)
        predictions = np.asarray(self.pipeline.predict(validated), dtype=float)
        references = self.metadata.get("local_reference_values", {})
        explanations: list[list[dict[str, Any]]] = []

        for row_index, (_, row) in enumerate(validated.iterrows()):
            row_frame = pd.DataFrame([row], columns=FEATURE_COLUMNS)
            row_explanations = []
            for feature in FEATURE_COLUMNS:
                if feature not in references:
                    continue
                comparison = row_frame.copy()
                comparison.loc[:, feature] = references[feature]
                comparison_prediction = float(self.pipeline.predict(comparison)[0])
                estimate = float(predictions[row_index])
                row_explanations.append(
                    {
                        "feature": feature,
                        "observed_value": row[feature].item() if hasattr(row[feature], "item") else row[feature],
                        "reference_value": references[feature],
                        "prediction_at_reference": comparison_prediction,
                        "change_from_estimate": comparison_prediction - estimate,
                    }
                )
            row_explanations.sort(key=lambda item: abs(item["change_from_estimate"]), reverse=True)
            explanations.append(row_explanations)
        return predictions, explanations
