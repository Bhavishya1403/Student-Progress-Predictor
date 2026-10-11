"""Cached prediction entry point for the versioned model bundle."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.model_bundle import ModelBundle, ModelBundleError
from src.schema import FEATURE_COLUMNS, validate_prediction_frame


DEFAULT_BUNDLE_PATH = Path(__file__).resolve().parents[2] / "artifacts" / "model_bundle.pkl"


class PredictPipeline:
    def __init__(self, bundle_path: str | Path | None = None):
        configured_path = bundle_path or os.environ.get("MODEL_BUNDLE_PATH") or DEFAULT_BUNDLE_PATH
        self.bundle_path = Path(configured_path).expanduser().resolve()
        self._bundle: ModelBundle | None = None

    def _get_bundle(self) -> ModelBundle:
        if self._bundle is None:
            self._bundle = ModelBundle.load(self.bundle_path)
        return self._bundle

    def status(self) -> dict[str, Any]:
        try:
            metadata = self._get_bundle().metadata
            return {
                "available": True,
                "message": "Model bundle loaded and schema verified.",
                "model_version": metadata.get("model_version"),
                "model_name": metadata.get("model_name"),
            }
        except ModelBundleError:
            return {
                "available": False,
                "message": "A compatible model bundle is not available. Run `python -m src.train`.",
                "model_version": None,
                "model_name": None,
            }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._get_bundle().metadata

    def predict(self, features: pd.DataFrame):
        validated = validate_prediction_frame(features)
        return self._get_bundle().predict(validated)

    def predict_with_explanation(self, features: pd.DataFrame):
        validated = validate_prediction_frame(features)
        return self._get_bundle().predict_with_explanation(validated)


class CustomData:
    """Compatibility wrapper that builds a schema-ordered prediction row."""

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float,
    ):
        self.values = {
            "gender": gender,
            "race_ethnicity": race_ethnicity,
            "parental_level_of_education": parental_level_of_education,
            "lunch": lunch,
            "test_preparation_course": test_preparation_course,
            "reading_score": reading_score,
            "writing_score": writing_score,
        }

    def get_data_as_data_frame(self) -> pd.DataFrame:
        return validate_prediction_frame(pd.DataFrame([self.values], columns=FEATURE_COLUMNS))
