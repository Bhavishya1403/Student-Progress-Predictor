from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app import create_app
from src.model_bundle import ModelBundle, ModelBundleError, load_model_bundle
from src.model_training import DEFAULT_DATA_PATH, ModelSpec, make_preprocessor, train_model
from src.pipeline.predict_pipeline import PredictPipeline
from src.schema import DataValidationError, FEATURE_COLUMNS, validate_prediction_frame
from sklearn.linear_model import LinearRegression


VALID_ROW = {
    "gender": "female",
    "race_ethnicity": "group C",
    "parental_level_of_education": "some college",
    "lunch": "standard",
    "test_preparation_course": "completed",
    "reading_score": 78,
    "writing_score": 76,
}


class TrainingAndPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp_dir.name)
        cls.bundle_path = cls.root / "model_bundle.pkl"
        cls.report_path = cls.root / "model_report.json"
        cls.report = train_model(
            data_path=DEFAULT_DATA_PATH,
            bundle_path=cls.bundle_path,
            report_path=cls.report_path,
            plots_dir=cls.root / "plots",
            seed=42,
            cv_splits=3,
            model_specs=[ModelSpec("LinearRegression", LinearRegression(), {})],
            generate_plots=False,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_preprocessor_ignores_valid_category_not_seen_during_fit(self):
        data = pd.read_csv(DEFAULT_DATA_PATH)
        train = data.loc[:7, FEATURE_COLUMNS]
        transformed = make_preprocessor().fit(train).transform(
            pd.DataFrame([{**VALID_ROW, "race_ethnicity": "group E"}])
        )
        self.assertEqual(transformed.shape[0], 1)
        self.assertGreater(transformed.shape[1], 0)

    def test_prediction_validation_rejects_bad_values(self):
        bad_score = {**VALID_ROW, "reading_score": 101}
        with self.assertRaises(DataValidationError):
            validate_prediction_frame(pd.DataFrame([bad_score]))

        bad_category = {**VALID_ROW, "gender": "unknown"}
        with self.assertRaises(DataValidationError):
            validate_prediction_frame(pd.DataFrame([bad_category]))

        with self.assertRaises(DataValidationError):
            validate_prediction_frame(pd.DataFrame([{k: v for k, v in VALID_ROW.items() if k != "lunch"}]))

    def test_training_writes_bundle_and_detailed_cv_report(self):
        self.assertTrue(self.bundle_path.is_file())
        self.assertTrue(self.report_path.is_file())
        report = json.loads(self.report_path.read_text(encoding="utf-8"))
        candidate = report["candidate_searches"]["LinearRegression"][0]
        self.assertEqual(len(candidate["folds"]), 3)
        self.assertEqual(set(candidate["folds"][0]["validation"]), {"r2", "mae", "rmse"})
        self.assertIn("untouched_holdout", report["metadata"]["metrics"])
        self.assertNotIn("test", report["metadata"]["cross_validation"])

    def test_bundle_schema_and_prediction(self):
        bundle = ModelBundle.load(self.bundle_path)
        frame = validate_prediction_frame(pd.DataFrame([VALID_ROW]))
        prediction = bundle.predict(frame)
        self.assertEqual(prediction.shape, (1,))
        self.assertTrue(0 <= float(prediction[0]) <= 100)

        predicted, explanation = bundle.predict_with_explanation(frame)
        self.assertEqual(predicted.shape, (1,))
        self.assertEqual(len(explanation[0]), len(FEATURE_COLUMNS))
        self.assertTrue(all("change_from_estimate" in item for item in explanation[0]))

    def test_incompatible_bundle_is_rejected(self):
        original = load_model_bundle(self.bundle_path)
        original["metadata"]["feature_columns"] = ["wrong"]
        invalid_path = self.root / "invalid_bundle.pkl"
        with invalid_path.open("wb") as handle:
            pickle.dump(original, handle)
        with self.assertRaises(ModelBundleError):
            load_model_bundle(invalid_path)

    def test_flask_prediction_route_and_server_validation(self):
        app = create_app(PredictPipeline(self.bundle_path))
        app.testing = True
        client = app.test_client()
        self.assertEqual(client.get("/predictdata").status_code, 200)

        result = client.post("/predictdata", data=VALID_ROW)
        self.assertEqual(result.status_code, 200)
        self.assertIn(b"Model estimate", result.data)
        self.assertIn(b"Local sensitivity checks", result.data)

        missing = {**VALID_ROW, "reading_score": ""}
        bad = client.post("/predictdata", data=missing)
        self.assertEqual(bad.status_code, 400)
        self.assertIn(b"Please provide", bad.data)

        bad_range = {**VALID_ROW, "writing_score": "101"}
        bad = client.post("/predictdata", data=bad_range)
        self.assertEqual(bad.status_code, 400)
        self.assertIn(b"between 0 and 100", bad.data)

        bad_category = {**VALID_ROW, "gender": "unknown"}
        bad = client.post("/predictdata", data=bad_category)
        self.assertEqual(bad.status_code, 400)
        self.assertIn(b"Unsupported value", bad.data)

    def test_missing_artifact_returns_service_unavailable(self):
        missing_path = self.root / "missing.pkl"
        app = create_app(PredictPipeline(missing_path))
        app.testing = True
        client = app.test_client()
        result = client.post("/predictdata", data=VALID_ROW)
        self.assertEqual(result.status_code, 503)
        self.assertIn(b"model bundle is not available", result.data)

    def test_evaluation_page_reads_bundle_metadata(self):
        app = create_app(PredictPipeline(self.bundle_path))
        client = app.test_client()
        response = client.get("/evaluation")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Final untouched holdout", response.data)
        self.assertIn(b"Candidate comparison", response.data)


if __name__ == "__main__":
    unittest.main()
