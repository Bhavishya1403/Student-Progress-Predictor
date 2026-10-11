"""Flask interface for the Student Performance Predictor."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, abort, render_template, request, send_from_directory

from src.model_bundle import ModelBundleError
from src.pipeline.predict_pipeline import PredictPipeline
from src.schema import DataValidationError, FEATURE_COLUMNS, SUPPORTED_CATEGORIES, validate_form_values

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
PLOT_DIRECTORY = PROJECT_ROOT / "artifacts" / "evaluation"
PLOT_FILES = {
    "actual_vs_predicted.png",
    "residuals.png",
    "error_distribution.png",
    "permutation_importance.png",
}


def create_app(predictor: PredictPipeline | None = None) -> Flask:
    flask_app = Flask(__name__)
    flask_app.config.update(MAX_CONTENT_LENGTH=2 * 1024 * 1024)
    prediction_service = predictor or PredictPipeline()

    def home_context(**extra):
        return {
            "feature_columns": FEATURE_COLUMNS,
            "category_options": SUPPORTED_CATEGORIES,
            "model_status": prediction_service.status(),
            **extra,
        }

    @flask_app.get("/")
    def index():
        return render_template("index.html", model_status=prediction_service.status())

    @flask_app.route("/predictdata", methods=["GET", "POST"])
    def predict_datapoint():
        if request.method == "GET":
            return render_template("home.html", **home_context(form_values={}))

        form_values = request.form.to_dict(flat=True)
        # Keep the old form field working while the template uses the schema name.
        if not form_values.get("race_ethnicity"):
            form_values["race_ethnicity"] = form_values.get("ethnicity")
        try:
            features = validate_form_values(form_values)
        except DataValidationError as exc:
            return (
                render_template(
                    "home.html",
                    **home_context(form_values=form_values, validation_error=str(exc)),
                ),
                400,
            )

        status = prediction_service.status()
        if not status["available"]:
            return (
                render_template(
                    "home.html",
                    **home_context(form_values=form_values, service_error=status["message"]),
                ),
                503,
            )

        try:
            predictions, explanations = prediction_service.predict_with_explanation(features)
            return render_template(
                "home.html",
                **home_context(
                    form_values=form_values,
                    prediction=float(predictions[0]),
                    explanation=explanations[0],
                ),
            )
        except DataValidationError as exc:
            return (
                render_template(
                    "home.html",
                    **home_context(form_values=form_values, validation_error=str(exc)),
                ),
                400,
            )
        except ModelBundleError:
            logger.exception("Prediction model bundle could not be loaded")
            return (
                render_template(
                    "home.html",
                    **home_context(
                        form_values=form_values,
                        service_error="The model bundle is unavailable or incompatible. Retrain it and restart the app.",
                    ),
                ),
                503,
            )
        except Exception:
            logger.exception("Prediction request failed")
            return (
                render_template(
                    "home.html",
                    **home_context(
                        form_values=form_values,
                        service_error="Prediction failed unexpectedly. Please try again later.",
                    ),
                ),
                500,
            )

    @flask_app.get("/evaluation")
    def evaluation_dashboard():
        status = prediction_service.status()
        if not status["available"]:
            return render_template("evaluation.html", model_status=status), 503
        try:
            metadata = prediction_service.metadata
        except ModelBundleError:
            logger.exception("Evaluation metadata could not be loaded")
            return render_template(
                "evaluation.html",
                model_status={"available": False, "message": "Model metadata is unavailable."},
            ), 503

        plot_files = [name for name in sorted(PLOT_FILES) if (PLOT_DIRECTORY / name).is_file()]
        return render_template(
            "evaluation.html",
            model_status=status,
            metadata=metadata,
            plot_files=plot_files,
        )

    @flask_app.get("/evaluation/plots/<filename>")
    def evaluation_plot(filename: str):
        if filename not in PLOT_FILES:
            abort(404)
        return send_from_directory(PLOT_DIRECTORY, filename, conditional=True)

    return flask_app


application = create_app()
app = application


if __name__ == "__main__":
    application.run(host="0.0.0.0", port=5000, debug=True)
