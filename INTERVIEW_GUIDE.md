# Interview Guide — Student Score Lab

## 60-second explanation

“Student Score Lab estimates a student's mathematics exam score from five categorical inputs plus reading and writing scores. The included dataset has 1,000 rows. I built a reproducible training command that validates the schema, holds out 20% for a final evaluation, and uses a scikit-learn `Pipeline` so imputation, scaling, and one-hot encoding are fitted inside each cross-validation fold. It compares seven regressors and a `DummyRegressor` baseline using five-fold CV with R², MAE, and RMSE. The selected model is Linear Regression, with CV R² 0.8690 ± 0.0139 and untouched holdout R² 0.8804, MAE 4.2148, and RMSE 5.3940. A Flask form loads a schema-checked model bundle, validates inputs on the server, and shows a model estimate plus clearly labeled one-feature sensitivity checks. It is a demonstration, not a decision tool, and its evaluation on this small dataset does not establish generalization to other student populations.”

## Two-minute explanation

“The project uses the Students Performance in Exams CSV copied into `notebook/data/stud.csv`; the EDA notebook attributes the source to Kaggle. The target is `math_score`; the inputs are gender, race/ethnicity, parental education, lunch, test preparation, reading score, and writing score. The training command rejects unexpected or missing columns, validates score ranges and target values, and uses an 80/20 split with seed 42.

“For each of five shuffled CV folds on the 800-row development split, the estimator pipeline fits its own numeric median imputer and scaler plus categorical most-frequent imputer, one-hot encoder, and sparse scaler. GridSearchCV uses explicit R², negative MAE, and negative RMSE scoring, refits the candidate with the best mean CV R², and records all parameter candidates, fold metrics, means, and standard deviations. I compare DummyRegressor with Linear Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, XGBoost, and CatBoost. Linear Regression won the measured run. I then evaluated the refitted pipeline once on the 200-row untouched holdout; I did not use that score for model selection.

“The selected model’s holdout R² was 0.8804. That is not 88% accuracy: it means squared error was 88.04% lower than the mean-prediction baseline on this particular holdout. I save the fitted preprocessing and estimator together with schema, versions, data hash, split/CV settings, metrics, and model version. Flask loads and caches that bundle, rejects malformed or unsupported inputs, and serves a separate evaluation page. The main caveats are the dataset’s size and limited scope, the use of demographic fields, and the fact that this evaluation does not prove performance on new cohorts or that the features are appropriate for a real decision.”

## End-to-end walkthrough

1. `src/train.py` parses the CLI options and calls `train_model`.
2. `src/schema.py` enforces the target and seven-feature contract, score bounds, and supported categories.
3. `src/model_training.py` splits raw rows into development and final holdout partitions using seed 42.
4. `make_preprocessor` builds a `ColumnTransformer` with numeric and categorical pipelines. It is inside each model's sklearn `Pipeline`, so `GridSearchCV` refits preprocessing in each CV fold.
5. The trainer evaluates a baseline and seven regression families with five-fold shuffled `KFold`. Grid search records R², MAE, RMSE, training-fold scores, validation-fold scores, and variability for every candidate.
6. The highest mean CV R² selects the estimator and hyperparameters. GridSearchCV refits that pipeline on all development rows.
7. The selected pipeline predicts the untouched holdout once. The run saves a bundle, JSON report, actual-vs-predicted plot, residual plot, error distribution, and global permutation-importance plot.
8. `app.py` creates the Flask app. `PredictPipeline` lazily loads and caches only the versioned bundle; `ModelBundle` checks its schema before prediction.
9. The POST form is converted into schema-ordered inputs, validated server-side, passed through the saved pipeline, and rendered with a score estimate and local sensitivity checks.

## Important files and functions

| File | Responsibility |
|---|---|
| `notebook/data/stud.csv` | Included input dataset. |
| `src/schema.py` | Feature/target constants, accepted categories, training and request validation. |
| `src/model_training.py:make_preprocessor` | Fold-local numeric and categorical transformations. |
| `src/model_training.py:make_model_specs` | Baseline, seven model families, deterministic estimators, parameter grids, optional dependency status. |
| `src/model_training.py:train_model` | Split, GridSearchCV, CV-based model selection, final holdout, report, plots, atomic bundle save. |
| `src/train.py:main` | Reproducible command-line training entry point. |
| `src/model_bundle.py:load_model_bundle` | Artifact type/version/feature-order compatibility checks. |
| `src/model_bundle.py:ModelBundle.predict_with_explanation` | Model prediction and one-feature reference-value sensitivity checks. |
| `src/pipeline/predict_pipeline.py:PredictPipeline` | Lazy cached bundle access for Flask. |
| `app.py:create_app` | Routes, validation errors, service status, dashboard and plot delivery. |
| `templates/home.html` | Prediction form, estimate, local sensitivity results. |
| `templates/evaluation.html` | CV comparison, final holdout metrics, and diagnostics. |
| `tests/test_pipeline.py` | Schema, preprocessing, training, bundle, and Flask route tests. |
| `artifacts/model_report.json` | Generated report with metadata and all candidate/fold scores. |

## Why these algorithms and transformations?

- **DummyRegressor:** establishes the score of a simple mean-prediction baseline.
- **Linear Regression:** interpretable numeric baseline that represents a linear relationship in the transformed feature space.
- **Decision Tree:** captures nonlinear splits and interactions; bounded depth and leaf-size candidates reduce unrestricted growth.
- **Random Forest:** averages randomized trees to reduce the variance of an individual tree.
- **Gradient Boosting and AdaBoost:** sequential boosting approaches that fit later learners to remaining errors or reweighted examples.
- **XGBoost and CatBoost:** optional boosted-tree implementations included to preserve the original comparison. If their imports fail, the report lists the skipped model and reason.
- **Median imputation and standard scaling:** handle numeric missingness and put numeric features on comparable scales for scale-sensitive estimators.
- **Most-frequent imputation and one-hot encoding:** represent categorical inputs numerically while allowing an encoder to handle a valid category absent from a fold.

Scaling is useful for the linear model and distance-based or regularized methods, but usually unnecessary for trees. It remains in the shared preprocessor so candidates receive a consistent input pipeline.

The grid search uses these estimator settings (the `model__` prefix targets the model step inside the pipeline):

| Estimator | Candidate values |
|---|---|
| DummyRegressor | Mean strategy; one candidate |
| LinearRegression | Defaults; one candidate |
| DecisionTreeRegressor | `max_depth`: `None`, 5, 10; `min_samples_leaf`: 1, 5 |
| RandomForestRegressor | `n_estimators`: 100, 250; `max_depth`: `None`, 8; `min_samples_leaf`: 1, 3 |
| GradientBoostingRegressor | `n_estimators`: 100, 200; `learning_rate`: 0.05, 0.1; `max_depth`: 2, 3 |
| AdaBoostRegressor | `n_estimators`: 50, 100, 200; `learning_rate`: 0.05, 0.5 |
| XGBRegressor | `n_estimators`: 100, 200; `learning_rate`: 0.05, 0.1; `max_depth`: 2, 4 |
| CatBoostRegressor | `iterations`: 100, 200; `learning_rate`: 0.03, 0.1; `depth`: 4, 6 |

The actual default run searched 1, 1, 6, 8, 8, 6, 8, and 8 candidates respectively. Linear Regression won with its single default configuration; the full per-candidate settings and fold scores are saved in `artifacts/model_report.json` (`src/model_training.py:make_model_specs`, `train_model`).

## Cross-validation, selection, and the previous leakage

The trainer uses `KFold(n_splits=5, shuffle=True, random_state=42)` on the development partition. `GridSearchCV` scoring is explicit: R², negative MAE, and negative RMSE. R² is the refit/selection metric; errors are negated for scikit-learn's maximization convention and written as positive MAE/RMSE in the report. The JSON report includes fold values and mean/std for all candidates.

The previous notebook fitted its encoder and scaler to all rows before making the train/test split, so test-set feature distributions influenced preprocessing. The previous production trainer fitted preprocessing before inner CV and used the outer test R² to select the winning algorithm, then reported that same score. That could leak fold validation information into preprocessing and made the winning test score optimistic.

The new pipeline fixes both: preprocessing is inside the model pipeline passed to CV, and CV results—not the final holdout—choose the model. The selected model's holdout metrics are measured only after selection. Repeatedly changing the system based on this holdout would invalidate it; use a new holdout or nested CV for another selection cycle.

## Measured run

These are the saved results from the documented `python -m src.train` run in the current Python 3.11 environment. Rows show each family's best candidate by mean validation-fold R²; `±` is standard deviation across five folds.

| Candidate | CV R² | CV MAE | CV RMSE |
|---|---:|---:|---:|
| Linear Regression (selected) | 0.8690 ± 0.0139 | 4.3424 ± 0.1901 | 5.4064 ± 0.2496 |
| CatBoost | 0.8587 ± 0.0116 | 4.5133 ± 0.1575 | 5.6200 ± 0.1908 |
| XGBoost | 0.8531 ± 0.0162 | 4.6028 ± 0.1866 | 5.7243 ± 0.2491 |
| Gradient Boosting | 0.8530 ± 0.0143 | 4.6001 ± 0.1727 | 5.7282 ± 0.2166 |
| Random Forest | 0.8408 ± 0.0195 | 4.8075 ± 0.2474 | 5.9546 ± 0.2731 |
| AdaBoost | 0.8218 ± 0.0123 | 5.0230 ± 0.1445 | 6.3122 ± 0.1800 |
| Decision Tree | 0.8060 ± 0.0177 | 5.2795 ± 0.2425 | 6.5800 ± 0.1815 |
| DummyRegressor | -0.0072 ± 0.0054 | 11.9631 ± 0.4352 | 15.0260 ± 0.4794 |

The selected model's development training metrics are R² 0.8743, MAE 4.2667, RMSE 5.3231. Its untouched 200-row holdout metrics are **R² 0.8804, MAE 4.2148, RMSE 5.3940**. Do not call R² accuracy. The historical notebook's 88.03 output came from another preprocessing/model run and is not the provenance for this artifact.

## Core metrics and concepts

- **R²:** `1 - SS_res / SS_tot`, squared error relative to the mean-prediction baseline. It can be negative; it is not the fraction of exact predictions.
- **MAE:** mean absolute error; easy to describe as average score-point error and less dominated by large misses.
- **MSE:** mean squared error; larger mistakes weigh more, with units in score-points squared. It is not one of the three reported comparison metrics.
- **RMSE:** square root of MSE; large errors count more than with MAE, and the result is again measured in score points.
- **Overfitting:** excellent fit to observed training rows but weaker fit on unseen rows. CV train/validation gaps and holdout metrics help identify it.
- **Preprocessing pipeline:** sequences transformations and a model, ensuring each fold learns its own imputation statistics, scales, and category vocabulary.
- **Data leakage:** accidental use of validation/test information in preprocessing, selection, or feature construction. The final holdout must not drive tuning.

## Likely technical interview questions

1. **What is the prediction target and feature set?**  
   The target is `math_score`; the seven inputs are five categorical fields and two numeric scores. `src/schema.py` is the shared contract.

2. **Where did the data come from?**  
   Training reads `notebook/data/stud.csv`; the EDA notebook attributes it to Kaggle's Students Performance in Exams dataset. The source copy has 1,000 rows and eight columns (`src/model_training.py:train_model`).

3. **Why is this regression rather than classification?**  
   The output is a numeric score, and the app displays a continuous estimate. The model code uses sklearn regressors and regression metrics (`src/model_training.py:make_model_specs`).

4. **How do you handle missing data?**  
   Numeric features use median imputation and categorical features use most-frequent imputation. The included CSV currently has no missing values; this handling supports incomplete future rows (`src/model_training.py:make_preprocessor`).

5. **Why use one-hot encoding?**  
   The categories do not have a defensible numeric ordering, so one-hot encoding represents them without assigning ranks. `handle_unknown="ignore"` allows a valid category absent in one training fold (`src/model_training.py:make_preprocessor`).

6. **Why scale the input features?**  
   Standard scaling helps linear and scale-sensitive models. Tree models usually do not need it, but one consistent preprocessing pipeline keeps comparison and serving simpler (`src/model_training.py:make_preprocessor`).

7. **What is the split strategy?**  
   `train_test_split` reserves 20% using seed 42; the remaining 80% is the development set for CV and model selection (`src/model_training.py:train_model`).

8. **How do you keep preprocessing out of CV validation folds?**  
   The `ColumnTransformer` is a step in each estimator `Pipeline` passed into `GridSearchCV`, so each fold fits its own preprocessing parameters (`src/model_training.py:make_preprocessor`, `train_model`).

9. **Which algorithms are compared?**  
   A DummyRegressor baseline plus Linear Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, XGBoost, and CatBoost. Optional libraries are reported as skipped if unavailable (`src/model_training.py:make_model_specs`).

10. **How does GridSearchCV choose hyperparameters?**  
    It evaluates each parameter candidate on five development folds, records R², MAE, and RMSE, and refits the best R² candidate on all development rows (`src/model_training.py:train_model`).

11. **Which metric controls selection?**  
    Mean CV R² is the declared selection metric. MAE and RMSE provide complementary error magnitudes and are reported for every candidate (`src/model_training.py:SCORING`, `train_model`).

12. **What does the reported R² of 0.8804 mean?**  
    On this 200-row holdout, squared error was 88.04% lower than always predicting that holdout's mean. It does not mean 88% of estimates are correct, and it is not classification accuracy (`regression_metrics`, `src/model_training.py`).

13. **What are MAE and RMSE in this project?**  
    MAE is average absolute error; RMSE is the square root of mean squared error and weights large misses more. Both are in math-score points (`regression_metrics`, `src/model_training.py`).

14. **Why include DummyRegressor?**  
    It provides a baseline to show whether feature-based models beat a constant mean predictor. Its CV R² was -0.0072 in the measured run (`make_model_specs`, `model_report.json`).

15. **How did you select the final estimator?**  
    The candidate with the highest mean CV R² was selected before inspecting holdout performance. Linear Regression won this run (`train_model`, `model_report.json`).

16. **What are the final holdout results?**  
    Linear Regression achieved R² 0.8804, MAE 4.2148, and RMSE 5.3940 on the untouched 200-row holdout. The report also stores development training metrics (`train_model`, `artifacts/model_report.json`).

17. **What was wrong with the previous evaluation?**  
    The notebook fit preprocessing before splitting; the previous trainer used the final test scores to choose the winner and reported that same winner's test score. The new pipeline moves preprocessing inside CV and reserves the holdout (`src/model_training.py`, historical notebook cells 13–23, legacy `src/utils.py:evaluate_models`).

18. **What is overfitting, and did you see evidence of it?**  
    Overfitting is strong training fit with weaker validation performance. The historical notebook's Decision Tree had train R² 0.9997 and test R² 0.7603; the new report records CV train/validation gaps (`artifacts/model_report.json`).

19. **How is the model saved and loaded?**  
    The fitted preprocessing/model pipeline and metadata are stored together in `artifacts/model_bundle.pkl`; `load_model_bundle` checks the schema and fitted feature names before use (`src/model_training.py:train_model`, `src/model_bundle.py:load_model_bundle`).

20. **How does a form request become a prediction?**  
    `app.py:create_app` maps the POST fields into a validated dataframe; `PredictPipeline` gets the cached bundle; `ModelBundle.predict` calls the saved pipeline; the template renders the estimate (`app.py`, `src/pipeline/predict_pipeline.py`, `src/model_bundle.py`).

21. **How are malformed or unknown request values handled?**  
    `validate_form_values` rejects missing fields, non-finite or out-of-range scores, and categories outside the domain. The route returns a human-readable 400 instead of exposing a traceback (`src/schema.py`, `app.py:create_app`).

22. **What does local sensitivity explain?**  
    It shows how the model estimate changes when one input is replaced with a training median or mode. It is not an additive contribution, a causal effect, or a global importance measure (`ModelBundle.predict_with_explanation`).

23. **What does global permutation importance show?**  
    It measures the decrease in training R² when a feature is shuffled, averaged over repeats. It is descriptive, training-set based, and not causal (`src/model_training.py:train_model`, `permutation_importance`).

24. **What is known about deployment?**  
    The Flask app has an exported WSGI object and Gunicorn is a dependency, but no Render configuration or hosted deployment has been verified (`app.py`, `requirements.txt`).

25. **What is the main limitation before real use?**  
    The dataset is small, demographic inputs may encode structural disparities, and no external-cohort validation or decision-policy review has been done. The interface labels the result as a demonstration estimate (`README.md`, `templates/home.html`).

## Five debugging scenarios

1. **POST returns an error when a score is blank.** Check `validate_form_values`; it should reject the request with a 400 and display a field message, not raise a raw exception (`src/schema.py`, `app.py:create_app`).
2. **A category unseen in training arrives.** Valid domain categories are accepted and the fold-fitted encoder uses `handle_unknown="ignore"`; values outside the declared domain are rejected with a clear message (`src/schema.py`, `src/model_training.py:make_preprocessor`).
3. **A restored model reports the wrong feature count/order.** `load_model_bundle` checks schema version, target, and fitted feature names. Re-train a bundle compatible with this code instead of loading the legacy two-file artifacts (`src/model_bundle.py:load_model_bundle`).
4. **A CV score is strong but the holdout drops.** Inspect fold standard deviations and train/validation gaps, verify the holdout was not used repeatedly, and collect a larger independent cohort before revising model choice (`src/model_training.py:train_model`).
5. **The app shows “model unavailable.”** Run `python -m src.train`, confirm `artifacts/model_bundle.pkl` exists and is compatible, then restart the app. The app intentionally does not fall back to an unverified old artifact (`src/pipeline/predict_pipeline.py:PredictPipeline`).

## Five design trade-offs

1. **R² vs RMSE for model selection:** This run selects by mean CV R² and reports MAE/RMSE alongside it. A cost-sensitive application might prefer a predeclared error metric instead.
2. **Small dataset vs larger search:** A larger grid may find better candidates but increases variance and search cost; nested CV or more data is preferable to repeatedly tuning against the holdout.
3. **One-hot encoding vs ordinal encoding:** One-hot encoding avoids inventing an order among education or group labels, at the cost of more columns and sparse features.
4. **Local sensitivity vs SHAP:** One-feature reference perturbations are light and model-agnostic, but not additive and can miss interactions. SHAP could be richer, but adds dependency, runtime, and explanation assumptions.
5. **Sensitive features vs predictive performance:** Removing demographics may change fit, but neither inclusion nor exclusion by itself establishes fairness. Evaluate subgroup errors and intended use with stakeholders before any real deployment.

## Facts to verify before making claims

- The Kaggle attribution and its terms/license are documented by the notebook; verify the current source and usage rights before redistribution.
- The published results correspond to the exact CSV whose SHA-256 is recorded in `artifacts/model_report.json`.
- State the split seed (42), five-fold CV settings, scoring metrics, selected model, and holdout size accurately.
- Quote the actual result as R² 0.8804, MAE 4.2148, RMSE 5.3940 for the measured run; never call it 88% accuracy.
- Confirm the optional XGBoost and CatBoost packages were present for the run being described; check `skipped_models` and dependency versions in the report.
- Distinguish this corrected run from the historical notebook's 88.03 output and from the legacy 14-feature pickle.
- Verify the model bundle and generated plots/report correspond to the run you demonstrate; generated artifacts are locally ignored by Git.
- Do not claim Render hosting is live or tested; no provider configuration is present in the repository.
- Do not claim the model is fair, causal, or suitable for student-level educational decisions.
- Verify the deployment context can legitimately know reading and writing scores at prediction time.
