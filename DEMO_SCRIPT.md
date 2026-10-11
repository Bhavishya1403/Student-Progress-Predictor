# Interview Demo Script (3–5 minutes)

## Before the interview

From the repository root, activate the project environment and verify the generated model files are present:

```bash
source .venv/bin/activate
test -f artifacts/model_bundle.pkl && test -f artifacts/model_report.json
python app.py
```

Open `http://127.0.0.1:5000` and keep the README and `artifacts/model_report.json` available in another tab. If the bundle is absent, run `python -m src.train` before starting the app; do not train during the timed demonstration.

## Walkthrough

### 0:00–0:40 — Problem and scope

Open the landing page. Say: “This estimates a math score from seven student features. It is a small-dataset applied ML demonstration, not an educational decision system. Reading and writing scores must already be available at prediction time.” Point to the model-ready status and the evaluation link.

### 0:40–1:25 — Reproducible experiment

Open `README.md` and `src/model_training.py`. Show the data split, `make_preprocessor`, `GridSearchCV`, and the final holdout evaluation. Explain that preprocessing lives inside each CV pipeline and the holdout never selects the winner. Mention the seven regression families plus DummyRegressor.

### 1:25–2:10 — Results and diagnostics

Open `/evaluation`. Show the selected model, five-fold CV metrics with standard deviations, and the separate untouched holdout metrics. Say: “R² is not accuracy; on this measured run, holdout R² is 0.8804, MAE is 4.2148 points, and RMSE is 5.3940 points.” Show the CV model comparison and one holdout diagnostic plot. Point out that permutation importance is global and training-data based.

### 2:10–3:10 — Live prediction

Open the prediction form, choose “Load example values,” then submit. Show the estimate and read the one-feature sensitivity table. Explain: “Each row changes one feature to a training median or mode. These checks are not additive explanations or causal effects, and they can miss interactions.”

### 3:10–4:00 — Validation and artifact safety

Briefly show `src/schema.py` and `src/model_bundle.py`. Explain that server-side validation rejects unsupported categories and invalid scores, and the app refuses the old unversioned two-file artifacts. If useful, run `python -m unittest discover -s tests -v` to show the input and missing-artifact checks.

### 4:00–4:40 — Evaluation caveat and close

Summarize the historical issue: the notebook fitted preprocessing before splitting and the former production code selected models on the final test set. The revised code nests preprocessing within CV and uses the holdout only after model selection. Close with the main limits: one 1,000-row source, demographic associations are not causal, and no Render deployment is configured or claimed.

## Keep these claims precise

- Quote “R² 0.8804,” not “88% accuracy.”
- Separate the 0.8690 ± 0.0139 five-fold CV R² from the 0.8804 untouched holdout R².
- State that the measured result is tied to the included CSV hash and current dependency versions in the report.
- Do not say the model is fair, causal, or ready to make decisions about students.
- Do not say the service is deployed to Render; only local Flask serving was exercised.
