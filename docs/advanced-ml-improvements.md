# Advanced ML Improvement Methods

Use this reference when improving a Space Economy AI/ML project beyond a basic prototype.

## Model Comparison

Prefer a reusable training loop that evaluates multiple candidates with the same split and metrics. For classification, include at least two substantially different techniques:

- Linear baseline: `LogisticRegression` with scaling.
- Tree ensemble: `RandomForestClassifier`, `GradientBoostingClassifier`, or `HistGradientBoostingClassifier`.
- Optional margin model: `SVC(probability=True)` when dataset size is small enough.

Use `StratifiedKFold` for classification and preserve class ratios.

## Hyperparameter Search

Use `GridSearchCV` for small search spaces and `RandomizedSearchCV` for broader spaces. Score with the metric that matches the project objective:

- Failure detection: prioritize recall, F1, ROC-AUC, or PR-AUC.
- Balanced general classification: use F1 or ROC-AUC.
- Regression: use MAE, RMSE, and R2.
- Clustering: use silhouette score and domain interpretation.

Keep search spaces modest for student projects so the workflow remains runnable on a laptop.

## Imbalanced Classification

For satellite failure prediction, missing true failures is often worse than a false alert. Consider:

- `class_weight="balanced"` for linear models and random forests.
- Threshold tuning on predicted probabilities.
- Precision-recall curve analysis.
- Confusion matrix narrative focused on false negatives.

Only add external resampling packages when already allowed by the project dependencies or when the user agrees to expand requirements.

## Feature Engineering

Make feature engineering domain-relevant and explainable:

- Thermal stress: absolute deviation from nominal temperature.
- Low-voltage risk: nominal voltage minus observed voltage, clipped at zero.
- Radiation pressure: radiation multiplied by age or solar flux buckets.
- Fuel reserve risk: inverse or low-fuel indicator.
- Attitude instability: norm or dispersion of orientation components.

Add these inside a reproducible preprocessing function or transformer. Document each engineered feature in `model_report.md`.

## Feature Selection

Use one or more simple, defensible methods:

- Mutual information scores.
- Permutation importance on the validation/test set.
- Model coefficients for linear models.
- Tree feature importance as a secondary signal.

Avoid using test data to decide selected features unless explicitly reporting it as post-hoc analysis.

## Calibration and Thresholds

When the app shows probabilities, verify calibration when feasible:

- Compare raw probabilities with `CalibratedClassifierCV`.
- Report Brier score if probability quality matters.
- Pick a decision threshold based on the assignment goal, for example maximizing F1 or targeting high recall.

## Error Analysis

After selecting a model, inspect false positives and false negatives:

- Summarize feature ranges for each error type.
- Identify whether failures cluster around high radiation, low voltage, old age, or fuel conditions.
- Add a short limitations section explaining synthetic-data bias and real-world validation needs.

## SHAP

For a selected model:

- Provide local SHAP explanations for one prediction in the app.
- Provide global mean absolute SHAP values in `model_report.md`.
- Align transformed features back to human-readable names.
- For linear models with scaled inputs, explain that SHAP is computed on transformed numeric values unless the code maps values back.

## Reproducibility

Persist outputs deterministically:

- `best_model.pkl`
- `model_report.md`
- optional `metrics.json` or `model_metrics.csv`
- optional `shap_summary.csv`

Use fixed `random_state` values and record package requirements.
