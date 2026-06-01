# Project Deliverables Reference

Use this reference when creating or updating documentation for the Space Economy AI/ML project.

## README Required Sections

A complete README should include:

1. Problem context in Space Economy.
2. Data source or synthetic data generation method.
3. Dataset shape, columns, target variable, and known limitations.
4. Methodology: preprocessing, feature engineering, model training, validation, comparison, and selection.
5. Models tested and metrics table.
6. SHAP interpretation with most influential variables.
7. How to run locally.
8. How to retrain the model.
9. Deployment link or explicit note that deployment is local only.
10. Repository organization.

Do not leave placeholders such as `<URL_DO_REPOSITORIO>` or `<APP_LINK>` in final documentation.

## model_report.md Structure

Use this structure for reports:

```markdown
# Model Report

## Dataset
- Source/generation method
- Shape
- Feature list
- Target distribution
- Missing values

## Pipeline
- Preprocessing
- Feature engineering/selection
- Train/test split or cross-validation

## Models Compared
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|

## Best Model
Explain which metric was used and why.

## SHAP Interpretation
List global feature influence and provide one local prediction explanation.

## Limitations
Mention synthetic data, class imbalance, deployment limitations, and need for real mission data.

## Reproducibility
Commands to install, train, test, and run the app.
```

## Rubric Evidence Checklist

When evaluating a project, build a checklist with:

- Requirement
- Evidence file/path
- Status: OK, Partial, Missing
- Recommended fix

Prefer concrete file references over general statements.
