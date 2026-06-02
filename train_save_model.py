from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_engineering import RAW_FEATURES, SatelliteFeatureEngineer


DATA_PATH = Path("satellite_failure_dataset.csv")
TARGET = "component_failure"
RANDOM_STATE = 42


def build_pipeline(model):
    return Pipeline(
        [
            ("features", SatelliteFeatureEngineer()),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def candidate_models():
    return {
        "Logistic Regression": build_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "Random Forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=250,
                max_depth=6,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "Gradient Boosting": build_pipeline(
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                random_state=RANDOM_STATE,
            )
        ),
    }


def evaluate_model(name, pipeline, X_train, y_train, X_test, y_test, cv):
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)

    return {
        "model": name,
        "cv_accuracy": cv_scores["test_accuracy"].mean(),
        "cv_precision": cv_scores["test_precision"].mean(),
        "cv_recall": cv_scores["test_recall"].mean(),
        "cv_f1": cv_scores["test_f1"].mean(),
        "cv_roc_auc": cv_scores["test_roc_auc"].mean(),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": cm.tolist(),
        "pipeline": pipeline,
    }


def transformed_training_data(pipeline, X):
    engineered = pipeline.named_steps["features"].transform(X)
    scaled = pipeline.named_steps["scaler"].transform(engineered)
    feature_names = pipeline.named_steps["features"].get_feature_names_out().tolist()
    return scaled, feature_names


def compute_shap_summary(best_pipeline, X_train, X_test):
    background_raw = shap.sample(X_train, min(200, len(X_train)), random_state=RANDOM_STATE)
    explain_raw = shap.sample(X_test, min(100, len(X_test)), random_state=RANDOM_STATE)

    background_scaled, feature_names = transformed_training_data(best_pipeline, background_raw)
    explain_scaled, _ = transformed_training_data(best_pipeline, explain_raw)
    model = best_pipeline.named_steps["model"]

    explainer = shap.Explainer(model, background_scaled, feature_names=feature_names)
    shap_values = explainer(explain_scaled)
    values = shap_values.values
    if values.ndim == 3:
        values = values[:, :, 1]

    summary = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": np.abs(values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    summary.to_csv("shap_summary.csv", index=False)
    return summary


def write_model_report(data, metrics_df, best_result, shap_summary):
    target_counts = data[TARGET].value_counts().sort_index().to_dict()
    best_model = best_result["model"]
    cm = best_result["confusion_matrix"]

    metrics_table = markdown_table(
        metrics_df,
        ["model", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc", "cv_f1", "cv_roc_auc"],
    )
    shap_table = markdown_table(shap_summary.head(10), ["feature", "mean_abs_shap"], decimals=4)

    report = f"""# Relatorio do Modelo

## Dataset

O projeto usa um dataset sintetico e reproduzivel de telemetria de satelite gerado por `generate_dataset.py`.
O dataset tem {data.shape[0]} linhas e {data.shape[1]} colunas: {len(RAW_FEATURES)} variaveis de entrada mais o alvo binario `{TARGET}`.

- Distribuicao do alvo: {target_counts}
- Valores ausentes: {int(data.isna().sum().sum())}
- Tipo do problema: classificacao binaria, em que `1` indica falha do componente.

## Pipeline

O pipeline de treinamento usa `SatelliteFeatureEngineer`, `StandardScaler` e um classificador supervisionado.
As variaveis criadas por engenharia de atributos sao:

- `thermal_stress`: distancia absoluta da temperatura nominal de 25 C.
- `low_voltage_risk`: quanto a bateria esta abaixo de 12 V.
- `fuel_reserve_risk`: quanto o combustivel esta abaixo de 500 kg.
- `radiation_age_interaction`: proxy de exposicao acumulada usando radiacao vezes tempo operacional.
- `orientation_magnitude`: magnitude do vetor de orientacao.

Os dados sao divididos com estratificacao em 80% treino e 20% teste. O conjunto de treino tambem e avaliado com validacao cruzada estratificada de 5 folds.

## Modelos Comparados

{metrics_table}

## Melhor Modelo

O modelo selecionado e **{best_model}**, escolhido pelo maior F1-score no teste. F1 e a metrica principal porque a classe positiva representa falha do componente e e menos frequente que a classe normal.

Matriz de confusao do modelo selecionado (`[[TN, FP], [FN, TP]]`):

```text
{cm}
```

## Interpretabilidade com SHAP

O SHAP foi calculado para o modelo selecionado depois das mesmas etapas de engenharia de atributos e normalizacao usadas no treinamento. Os maiores valores medios absolutos indicam as variaveis que mais influenciaram as previsoes de forma global.

{shap_table}

## Limitacoes

O dataset e sintetico e deve ser tratado como simulacao, nao como telemetria certificada de missao real. Um uso real exigiria dados observados de satelites, validacao operacional, monitoramento de drift e revisao por especialistas de missao.

## Reprodutibilidade

```powershell
pip install -r requirements.txt
python generate_dataset.py
python train_save_model.py
python -m streamlit run app.py
```
"""
    Path("model_report.md").write_text(report, encoding="utf-8")


def markdown_table(df, columns, decimals=3):
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] + ["---:" for _ in columns[1:]]) + " |"
    rows = []
    for _, row in df[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.{decimals}f}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator] + rows)


def main():
    data = pd.read_csv(DATA_PATH)
    X = data[RAW_FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = []
    for name, pipeline in candidate_models().items():
        print(f"Training and evaluating {name}...")
        results.append(evaluate_model(name, pipeline, X_train, y_train, X_test, y_test, cv))

    best_result = max(results, key=lambda result: result["test_f1"])
    best_pipeline = best_result["pipeline"]

    metrics_df = pd.DataFrame(
        [
            {key: value for key, value in result.items() if key != "pipeline"}
            for result in results
        ]
    ).sort_values("test_f1", ascending=False)
    metrics_df.to_csv("model_metrics.csv", index=False)

    joblib.dump(best_pipeline, "best_model.pkl")
    joblib.dump(best_pipeline, "logistic_model.pkl")

    shap_summary = compute_shap_summary(best_pipeline, X_train, X_test)
    write_model_report(data, metrics_df, best_result, shap_summary)

    metadata = {
        "best_model": best_result["model"],
        "selection_metric": "test_f1",
        "raw_features": RAW_FEATURES,
        "target": TARGET,
    }
    Path("model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Best model: {best_result['model']}")
    print(metrics_df[["model", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]])
    print("Saved best_model.pkl, model_metrics.csv, shap_summary.csv, model_report.md, and model_metadata.json")


if __name__ == "__main__":
    main()

