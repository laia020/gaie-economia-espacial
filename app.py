from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from feature_engineering import RAW_FEATURES


DATA_PATH = Path("satellite_failure_dataset.csv")
MODEL_PATH = Path("best_model.pkl")
METADATA_PATH = Path("model_metadata.json")
METRICS_PATH = Path("model_metrics.csv")
TARGET = "component_failure"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metadata():
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {"best_model": "Modelo treinado", "selection_metric": "test_f1"}


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return pd.read_csv(METRICS_PATH)
    return pd.DataFrame()


@st.cache_resource
def build_explainer(_model, background):
    engineered = _model.named_steps["features"].transform(background)
    scaled = _model.named_steps["scaler"].transform(engineered)
    feature_names = _model.named_steps["features"].get_feature_names_out().tolist()
    estimator = _model.named_steps["model"]
    explainer = shap.Explainer(estimator, scaled, feature_names=feature_names)
    return explainer, feature_names


def transformed_sample(model, sample):
    engineered = model.named_steps["features"].transform(sample)
    return model.named_steps["scaler"].transform(engineered)


def shap_values_for_prediction(explainer, transformed):
    shap_values = explainer(transformed)
    values = shap_values.values
    if values.ndim == 3:
        values = values[:, :, 1]
    return values[0]


def number_input_from_data(label, data, column, step=0.1):
    return st.number_input(
        label,
        min_value=float(data[column].min()),
        max_value=float(data[column].max()),
        value=float(data[column].mean()),
        step=step,
    )


def main():
    st.set_page_config(page_title="Previsao de Falha em Satelites", layout="wide")
    st.title("Previsao de Falha em Componentes de Satelites")

    model = load_model()
    data = load_data()
    metadata = load_metadata()
    metrics = load_metrics()

    st.write(
        "Este app estima a probabilidade de falha de um componente de satelite "
        f"a partir de leituras de telemetria. Modelo selecionado: **{metadata.get('best_model', 'modelo treinado')}**."
    )

    if not metrics.empty:
        with st.expander("Metricas de comparacao dos modelos", expanded=False):
            metric_columns = [
                "model",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_roc_auc",
            ]
            metrics_display = metrics[metric_columns].rename(
                columns={
                    "model": "modelo",
                    "test_accuracy": "accuracy",
                    "test_precision": "precision",
                    "test_recall": "recall",
                    "test_f1": "f1_score",
                    "test_roc_auc": "roc_auc",
                }
            )
            st.dataframe(metrics_display, use_container_width=True, hide_index=True)

    st.subheader("Variaveis de entrada")
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            temp = number_input_from_data("Temperatura (C)", data, "temperature_c")
            voltage = number_input_from_data("Tensao da bateria (V)", data, "battery_voltage")
            orientation_x = st.number_input("Orientacao X", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
            orientation_y = st.number_input("Orientacao Y", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
            orientation_z = st.number_input("Orientacao Z", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
        with col2:
            radiation = number_input_from_data("Radiacao cosmica (counts)", data, "cosmic_radiation")
            solar_flux = number_input_from_data("Fluxo solar (W/m2)", data, "solar_flux")
            data_rate = number_input_from_data("Taxa de dados (Mbps)", data, "data_rate_mbps")
            fuel = number_input_from_data("Combustivel do propulsor (kg)", data, "thruster_fuel_kg")
            age = number_input_from_data("Idade (anos)", data, "age_years")

        submitted = st.form_submit_button("Prever")

    sample = pd.DataFrame(
        {
            "temperature_c": [temp],
            "battery_voltage": [voltage],
            "orientation_x": [orientation_x],
            "orientation_y": [orientation_y],
            "orientation_z": [orientation_z],
            "cosmic_radiation": [radiation],
            "solar_flux": [solar_flux],
            "data_rate_mbps": [data_rate],
            "thruster_fuel_kg": [fuel],
            "age_years": [age],
        }
    )[RAW_FEATURES]

    if submitted:
        prob = model.predict_proba(sample)[0, 1]
        prediction = model.predict(sample)[0]

        st.subheader("Previsao")
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Probabilidade de falha", f"{prob:.1%}")
        metric_col2.metric("Classe prevista", "Falha" if prediction == 1 else "Sem falha")

        st.subheader("Explicacao do modelo (SHAP)")
        background = data[RAW_FEATURES].sample(min(200, len(data)), random_state=42)
        explainer, feature_names = build_explainer(model, background)
        local_values = shap_values_for_prediction(explainer, transformed_sample(model, sample))
        shap_df = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "shap_value": local_values,
                    "abs_shap": np.abs(local_values),
                }
            )
            .sort_values("abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        shap_df["impacto"] = np.where(
            shap_df["shap_value"] >= 0,
            "Aumenta risco de falha",
            "Reduz risco de falha",
        )

        st.write(
            "Valores SHAP positivos aumentam a previsao de falha; valores negativos reduzem essa previsao."
        )
        top_shap = shap_df.head(8).copy()
        chart = (
            alt.Chart(top_shap.sort_values("abs_shap"))
            .mark_bar()
            .encode(
                x=alt.X("shap_value:Q", title="Valor SHAP"),
                y=alt.Y("feature:N", title="Feature", sort=None),
                color=alt.condition(
                    alt.datum.shap_value >= 0,
                    alt.value("#d95f02"),
                    alt.value("#1f77b4"),
                ),
                tooltip=[
                    alt.Tooltip("feature:N", title="Feature"),
                    alt.Tooltip("shap_value:Q", title="Valor SHAP", format=".4f"),
                    alt.Tooltip("impacto:N", title="Impacto"),
                ],
            )
        )
        st.altair_chart(chart, use_container_width=True)

        table = top_shap[["feature", "shap_value", "impacto"]].copy()
        table["shap_value"] = table["shap_value"].round(4)
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Amostra dos dados de treinamento")
    st.dataframe(data.head(), use_container_width=True)


if __name__ == "__main__":
    main()
