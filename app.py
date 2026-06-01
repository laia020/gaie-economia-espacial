from __future__ import annotations

import json
from pathlib import Path

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
    return {"best_model": "Trained model", "selection_metric": "test_f1"}


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
    st.set_page_config(page_title="Satellite Failure Prediction", layout="wide")
    st.title("Satellite Component Failure Prediction")

    model = load_model()
    data = load_data()
    metadata = load_metadata()
    metrics = load_metrics()

    st.write(
        "This app predicts satellite component failure from telemetry readings using "
        f"the selected model: **{metadata.get('best_model', 'trained model')}**."
    )

    if not metrics.empty:
        with st.expander("Model comparison metrics", expanded=False):
            metric_columns = [
                "model",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_roc_auc",
            ]
            st.dataframe(metrics[metric_columns], use_container_width=True)

    st.subheader("Input Features")
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            temp = number_input_from_data("Temperature (C)", data, "temperature_c")
            voltage = number_input_from_data("Battery Voltage (V)", data, "battery_voltage")
            orientation_x = st.number_input("Orientation X", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
            orientation_y = st.number_input("Orientation Y", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
            orientation_z = st.number_input("Orientation Z", min_value=-1.0, max_value=1.0, value=0.0, step=0.01)
        with col2:
            radiation = number_input_from_data("Cosmic Radiation (counts)", data, "cosmic_radiation")
            solar_flux = number_input_from_data("Solar Flux (W/m2)", data, "solar_flux")
            data_rate = number_input_from_data("Data Rate (Mbps)", data, "data_rate_mbps")
            fuel = number_input_from_data("Thruster Fuel (kg)", data, "thruster_fuel_kg")
            age = number_input_from_data("Age (years)", data, "age_years")

        submitted = st.form_submit_button("Predict")

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

        st.subheader("Prediction")
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("Failure probability", f"{prob:.1%}")
        metric_col2.metric("Predicted class", "Failure" if prediction == 1 else "No failure")

        st.subheader("Model Explanation (SHAP)")
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

        st.write(
            "Positive SHAP values push the prediction toward failure; negative values push it toward no failure."
        )
        st.bar_chart(shap_df.set_index("feature")["shap_value"])
        st.dataframe(shap_df.head(8)[["feature", "shap_value"]], use_container_width=True)

    st.subheader("Sample of Training Data")
    st.dataframe(data.head(), use_container_width=True)


if __name__ == "__main__":
    main()
