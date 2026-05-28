"""
Streamlit application for the GAIE space‑economy project.

This app demonstrates a simple machine‑learning pipeline built from a synthetic
dataset that mimics telemetry readings from a satellite. The purpose of the
pipeline is to predict whether a satellite component will fail (binary
classification) based on sensor readings and mission conditions. The app
allows users to input values for each feature, produces a prediction using the
trained logistic regression model, and displays the SHAP values for the
prediction to help interpret the model's decision.

To run the app locally, install the dependencies:

```
pip install streamlit scikit-learn pandas numpy shap joblib
```

Then start the application with:

```
streamlit run app.py
```

The model (`logistic_model.pkl`) and the dataset (`satellite_failure_dataset.csv`)
must be located in the same directory as this script.
"""

import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np

# Load the trained logistic regression pipeline
@st.cache_resource
def load_model():
    return joblib.load("logistic_model.pkl")

# Load dataset for showing examples and to infer feature ranges
@st.cache_data
def load_data():
    return pd.read_csv("satellite_failure_dataset.csv")

def main():
    st.title("Satellite Component Failure Prediction")
    st.write(
        "This app uses a logistic regression model trained on a synthetic dataset to "
        "predict whether a satellite component is likely to fail. You can adjust the "
        "sensor readings and mission parameters below and view the prediction along "
        "with an explanation of the model's decision via SHAP values."
    )

    model = load_model()
    data = load_data()

    # Define the input form
    st.subheader("Input Features")
    with st.form("input_form"):
        temp = st.number_input(
            "Temperature (°C)",
            min_value=float(data["temperature_c"].min()),
            max_value=float(data["temperature_c"].max()),
            value=float(data["temperature_c"].mean()),
            step=0.1,
        )
        voltage = st.number_input(
            "Battery Voltage (V)",
            min_value=float(data["battery_voltage"].min()),
            max_value=float(data["battery_voltage"].max()),
            value=float(data["battery_voltage"].mean()),
            step=0.1,
        )
        orientation_x = st.number_input(
            "Orientation X", min_value=-1.0, max_value=1.0, value=0.0, step=0.01
        )
        orientation_y = st.number_input(
            "Orientation Y", min_value=-1.0, max_value=1.0, value=0.0, step=0.01
        )
        orientation_z = st.number_input(
            "Orientation Z", min_value=-1.0, max_value=1.0, value=0.0, step=0.01
        )
        radiation = st.number_input(
            "Cosmic Radiation (counts)",
            min_value=float(data["cosmic_radiation"].min()),
            max_value=float(data["cosmic_radiation"].max()),
            value=float(data["cosmic_radiation"].mean()),
            step=0.1,
        )
        solar_flux = st.number_input(
            "Solar Flux (W/m²)",
            min_value=float(data["solar_flux"].min()),
            max_value=float(data["solar_flux"].max()),
            value=float(data["solar_flux"].mean()),
            step=0.1,
        )
        data_rate = st.number_input(
            "Data Rate (Mbps)",
            min_value=float(data["data_rate_mbps"].min()),
            max_value=float(data["data_rate_mbps"].max()),
            value=float(data["data_rate_mbps"].mean()),
            step=0.1,
        )
        fuel = st.number_input(
            "Thruster Fuel (kg)",
            min_value=float(data["thruster_fuel_kg"].min()),
            max_value=float(data["thruster_fuel_kg"].max()),
            value=float(data["thruster_fuel_kg"].mean()),
            step=0.1,
        )
        age = st.number_input(
            "Age (years)",
            min_value=float(data["age_years"].min()),
            max_value=float(data["age_years"].max()),
            value=float(data["age_years"].mean()),
            step=0.1,
        )

        submitted = st.form_submit_button("Predict")

    if submitted:
        # Assemble input into DataFrame
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
        )

        # Predict probability and class
        prob = model.predict_proba(sample)[0, 1]
        prediction = model.predict(sample)[0]

        st.subheader("Prediction")
        st.write(
            f"**Failure Probability:** {prob:.2f}\n\n**Predicted Class:** {'Failure' if prediction == 1 else 'No Failure'}"
        )

        # SHAP explanation
        st.subheader("Model Explanation (SHAP)")
        # Create a SHAP explainer on the fly
        explainer = shap.Explainer(
            model.named_steps["logreg"],
            model.named_steps["scaler"].transform(data.drop("component_failure", axis=1)),
        )
        shap_values = explainer(model.named_steps["scaler"].transform(sample))

        # Display SHAP values as a bar chart
        shap_df = pd.DataFrame(
            {
                "feature": sample.columns,
                "shap_value": shap_values.values[0],
                "abs_shap": np.abs(shap_values.values[0]),
            }
        ).sort_values(by="abs_shap", ascending=False)

        st.write(
            "Features with larger absolute SHAP values contribute more to the model's decision. "
            "Positive values push the prediction towards failure and negative values push it towards no failure."
        )
        st.bar_chart(shap_df.set_index("feature")["shap_value"])

    # Show a sample of the dataset for reference
    st.subheader("Sample of Training Data")
    st.write(data.head())


if __name__ == "__main__":
    main()
