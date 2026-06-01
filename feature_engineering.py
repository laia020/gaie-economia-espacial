from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


RAW_FEATURES = [
    "temperature_c",
    "battery_voltage",
    "orientation_x",
    "orientation_y",
    "orientation_z",
    "cosmic_radiation",
    "solar_flux",
    "data_rate_mbps",
    "thruster_fuel_kg",
    "age_years",
]


ENGINEERED_FEATURES = [
    "thermal_stress",
    "low_voltage_risk",
    "fuel_reserve_risk",
    "radiation_age_interaction",
    "orientation_magnitude",
]


class SatelliteFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add domain-inspired telemetry risk features for satellite failure models."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()

        missing = [column for column in RAW_FEATURES if column not in df.columns]
        if missing:
            raise ValueError(f"Missing required input features: {missing}")

        df["thermal_stress"] = np.abs(df["temperature_c"] - 25.0)
        df["low_voltage_risk"] = np.maximum(0.0, 12.0 - df["battery_voltage"])
        df["fuel_reserve_risk"] = np.maximum(0.0, 500.0 - df["thruster_fuel_kg"])
        df["radiation_age_interaction"] = df["cosmic_radiation"] * df["age_years"]
        df["orientation_magnitude"] = np.sqrt(
            df["orientation_x"] ** 2 + df["orientation_y"] ** 2 + df["orientation_z"] ** 2
        )

        return df[RAW_FEATURES + ENGINEERED_FEATURES]

    def get_feature_names_out(self, input_features=None):
        return np.array(RAW_FEATURES + ENGINEERED_FEATURES, dtype=object)
