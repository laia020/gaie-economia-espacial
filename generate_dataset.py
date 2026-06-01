from __future__ import annotations

import numpy as np
import pandas as pd


OUTPUT_PATH = "satellite_failure_dataset.csv"
RANDOM_SEED = 42
N_ROWS = 1000


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def generate_dataset(n_rows=N_ROWS, random_seed=RANDOM_SEED):
    rng = np.random.default_rng(random_seed)

    temperature_c = rng.normal(25.0, 5.0, n_rows)
    battery_voltage = rng.normal(12.0, 0.5, n_rows)
    orientation_x = rng.uniform(-1.0, 1.0, n_rows)
    orientation_y = rng.uniform(-1.0, 1.0, n_rows)
    orientation_z = rng.uniform(-1.0, 1.0, n_rows)
    cosmic_radiation = rng.normal(50.0, 10.0, n_rows)
    solar_flux = rng.normal(1361.0, 20.0, n_rows)
    data_rate_mbps = rng.normal(100.0, 20.0, n_rows)
    thruster_fuel_kg = rng.normal(500.0, 55.0, n_rows)
    age_years = rng.uniform(0.0, 10.0, n_rows)

    orientation_magnitude = np.sqrt(orientation_x**2 + orientation_y**2 + orientation_z**2)
    thermal_stress = np.abs(temperature_c - 25.0)
    low_voltage_risk = np.maximum(0.0, 12.0 - battery_voltage)
    fuel_reserve_risk = np.maximum(0.0, 500.0 - thruster_fuel_kg)

    logit = (
        -2.15
        + 0.075 * thermal_stress
        + 1.15 * low_voltage_risk
        + 0.035 * (cosmic_radiation - 50.0)
        + 0.11 * age_years
        + 0.006 * fuel_reserve_risk
        + 0.18 * orientation_magnitude
        + 0.004 * (data_rate_mbps - 100.0)
        + 0.003 * (solar_flux - 1361.0)
    )
    failure_probability = sigmoid(logit)
    component_failure = rng.binomial(1, failure_probability)

    return pd.DataFrame(
        {
            "temperature_c": temperature_c,
            "battery_voltage": battery_voltage,
            "orientation_x": orientation_x,
            "orientation_y": orientation_y,
            "orientation_z": orientation_z,
            "cosmic_radiation": cosmic_radiation,
            "solar_flux": solar_flux,
            "data_rate_mbps": data_rate_mbps,
            "thruster_fuel_kg": thruster_fuel_kg,
            "age_years": age_years,
            "component_failure": component_failure,
        }
    )


def main():
    data = generate_dataset()
    data.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset saved to {OUTPUT_PATH}")
    print(f"Shape: {data.shape[0]} rows x {data.shape[1]} columns")
    print(f"Target distribution: {data['component_failure'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
