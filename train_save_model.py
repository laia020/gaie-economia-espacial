"""
Script to train the logistic regression model for GAIE project.

This script loads the synthetic dataset, performs a train/test split,
trains a logistic regression pipeline with scaling and saves the model
using joblib. Running this script is optional because the model is already
saved in logistic_model.pkl.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
data = pd.read_csv("satellite_failure_dataset.csv")

# Separate features and target
X = data.drop("component_failure", axis=1)
y = data["component_failure"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define pipeline
pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ]
)

# Fit model
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "logistic_model.pkl")
print("Model trained and saved to logistic_model.pkl")
