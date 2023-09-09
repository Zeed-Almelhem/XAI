## Test 11

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import shap
import matplotlib.pyplot as plt
from XAI.xai.regression.visualizations import create_residual_plot_with_shapley

# Generate a synthetic regression dataset
X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Define feature names (replace with your actual feature names)
feature_names = [f"Feature {i}" for i in range(X.shape[1])]

# Create a residual plot with Shapley values
create_residual_plot_with_shapley(model, X_test, y_test, feature_names)
