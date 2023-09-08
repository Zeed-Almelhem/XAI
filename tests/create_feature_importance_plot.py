## Test 7

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import plotly.express as px
from XAI.xai.regression.visualizations import create_feature_importance_plot

# Sample data
np.random.seed(42)
X = np.random.rand(100, 3)  # Three input features
y = 2*X[:, 0] + 3*X[:, 1] + 1*X[:, 2] + np.random.randn(100)  # Linear relationship with noise

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Extract feature importances (in this case, coefficients)
feature_importance = model.coef_

# Feature names (assuming you have them)
feature_names = ["Feature 1", "Feature 2", "Feature 3"]

# Create the feature importance plot
create_feature_importance_plot(feature_importance, feature_names, height=700, width=1400)


