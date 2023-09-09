## Test 6

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from XAI.xai.regression.visualizations import create_residual_plots

# Generate a sample dataset for demonstration
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
df = pd.DataFrame({'X': X[:, 0], 'Y': y})

# Create and train a linear regression model
model = LinearRegression()
model.fit(df[['X']], df['Y'])

# Call the create_residual_plots function to create residual plots
create_residual_plots(model, df, 'Y')