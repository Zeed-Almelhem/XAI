## Test 9

import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from XAI.xai.regression.visualizations import create_qq_plot

# Example usage:
# Create some sample data
np.random.seed(0)
X = np.random.rand(100, 2)
y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100)

# Fit a linear regression model
X_with_const = sm.add_constant(X)  # Add a constant term
model = sm.OLS(y, X_with_const).fit()

# Calculate residuals
residuals = y - model.predict(X_with_const)

# Create a QQ Plot
create_qq_plot(residuals, title="QQ Plot for Residuals")