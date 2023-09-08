## Test 1

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from XAI.xai.regression.visualizations import create_two_column_scatter_plot

#from create_scatter_plot import create_two_column_scatter_plot  # Import the custom function

# Generate synthetic data for demonstration
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1)

# Create a pandas DataFrame from the synthetic data
df = pd.DataFrame({'X': X.flatten(), 'y': y.flatten()})

# Create and train a simple linear regression model
model = LinearRegression()
model.fit(df[['X']], df['y'])

# Specify customization options
title = "Scatter Plot: X vs. y"
xlabel = "X-axis"
ylabel = "y-axis"
figsize = (8, 6)
save_path = "scatter_plot.png"  # Optional: Specify a file path to save the plot

# Create the scatter plot with improved aesthetics
create_two_column_scatter_plot(model, df, 'X', 'y', title=title, xlabel=xlabel, ylabel=ylabel, figsize=figsize, save_path=save_path)
