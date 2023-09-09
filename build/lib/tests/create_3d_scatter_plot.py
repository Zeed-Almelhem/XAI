## Test 3 

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.offline as pyo
from XAI.xai.regression.visualizations import create_3d_scatter_plot


# Sample data
data = {
    'Column1': np.random.rand(100),
    'Column2': np.random.rand(100),
    'TargetColumn': 2 * np.random.rand(100) + 3,
}

# Create a DataFrame
df = pd.DataFrame(data)

# Train a regression model
X = df[['Column1', 'Column2']]
y = df['TargetColumn']
model = LinearRegression()
model.fit(X, y)

# Create the 3D scatter plot
fig = create_3d_scatter_plot(model, df, 'Column1', 'Column2', 'TargetColumn')

# Show the plot
pyo.plot(fig, filename='3d_scatter_plot.html')


