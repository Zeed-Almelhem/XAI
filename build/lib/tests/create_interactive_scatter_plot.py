## Test 2 

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from XAI.xai.regression.visualizations import create_interactive_scatter_plot

# Sample data
data = {
    'x1': np.arange(1, 11),
    'x2': np.arange(2, 22, 2),
    'y': np.array([2.3, 4.5, 6.7, 8.9, 11.1, 13.3, 15.5, 17.7, 19.9, 22.1])
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define the x-column names (you can provide your own list of column names)
x_column_names = ['x1', 'x2']

# Create a linear regression model
model = LinearRegression()
model.fit(df[x_column_names[0]].values.reshape(-1, 1), df['y'])

# Use the create_interactive_scatter_plot function
create_interactive_scatter_plot(model, df, x_column_names, 'y')