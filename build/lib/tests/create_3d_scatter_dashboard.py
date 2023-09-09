## Test 4 

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from XAI.xai.regression.visualizations import create_3d_scatter_dashboard
# Sample data
data = {
    'Column1': np.random.rand(100),
    'Column2': np.random.rand(100),
    'TargetColumn': 2 * np.random.rand(100) + 3,
}

# Create a DataFrame
df = pd.DataFrame(data)

# Names of X columns and Y column
x_columns = ['Column1', 'Column2']
y_column = 'TargetColumn'

# Sample model (you can replace this with your own model)
model = LinearRegression()

# Train the model on your data
X = df[x_columns].values
y = df[y_column].values
model.fit(X, y)  # Fit the model to your data

# Create the Dash app
app = create_3d_scatter_dashboard(df, x_columns, y_column, model)  # Pass the trained and fitted model

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

## Test 5 : you have to give the function all the columns that you used to fit the model with
 
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Sample data
data = {
    'X1': np.random.rand(100),
    'X2': np.random.rand(100),
    'Y': 2 * np.random.rand(100) + 3,
}

# Create a DataFrame
df = pd.DataFrame(data)

# Names of X columns and Y column
x_columns = ['X1', 'X2']
y_column = 'Y'

# Create polynomial features
degree = 2
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(df[x_columns])

# Fit a polynomial regression model
model = LinearRegression()
model.fit(X_poly, df[y_column])

# Create DataFrames for X_poly and Y
df_x_poly = pd.DataFrame(X_poly, columns=[f'Poly_{i+1}' for i in range(X_poly.shape[1])])
df_y = df[[y_column]]

# Create the Dash app for the 3D scatter plot with the model shape
app = create_3d_scatter_dashboard(df_x_poly, x_columns, df_y, model)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
