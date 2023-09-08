## Test 12:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import plotly.express as px

# Generate sample data
np.random.seed(0)
data = {
    'Feature1': np.random.rand(100),
    'Feature2': np.random.rand(100),
    'Target': 2 * np.random.rand(100) + 3
}
df = pd.DataFrame(data)

# Split the data into features and target
X = df[['Feature1', 'Feature2']]
y = df['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Add predicted values to the test data
df_test = X_test.copy()
df_test['Target'] = y_test
df_test['Predicted'] = model.predict(X_test)

# Create an error heatmap by feature
create_error_heatmap(df_test, 'Target', model, ['Feature1', 'Feature2'])
