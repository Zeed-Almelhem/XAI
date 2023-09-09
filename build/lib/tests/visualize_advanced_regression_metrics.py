## Test 10

import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from XAI.xai.regression.visualizations import visualize_advanced_regression_metrics

# Generate synthetic regression data
X, y = make_regression(n_samples=100, n_features=2, noise=0.5, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate additional custom loss functions if needed
def custom_loss_function1(y_true, y_pred):
    # Implement your custom loss calculation here
    return np.mean(np.abs(y_true - y_pred))

def custom_loss_function2(y_true, y_pred):
    # Implement another custom loss calculation here
    return np.mean((y_true - y_pred) ** 2)

# Visualize advanced regression metrics including custom losses
visualize_advanced_regression_metrics(y_test, y_pred, custom_losses=[("Custom Loss 1", custom_loss_function1),
                                                                   ("Custom Loss 2", custom_loss_function2)])
