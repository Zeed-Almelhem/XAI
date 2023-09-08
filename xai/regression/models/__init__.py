# Import the libraries:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
import joblib

# 1. Build Linear Regression Model & Lasso Regression Model:

## Load the "Admission_Prediction" dataset
data = pd.read_csv("https://raw.githubusercontent.com/Zeed-Almelhem/XAI/main/xai/datasets/regression_data/Admission_Prediction.csv")

## Split the data into features (X) and target (y)
X = data.drop(columns=["Chance of Admit"])
y = data["Chance of Admit"]

## Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Create and train a Linear Regression model
linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)

## Save the Linear Regression model to a file
joblib.dump(linear_reg_model, "linear_regression_model.pkl")

## Create and train a Lasso Regression model
lasso_reg_model = Lasso(alpha=0.01)  # You can adjust the alpha parameter as needed
lasso_reg_model.fit(X_train, y_train)

## Save the Lasso Regression model to a file
joblib.dump(lasso_reg_model, "lasso_regression_model.pkl")

## Save the training and testing data to separate files
X_train.to_csv("admission_prediction_X_train.csv", index=False)
X_test.to_csv("admission_prediction_X_test.csv", index=False)
y_train.to_csv("admission_prediction_y_train.csv", index=False)
y_test.to_csv("admission_prediction_y_test.csv", index=False)


# 1. Test the Linear Regression Model & the Lasso Regression Model that we built:
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the testing data
X_test = pd.read_csv("admission_prediction_X_test.csv")
y_test = pd.read_csv("admission_prediction_y_test.csv")

# Load the saved Linear Regression model
linear_reg_model = joblib.load("linear_regression_model.pkl")

# Load the saved Lasso Regression model
lasso_reg_model = joblib.load("lasso_regression_model.pkl")

# Predict using the Linear Regression model
linear_reg_predictions = linear_reg_model.predict(X_test)

# Predict using the Lasso Regression model
lasso_reg_predictions = lasso_reg_model.predict(X_test)

# Evaluate the Linear Regression model
linear_reg_mae = mean_absolute_error(y_test, linear_reg_predictions)
linear_reg_mse = mean_squared_error(y_test, linear_reg_predictions)
linear_reg_r2 = r2_score(y_test, linear_reg_predictions)

print("Linear Regression Metrics:")
print("Mean Absolute Error (MAE):", linear_reg_mae)
print("Mean Squared Error (MSE):", linear_reg_mse)
print("R-squared (R²):", linear_reg_r2)

# Evaluate the Lasso Regression model
lasso_reg_mae = mean_absolute_error(y_test, lasso_reg_predictions)
lasso_reg_mse = mean_squared_error(y_test, lasso_reg_predictions)
lasso_reg_r2 = r2_score(y_test, lasso_reg_predictions)

print("\nLasso Regression Metrics:")
print("Mean Absolute Error (MAE):", lasso_reg_mae)
print("Mean Squared Error (MSE):", lasso_reg_mse)
print("R-squared (R²):", lasso_reg_r2)


# 2. Build Support Vector Regression (SVR) and Random Forest Regression Models:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the "bike_sharing_demand" dataset
data = pd.read_csv("https://raw.githubusercontent.com/Zeed-Almelhem/XAI/main/xai/datasets/regression_data/bike_sharing_demand.csv")

# Convert 'datetime' column to datetime format
data['datetime'] = pd.to_datetime(data['datetime'])

# Extract year, month, and day from the 'datetime' column
data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
data['day'] = data['datetime'].dt.day

# Drop the original 'datetime' column
data.drop(columns=['datetime'], inplace=True)

# Split the data into features (X) and target (y)
X = data.drop(columns=["count"])
y = data["count"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Support Vector Regression (SVR) model
svr_model = SVR(kernel='linear')  # You can choose the kernel type as needed
svr_model.fit(X_train, y_train)

# Save the SVR model to a file
joblib.dump(svr_model, "support_vector_model.pkl")

# Create and train a Random Forest Regression model
random_forest_model = RandomForestRegressor(n_estimators=100)  # You can adjust the number of estimators as needed
random_forest_model.fit(X_train, y_train)

# Save the Random Forest Regression model to a file
joblib.dump(random_forest_model, "random_forest_model.pkl")

# Save the training and testing data to separate files
X_train.to_csv("bike_sharing_demand_X_train.csv", index=False)
X_test.to_csv("bike_sharing_demand_X_test.csv", index=False)
y_train.to_csv("bike_sharing_demand_y_train.csv", index=False)
y_test.to_csv("bike_sharing_demand_y_test.csv", index=False)


# 2. Test Support Vector Regression (SVR) and Random Forest Regression Models:

import joblib
import pandas as pd

# Load the test data
X_test = pd.read_csv("bike_sharing_demand_X_test.csv")
y_test = pd.read_csv("bike_sharing_demand_y_test.csv")

# Load the SVR model
svr_model = joblib.load("support_vector_model.pkl")

# Load the Random Forest Regression model
random_forest_model = joblib.load("random_forest_model.pkl")

# Predict using SVR model
svr_predictions = svr_model.predict(X_test)

# Predict using Random Forest model
random_forest_predictions = random_forest_model.predict(X_test)

# Calculate evaluation metrics (e.g., RMSE, R-squared)
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

svr_rmse = np.sqrt(mean_squared_error(y_test, svr_predictions))
svr_r2 = r2_score(y_test, svr_predictions)

rf_rmse = np.sqrt(mean_squared_error(y_test, random_forest_predictions))
rf_r2 = r2_score(y_test, random_forest_predictions)

# Print the evaluation metrics
print(f"SVR RMSE: {svr_rmse:.2f}")
print(f"SVR R-squared: {svr_r2:.2f}")

print(f"Random Forest RMSE: {rf_rmse:.2f}")
print(f"Random Forest R-squared: {rf_r2:.2f}")