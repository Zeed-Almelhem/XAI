# Import the libraries:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
import joblib

# Build a Linear Regression Model & a Lasso Regression Model:

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

