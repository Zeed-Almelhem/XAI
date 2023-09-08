## Test 8

import pandas as pd
import numpy as np
from XAI.xai.regression.visualizations import create_actual_vs_predicted_distribution

# Sample data
actual_values = np.random.rand(100) * 10  # Actual target values
predicted_values = actual_values + np.random.normal(0, 1, 100)  # Predicted target values (with some noise)

# Create a DataFrame from the sample data
data = pd.DataFrame({'Actual': actual_values, 'Predicted': predicted_values})

# Create the Actual vs. Predicted Value Distribution plot
create_actual_vs_predicted_distribution(data['Actual'], data['Predicted'], title='Sample Actual vs. Predicted Distribution', height=800, width= 1600)
