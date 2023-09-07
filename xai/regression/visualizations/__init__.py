# Import the important libraries:
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go  # Import the graph_objs module
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score, explained_variance_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, \
    mean_squared_log_error
from scipy import stats
from statsmodels.graphics.gofplots import qqplot
import shap


# Function 1 - Scatter Plot:

def create_two_column_scatter_plot(
    model, df, x_column, y_column, title=None, xlabel=None, ylabel=None, figsize=(8, 6),
    save_path=None, **kwargs
):
    """
    Create a visually appealing scatter plot to visualize the relationship between two columns (x and y) using Seaborn.

    Parameters:
        model: A trained machine learning regression model (e.g., sklearn model).
        df (pandas.DataFrame): The input data frame containing the dataset.
        x_column (str): The name of the column to use for the x-axis.
        y_column (str): The name of the column to use for the y-axis.
        title (str, optional): The title of the scatter plot.
        xlabel (str, optional): The label for the x-axis.
        ylabel (str, optional): The label for the y-axis.
        figsize (tuple, optional): Figure size (width, height).
        save_path (str, optional): Path to save the plot as an image (e.g., 'scatter_plot.png').
        **kwargs: Additional keyword arguments to pass to Seaborn's scatterplot.

    Returns:
        matplotlib.figure.Figure: The created scatter plot.
    """
    # Set Seaborn style to darkgrid for a dark background
    sns.set(style="darkgrid", palette="pastel")

    # Extract the specified columns from the DataFrame
    x_data = df[x_column]
    y_actual = df[y_column]

    # Use the model to make predictions
    y_predicted = model.predict(x_data.values.reshape(-1, 1))

    # Create the scatter plot using Seaborn with improved aesthetics
    plt.figure(figsize=figsize)
    scatter = sns.scatterplot(x=x_data, y=y_actual, label="Actual", s=100, color="#3498db", alpha=0.7, **kwargs)
    line = sns.lineplot(x=x_data, y=y_predicted, label="Predicted", color="#e74c3c", linewidth=2, **kwargs)
    plt.xlabel(xlabel if xlabel else x_column, fontsize=12, color="black")
    plt.ylabel(ylabel if ylabel else y_column, fontsize=12, color="black")
    plt.title(title if title else f"Scatter Plot: {x_column} vs. {y_column}", fontsize=14, color="black")
    plt.legend(fontsize=12, loc='upper left', frameon=True, facecolor='white', edgecolor='black')

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

    return plt

### Test 1

# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# #from create_scatter_plot import create_two_column_scatter_plot  # Import the custom function

# # Generate synthetic data for demonstration
# np.random.seed(0)
# X = np.random.rand(100, 1) * 10
# y = 2 * X + 1 + np.random.randn(100, 1)

# # Create a pandas DataFrame from the synthetic data
# df = pd.DataFrame({'X': X.flatten(), 'y': y.flatten()})

# # Create and train a simple linear regression model
# model = LinearRegression()
# model.fit(df[['X']], df['y'])

# # Specify customization options
# title = "Scatter Plot: X vs. y"
# xlabel = "X-axis"
# ylabel = "y-axis"
# figsize = (8, 6)
# save_path = "scatter_plot.png"  # Optional: Specify a file path to save the plot

# # Create the scatter plot with improved aesthetics
# create_two_column_scatter_plot(model, df, 'X', 'y', title=title, xlabel=xlabel, ylabel=ylabel, figsize=figsize, save_path=save_path)


# Function 2 - Scatter Plot:

def create_interactive_scatter_plot(model, df, x_column_names, y_column_name):
    """
    Create an interactive scatter plot using Plotly and Dash with a dynamic dropdown menu for x-axis column selection
    and display the fitted model line.

    Parameters:
        model: A trained machine learning regression model (e.g., sklearn model).
        df (pandas.DataFrame): The input data frame containing the dataset.
        x_column_names (list): A list of column names for the x-axis dropdown menu.
        y_column_name (str): The name of the column to use for the y-axis.

    Returns:
        dash.Dash: The Dash app for the interactive scatter plot.
    """
    # Initialize the Dash app
    app = dash.Dash(__name__)

    # Define custom CSS for improved aesthetics
    app.css.append_css({'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

    # Define the layout of the app
    app.layout = html.Div([
        html.Label('Select X-Axis Column:'),
        dcc.Dropdown(
            id='x-axis-dropdown',
            options=[{'label': col, 'value': col} for col in x_column_names],
            value=x_column_names[0]  # Default selection
        ),
        dcc.Graph(id='scatter-plot'),
    ])

    # Define the callback to update the scatter plot
    @app.callback(
        Output('scatter-plot', 'figure'),
        [Input('x-axis-dropdown', 'value')]
    )
    def update_scatter_plot(selected_x_column):
        # Use the model to make predictions for the selected x-column
        x_data = df[selected_x_column]
        y_predicted = model.predict(x_data.values.reshape(-1, 1))
        
        # Create a scatter plot
        fig = go.Figure()

        # Add scatter points
        hover_texts = []  # Store hover texts
        for i in range(len(df)):
            hover_text = f'X: {x_data.iloc[i]:.2f}<br>Y: {df[y_column_name].iloc[i]:.2f}<br>Predicted: {y_predicted[i]:.2f}'
            hover_texts.append(hover_text)
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=df[y_column_name],
            mode='markers',
            name='Actual',
            text=hover_texts,
            hoverinfo='text',
            marker=dict(color='blue'),  # Set a single color
        ))
        
        # Add the fitted model line to the plot
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_predicted,
            mode='lines',
            name='Fitted Model',
            hoverinfo='none'
        ))

        # Customize the layout
        fig.update_layout(
            title=f'Scatter Plot: {selected_x_column} vs. {y_column_name}',
            xaxis_title=selected_x_column,
            yaxis_title=y_column_name,
            paper_bgcolor='rgb(17, 17, 17)',  # Dark background color
            plot_bgcolor='rgb(17, 17, 17)',  # Dark plot area color
            font=dict(color='white'),  # Text color
            legend=dict(font=dict(color='white')),  # Legend text color
        )

        return fig

    # Run the app
    if __name__ == '__main__':
        app.run_server(debug=True)

### Test 2 

# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression

# # Sample data
# data = {
#     'x1': np.arange(1, 11),
#     'x2': np.arange(2, 22, 2),
#     'y': np.array([2.3, 4.5, 6.7, 8.9, 11.1, 13.3, 15.5, 17.7, 19.9, 22.1])
# }

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Define the x-column names (you can provide your own list of column names)
# x_column_names = ['x1', 'x2']

# # Create a linear regression model
# model = LinearRegression()
# model.fit(df[x_column_names[0]].values.reshape(-1, 1), df['y'])

# # Use the create_interactive_scatter_plot function
# create_interactive_scatter_plot(model, df, x_column_names, 'y')


# Function 3 - Scatter Plot:

def create_3d_scatter_plot(model, df, x_column1, x_column2, y_column):
    """
    Create a 3D scatter plot with a fitted model based on three specified columns.

    Parameters:
        model: A trained machine learning regression model (e.g., sklearn model).
        df (pandas.DataFrame): The input data frame containing the dataset.
        x_column1 (str): The name of the first column for the x-axis.
        x_column2 (str): The name of the second column for the y-axis.
        y_column (str): The name of the column to use for the z-axis.

    Returns:
        go.Figure: The 3D scatter plot figure.
    """
    # Use the model to make predictions
    x_data1 = df[x_column1]
    x_data2 = df[x_column2]
    y_predicted = model.predict(np.column_stack((x_data1, x_data2)))

    # Create a 3D scatter plot
    fig = go.Figure()

    # Add scatter points
    hover_texts = []  # Store hover texts
    for i in range(len(df)):
        hover_text = f'{x_column1}: {x_data1.iloc[i]:.2f}<br>{x_column2}: {x_data2.iloc[i]:.2f}<br>{y_column}: {df[y_column].iloc[i]:.2f}<br>Predicted: {y_predicted[i]:.2f}'
        hover_texts.append(hover_text)

    fig.add_trace(go.Scatter3d(
        x=x_data1,
        y=x_data2,
        z=df[y_column],
        mode='markers',
        text=hover_texts,
        hoverinfo='text',
        marker=dict(size=5, color='blue'),  # Set marker size and color
        name='Actual',
    ))

    # Add the fitted model surface to the plot
    x_min, x_max = x_data1.min(), x_data1.max()
    y_min, y_max = x_data2.min(), x_data2.max()
    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
    z_grid = model.predict(np.column_stack((x_grid.ravel(), y_grid.ravel())))
    z_grid = z_grid.reshape(x_grid.shape)

    fig.add_trace(go.Surface(
        x=x_grid,
        y=y_grid,
        z=z_grid,
        colorscale='Viridis',
        opacity=0.7,
        name='Fitted Model',
    ))

    # Customize the layout
    fig.update_layout(
        scene=dict(
            xaxis_title=x_column1,
            yaxis_title=x_column2,
            zaxis_title=y_column,
        ),
        title=f'3D Scatter Plot: {x_column1}, {x_column2}, {y_column}',
        scene_bgcolor='rgb(17, 17, 17)',  # Dark background color
        font=dict(color='white'),  # Text color
        legend=dict(font=dict(color='white')),  # Legend text color
    )

    return fig

### Test 3 

# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# import plotly.offline as pyo

# # Sample data
# data = {
#     'Column1': np.random.rand(100),
#     'Column2': np.random.rand(100),
#     'TargetColumn': 2 * np.random.rand(100) + 3,
# }

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Train a regression model
# X = df[['Column1', 'Column2']]
# y = df['TargetColumn']
# model = LinearRegression()
# model.fit(X, y)

# # Create the 3D scatter plot
# fig = create_3d_scatter_plot(model, df, 'Column1', 'Column2', 'TargetColumn')

# # Show the plot
# pyo.plot(fig, filename='3d_scatter_plot.html')



# Function 4 - Scatter Plot:

def create_3d_scatter_dashboard(dataframe, x_columns, y_column, model):
    """

    Creates a 3D scatter plot dashboard using Dash with Plotly.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        x_columns (list): List of column names to be used as X-axis variables.
        y_column (str): The column name to be used as the Y-axis variable.
        model (sklearn.base.BaseEstimator): The machine learning model to fit and predict the data.

    Returns:
        dash.Dash: The Dash app object.

    """

    # Initialize the Dash app
    app = dash.Dash(__name__)

    # Define the app layout
    app.layout = html.Div([
        html.H1('3D Scatter Plot Dashboard'),
        html.Label('Select X Column 1'),
        dcc.Dropdown(
            id='x1-dropdown',
            options=[{'label': col, 'value': col} for col in x_columns],
            value=x_columns[0],
        ),
        html.Label('Select X Column 2'),
        dcc.Dropdown(
            id='x2-dropdown',
            options=[{'label': col, 'value': col} for col in x_columns],
            value=x_columns[1],
        ),
        dcc.Graph(id='scatter-plot'),
    ])

    # Define callback to update the scatter plot
    @app.callback(
        Output('scatter-plot', 'figure'),
        Input('x1-dropdown', 'value'),
        Input('x2-dropdown', 'value'),
    )
    def update_scatter_plot(x1_column, x2_column):
        # Extract the selected x columns
        x1 = dataframe[x1_column].values
        x2 = dataframe[x2_column].values
        y = dataframe[y_column].values

        # Create a 3D scatter plot for the data points
        trace_data = go.Scatter3d(
            x=x1,
            y=x2,
            z=y,
            mode='markers',
            marker=dict(
                size=5,
                color=y,
                colorscale='Viridis',
                opacity=0.8,
            ),
            name='Actual Data',
        )

        # Create a meshgrid for the X and Y axes
        x1_range = np.linspace(min(x1), max(x1), 50)
        x2_range = np.linspace(min(x2), max(x2), 50)
        x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)

        # Predict the model for the meshgrid
        x_combined = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]
        y_predicted = model.predict(x_combined)

        # Reshape the predicted values for plotting
        y_mesh = y_predicted.reshape(x1_mesh.shape)

        # Create a 3D surface plot for the model shape
        trace_model = go.Surface(
            x=x1_range,
            y=x2_range,
            z=y_mesh,
            colorscale='Viridis',
            opacity=0.7,
            showscale=False,
            name='Model Shape',
        )

        layout = go.Layout(
            scene=dict(
                xaxis=dict(title=x1_column),
                yaxis=dict(title=x2_column),
                zaxis=dict(title=y_column),
            ),
            margin=dict(l=0, r=0, b=0, t=0),
        )

        return {'data': [trace_data, trace_model], 'layout': layout}

    return app

### Test 4 

# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression

# # Sample data
# data = {
#     'Column1': np.random.rand(100),
#     'Column2': np.random.rand(100),
#     'TargetColumn': 2 * np.random.rand(100) + 3,
# }

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Names of X columns and Y column
# x_columns = ['Column1', 'Column2']
# y_column = 'TargetColumn'

# # Sample model (you can replace this with your own model)
# model = LinearRegression()

# # Train the model on your data
# X = df[x_columns].values
# y = df[y_column].values
# model.fit(X, y)  # Fit the model to your data

# # Create the Dash app
# app = create_3d_scatter_dashboard(df, x_columns, y_column, model)  # Pass the trained and fitted model

# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)


### Test 5 : you have to give the function all the columns that you used to fit the model with
 
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression

# # Sample data
# data = {
#     'X1': np.random.rand(100),
#     'X2': np.random.rand(100),
#     'Y': 2 * np.random.rand(100) + 3,
# }

# # Create a DataFrame
# df = pd.DataFrame(data)

# # Names of X columns and Y column
# x_columns = ['X1', 'X2']
# y_column = 'Y'

# # Create polynomial features
# degree = 2
# poly = PolynomialFeatures(degree=degree, include_bias=False)
# X_poly = poly.fit_transform(df[x_columns])

# # Fit a polynomial regression model
# model = LinearRegression()
# model.fit(X_poly, df[y_column])

# # Create DataFrames for X_poly and Y
# df_x_poly = pd.DataFrame(X_poly, columns=[f'Poly_{i+1}' for i in range(X_poly.shape[1])])
# df_y = df[[y_column]]

# # Create the Dash app for the 3D scatter plot with the model shape
# app = create_3d_scatter_dashboard(df_x_poly, x_columns, df_y, model)

# # Run the app
# if __name__ == '__main__':
#     app.run_server(debug=True)


# Function 5 - Residual Plot:

def create_residual_plots(model, dataset, target_column):
    """
    Create residual plots for a regression model.

    Parameters:
    - model: A trained regression model.
    - dataset: The dataset containing predictor variables and the target variable.
    - target_column: The name of the target variable in the dataset.

    Returns:
    - fig: Matplotlib figure object containing the subplots.
    """
    # Make predictions using the model
    predicted = model.predict(dataset.drop(target_column, axis=1))

    # Calculate residuals (differences between actual and predicted values)
    residuals = dataset[target_column] - predicted

    # Create residual plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Residuals vs. Predicted Values
    sns.scatterplot(x=predicted, y=residuals, color='blue', ax=axes[0])
    axes[0].axhline(y=0, color='red', linestyle='--')
    axes[0].set_title("Residuals vs. Predicted Values")
    axes[0].set_xlabel("Predicted Values")
    axes[0].set_ylabel("Residuals")

    # Residual Distribution
    sns.histplot(residuals, kde=True, color='green', ax=axes[1])
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residuals")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()

    # Return the Matplotlib figure object
    plt.show()
    return fig

### Test 6

# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.datasets import make_regression
# import matplotlib.pyplot as plt

# # Generate a sample dataset for demonstration
# X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
# df = pd.DataFrame({'X': X[:, 0], 'Y': y})

# # Create and train a linear regression model
# model = LinearRegression()
# model.fit(df[['X']], df['Y'])

# # Call the create_residual_plots function to create residual plots
# create_residual_plots(model, df, 'Y')


# Function 6 - Feature Importance Plot:

def create_feature_importance_plot(importance, feature_names, width=800, height=500):
    """
    Create an interactive feature importance plot for a regression model using provided importance scores.

    Parameters:
    - importance: An array or list of feature importance scores.
    - feature_names: A list of feature names corresponding to the importance scores.

    Returns:
    - None

    Note:
    To obtain feature importance scores based on your specific regression model, please refer to the following link
    for guidance on how to calculate them:
    
    [Calculating Feature Importance with Python](https://machinelearningmastery.com/calculate-feature-importance-with-python/)
    """
     
    # Create a DataFrame to store feature importance data
    data = {"Feature Names": feature_names, "Importance Score": importance}
    
    # Create an interactive bar plot using Plotly
    fig = px.bar(data, x="Importance Score", y="Feature Names", orientation="h", text="Importance Score",
                 title="Feature Importance Plot", labels={"Importance Score": "Importance Score"},
                 color_discrete_sequence=["#1f77b4"] * len(data))  # Set custom color

    # Customize hover text
    fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')

    # Configure plot layout
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        xaxis_title="Importance Score",
        yaxis_title="Feature Names",
        font=dict(size=14, color="white"),
        paper_bgcolor="black",  # Set dark black background
        plot_bgcolor="black",  # Set dark black background
    )

    # Show the interactive plot
    fig.show()
    return fig

### Test 7

# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# import plotly.express as px

# # Sample data
# np.random.seed(42)
# X = np.random.rand(100, 3)  # Three input features
# y = 2*X[:, 0] + 3*X[:, 1] + 1*X[:, 2] + np.random.randn(100)  # Linear relationship with noise

# # Train a linear regression model
# model = LinearRegression()
# model.fit(X, y)

# # Extract feature importances (in this case, coefficients)
# feature_importance = model.coef_

# # Feature names (assuming you have them)
# feature_names = ["Feature 1", "Feature 2", "Feature 3"]

# # Create the feature importance plot
# create_feature_importance_plot(feature_importance, feature_names, height=700, width=1400)


# Function 7 - Partial Dependence Plot (PDP):

def create_actual_vs_predicted_distribution(actual, predicted, title='Actual vs. Predicted Distribution', width=800, height=400):
    """
    Create an Actual vs. Predicted Value Distribution plot.

    Parameters:
    - actual: A pandas Series or list containing actual target values.
    - predicted: A pandas Series or list containing predicted target values.
    - title: The title of the plot.
    - width: The width of the plot.
    - height: The height of the plot.

    Returns:
    - A Plotly figure for the distribution plot.

    Example:
    create_actual_vs_predicted_distribution(actual, predicted)
    """

    # Create a DataFrame to store the data
    data = {"Actual": actual, "Predicted": predicted}

    # Create a scatter plot with histograms using Plotly
    fig = px.scatter(data, x="Actual", y="Predicted", marginal_x="histogram", marginal_y="histogram", title=title,
                     labels={"Actual": "Actual", "Predicted": "Predicted"},
                     color_discrete_sequence=["#1f77b4"])
    
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    # Customize plot layout
    fig.update_layout(
        width=width,
        height=height,
        font=dict(size=14, color="white"),  # Set text color to white
        paper_bgcolor="black",  # Set dark background
        plot_bgcolor="black",  # Set dark background
    )

    # Show the interactive plot
    fig.show()
    return fig

### Test 8

# import pandas as pd
# import numpy as np

# # Sample data
# actual_values = np.random.rand(100) * 10  # Actual target values
# predicted_values = actual_values + np.random.normal(0, 1, 100)  # Predicted target values (with some noise)

# # Create a DataFrame from the sample data
# data = pd.DataFrame({'Actual': actual_values, 'Predicted': predicted_values})

# # Create the Actual vs. Predicted Value Distribution plot
# create_actual_vs_predicted_distribution(data['Actual'], data['Predicted'], title='Sample Actual vs. Predicted Distribution', height=800, width= 1600)


# Function 8 - QQ Plot (Quantile-Quantile Plot):

def create_qq_plot(residuals, title="QQ Plot", figsize=(8, 6), color="#1f77b4"):
    """
    Create a QQ Plot to assess whether residuals follow a normal distribution using Seaborn.

    Parameters:
    - residuals: A NumPy array or list containing the residuals of a regression model.
    - title: Title for the QQ Plot (default is "QQ Plot").
    - figsize: Figure size (width, height) in inches (default is (8, 6)).
    - color: Color for the QQ Plot points (default is "#1f77b4").

    Returns:
    - None

    [How to interpret a QQ plot](https://stats.stackexchange.com/questions/101274/how-to-interpret-a-qq-plot)
    """

    # Sort residuals
    sorted_residuals = np.sort(residuals)

    # Generate theoretical quantiles for a normal distribution
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))

    # Create QQ Plot using Seaborn
    plt.figure(figsize=figsize)
    sns.scatterplot(x=theoretical_quantiles, y=sorted_residuals, color=color, edgecolor="k", alpha=0.7)
    plt.title(title)
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sorted Residuals")

    # Customize plot
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Show the QQ Plot
    plt.show()

### Test 9

# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Example usage:
# # Create some sample data
# np.random.seed(0)
# X = np.random.rand(100, 2)
# y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100)

# # Fit a linear regression model
# X_with_const = sm.add_constant(X)  # Add a constant term
# model = sm.OLS(y, X_with_const).fit()

# # Calculate residuals
# residuals = y - model.predict(X_with_const)

# # Create a QQ Plot
# create_qq_plot(residuals, title="QQ Plot for Residuals")



# Function 9 - Calculate regression evaluation metrics:

def visualize_regression_metrics(y_true, y_pred):
    """
    Visualize various regression evaluation metrics for model performance.

    Parameters:
    - y_true: Array-like, true target values.
    - y_pred: Array-like, predicted target values.

    Returns:
    - None
    """
    # Calculate regression evaluation metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    # Create a bar plot to visualize metrics
    metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'R-squared']
    values = [mae, mse, rmse, mape, r2]

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=values, y=metrics, palette="viridis")

    # Add labels and values to the bars
    for i, v in enumerate(values):
        ax.text(v + 0.005, i, f'{v:.4f}', va='center', color='black', fontsize=12)

    # Customize the plot
    plt.title("Regression Evaluation Metrics", fontsize=16)
    plt.xlabel("Metric Value", fontsize=14)
    plt.ylabel("Metric", fontsize=14)
    plt.xlim(0, max(values) + 0.1)

    # Show the plot
    plt.show()

### Test 9


# # Create a synthetic regression dataset
# X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Make predictions on the test data
# y_pred = model.predict(X_test)

# # Visualize regression metrics
# visualize_regression_metrics(y_test, y_pred)



# Function 9 - Calculate advanced regression evaluation metrics:

def visualize_advanced_regression_metrics(y_true, y_pred, custom_losses=None):
    """
    Create visualizations for advanced regression evaluation metrics.

    Parameters:
    - y_true: Array-like, true target values.
    - y_pred: Array-like, predicted target values.
    - custom_losses: List of tuples (name, function), where each function calculates a custom loss.

    Returns:
    - None
    """

    # Initialize subplots
    num_metrics = 8  # Number of metrics (excluding custom losses)
    num_rows = 2 if custom_losses else 4  # Number of subplot rows

    fig, axes = plt.subplots(num_rows, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.4)
    fig.suptitle("Advanced Regression Metrics Visualizations", fontsize=16)

    # Metric names
    metric_names = ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)",
                    "Mean Bias Deviation (MBD)", "Coefficient of Determination (COD)",
                    "Explained Variance Score", "Fraction of Explained Variance (FEV)", "Huber Loss"]
    # Define metric functions
    def mean_absolute_error(y, y_pred):
        return np.mean(np.abs(y - y_pred))

    def mean_squared_error_func(y, y_pred):
        return np.mean((y - y_pred) ** 2)

    def root_mean_squared_error(y, y_pred):
        return np.sqrt(mean_squared_error(y, y_pred))

    def mean_absolute_percentage_error(y, y_pred):
        return np.mean(np.abs((y - y_pred) / y)) * 100

    def huber_loss(y, y_pred):
        return np.mean(np.where(np.abs(y - y_pred) < 1, 0.5 * (y - y_pred) ** 2, np.abs(y - y_pred) - 0.5))
    
    # Metric functions
    metric_functions = [
    mean_absolute_error,
    mean_squared_error_func,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
    explained_variance_score,
    explained_variance_score,
    huber_loss
    ]


    # Calculate and plot metrics
    for i, ax in enumerate(axes.flatten()):
        if i >= num_metrics:
            break
        metric_name = metric_names[i]
        metric_func = metric_functions[i]

        # Calculate metric
        if metric_func == explained_variance_score:
            metric_value = metric_func(y_true, y_pred, multioutput="uniform_average")
        else:
            metric_value = metric_func(y_true, y_pred)

        # Plot metric
        residuals = np.array(y_true - y_pred)  # Convert residuals to a NumPy array
        sns.histplot(residuals, kde=True, ax=ax)
        ax.set_title(f"{metric_name}\n{metric_value:.4f}", fontsize=12)
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Frequency")

    # Custom Losses
    if custom_losses:
        for i, (loss_name, loss_function) in enumerate(custom_losses):
            ax = axes[num_rows - 1, i % 2]  # Place custom loss plots in the last row
            # Calculate custom loss residuals and convert them to a NumPy array
            custom_residuals = np.array([loss_function(y_t, y_p) for y_t, y_p in zip(y_true, y_pred)])
            # Plot custom loss histogram using matplotlib
            ax.hist(custom_residuals, bins=30, alpha=0.7, color='b', label=loss_name)
            ax.set_title(loss_name, fontsize=12)
            ax.set_xlabel("Custom Loss Residuals")
            ax.set_ylabel("Frequency")
            ax.legend()

    plt.show()


### Test 10

# import numpy as np
# from sklearn.datasets import make_regression
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split

# # Generate synthetic regression data
# X, y = make_regression(n_samples=100, n_features=2, noise=0.5, random_state=42)

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and train a Linear Regression model
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Make predictions on the test data
# y_pred = model.predict(X_test)

# # Calculate additional custom loss functions if needed
# def custom_loss_function1(y_true, y_pred):
#     # Implement your custom loss calculation here
#     return np.mean(np.abs(y_true - y_pred))

# def custom_loss_function2(y_true, y_pred):
#     # Implement another custom loss calculation here
#     return np.mean((y_true - y_pred) ** 2)

# # Visualize advanced regression metrics including custom losses
# visualize_advanced_regression_metrics(y_test, y_pred, custom_losses=[("Custom Loss 1", custom_loss_function1),
#                                                                    ("Custom Loss 2", custom_loss_function2)])



# Function 10 - Calculate advanced regression evaluation metrics:

def residual_plot_with_shapley(model, X_test, y_test, feature_names=None):
    # Ensure that the model is callable
    if not callable(getattr(model, "predict", None)):
        raise TypeError("The passed model is not callable and cannot be analyzed directly with the given masker! Model: " + str(model))
    
    # Get model predictions
    y_pred = model.predict(X_test)
    
    # Calculate residuals
    residuals = y_test - y_pred
    
    # Initialize the explainer
    explainer = shap.Explainer(model)
    
    # Calculate Shapley values
    shapley_values = explainer.shap_values(X_test)
    
    # Flatten Shapley values and residuals
    shapley_values_flat = shapley_values.flatten()
    residuals_flat = residuals.flatten()
    
    # Ensure that both arrays have the same length
    min_len = min(len(shapley_values_flat), len(residuals_flat))
    shapley_values_flat = shapley_values_flat[:min_len]
    residuals_flat = residuals_flat[:min_len]
    
    # Create a DataFrame for plotting
    df = pd.DataFrame({"Shapley Values": shapley_values_flat, "Residuals": residuals_flat})
    
    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Shapley Values"], df["Residuals"], alpha=0.5)
    plt.xlabel("Shapley Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot with Shapley Values")
    
    # Show the plot
    plt.show()

### Test 11

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.datasets import make_regression
# import shap
# import matplotlib.pyplot as plt

# # Generate a synthetic regression dataset
# X, y = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a RandomForestRegressor
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Define feature names (replace with your actual feature names)
# feature_names = [f"Feature {i}" for i in range(X.shape[1])]

# # Create a residual plot with Shapley values
# residual_plot_with_shapley(model, X_test, y_test, feature_names)