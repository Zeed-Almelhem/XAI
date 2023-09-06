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


# Function 6 - Residual Plot:


### Test 7
