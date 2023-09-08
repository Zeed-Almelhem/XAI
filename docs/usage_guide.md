# Usage Guide

Welcome to the XAI (eXplainable Artificial Intelligence) Library Usage Guide. In this guide, we'll walk you through the essential steps to effectively utilize the library's features for enhancing model interpretability and transparency. Whether you're new to the library or looking to explore advanced techniques, you'll find helpful instructions here.

## Table of Contents

- [Regression Analysis](#regression-analysis)
- [Classification Analysis](#classification-analysis)
- [Unsupervised Learning](#unsupervised-learning)
- [Reinforcement Learning](#reinforcement-learning)

## Regression Analysis

The XAI Library offers a rich set of tools and visualizations for in-depth regression analysis. Follow these steps to get started:

1. **Installation**: Ensure that you have the library installed..

2. **Loading Datasets**: Load a regression dataset of your choice or use one of the provided datasets.

3. **Exploratory Data Analysis (EDA)**: Begin by exploring your dataset with descriptive statistics, data visualizations, and feature engineering.

4. **Regression Visualizations**: Use advanced regression visualizations to understand your model's behavior, identify patterns, and evaluate its performance.

5. **Regression Models**: Train regression models using your dataset or explore pre-trained models provided by the library.

6. **Evaluation and Interpretation**: Evaluate your models using regression metrics and interpret their predictions with advanced visualization techniques.

In the `regression/visualizations` module, you'll find a rich collection of advanced visualizations tailored specifically for regression analysis. These visualizations serve as essential tools for gaining deeper insights into your regression models, evaluating their performance, and interpreting their behavior. Here's an in-depth look at the available visualizations:

### Two-Column Scatter Plot

`create_two_column_scatter_plot`:

This visualization allows you to create scatter plots that help you explore relationships between two columns in your dataset. By visualizing data points in a scatter plot, you can identify patterns, correlations, and potential outliers. It's a fundamental tool for understanding the relationships between your features and the target variable.

### Interactive Scatter Plot

`create_interactive_scatter_plot`:

Generate interactive scatter plots with customizable features. This visualization enables you to interactively explore data patterns by zooming in, selecting data points, and analyzing specific regions of interest. Interactive scatter plots are excellent for in-depth data exploration and hypothesis testing.

### 3D Scatter Plot

`create_3d_scatter_plot`:

Visualize complex relationships by creating three-dimensional scatter plots. With this visualization, you can study the interactions between three variables simultaneously. It's particularly useful when you're dealing with datasets with multiple features and want to understand how they collectively affect the target variable.

### 3D Scatter Dashboard

`create_3d_scatter_dashboard`:

Develop interactive 3D scatter plot dashboards for comprehensive exploration of multivariate relationships. These dashboards allow you to interact with the 3D scatter plots, change perspectives, and gain deeper insights into the interactions between different features. They are valuable for complex regression analysis.

### Residual Plots

`create_residual_plots`:

Construct residual plots to assess model performance, identify patterns, and detect potential issues like heteroscedasticity. Residual plots help you validate the assumptions of linear regression models and ensure that your model's predictions are unbiased.

### Feature Importance Plot

`create_feature_importance_plot`:

Visualize the importance of features in your regression models. This visualization provides insights into which factors significantly affect predictions. Understanding feature importance is crucial for model interpretation and feature selection.

### Actual vs. Predicted Distribution

`create_actual_vs_predicted_distribution`:

Compare the distributions of actual and predicted values to evaluate model accuracy and biases. This visualization helps you understand how well your regression model aligns with the ground truth data and identify areas where improvements may be needed.

### Quantile-Quantile (QQ) Plot

`create_qq_plot`:

Generate quantile-quantile plots to assess data distribution and model fit. QQ plots are useful for checking the normality of residuals and identifying deviations from a normal distribution. They are particularly valuable for diagnostic purposes.

### Regression Metrics Visualization

`visualize_regression_metrics`:

Visualize standard regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R² (R-squared), and more. This visualization provides a comprehensive overview of your model's performance and helps you compare different models.

### Advanced Regression Metrics

`visualize_advanced_regression_metrics`:

Explore advanced regression metrics including Mean Bias Deviation (MBD), Coefficient of Determination (COD), Explained Variance Score, Fraction of Explained Variance (FEV), Root Mean Squared Logarithmic Error (RMSLE), and even custom loss functions tailored to your specific objectives. These advanced metrics offer a deeper understanding of model behavior.

### Residual Plot with Shapley Values

`residual_plot_with_shapley`:

Combine residual plots with Shapley values to understand how individual features contribute to prediction errors. This visualization aids in feature importance analysis and highlights the impact of each feature on model predictions.

### Error Heatmap by Feature

`create_error_heatmap`:

Create error heatmaps to visualize how prediction errors vary across different feature combinations. This visualization helps you identify which feature combinations lead to larger errors, providing insights into potential model improvements.

**Datasets Available:**

1. **Admission Prediction**: A dataset for predicting university admission chances based on academic and personal attributes.

2. **Bike Sharing Demand**: A dataset containing historical bike rental data, suitable for predicting future bike rental demand.

**Pre-trained Regression Models:**

You'll find pre-trained models for various regression techniques, including:

1. **Linear Regression**: Ordinary Least Squares (OLS) linear regression.

2. **Lasso Regression**: Linear regression with L1 regularization (lasso regression).

3. **Support Vector Regression (SVR)**: A regression technique using support vector machines.

4. **Random Forest Regression**: Ensemble regression method using a collection of decision trees.

## Classification Analysis

Coming soon...

## Unsupervised Learning

Coming soon...

## Reinforcement Learning

Coming soon...

## Advanced Features

Coming soon...

## Getting Started

To start harnessing the power of the XAI library for regression analysis, follow these simple steps:

1. Install the library by including it in your Python environment.

2. Explore the regression visualizations and choose the ones that best suit your analysis needs.

3. Load the provided datasets or use your own regression datasets.

4. Utilize the pre-trained regression models or train your own models using the datasets.

5. Dive into the advanced regression metrics and visualizations to assess and interpret your models effectively.

In this first version of the XAI library, we've focused on enhancing your regression analysis capabilities. Stay tuned for upcoming releases with additional features, improvements, and support for various machine learning tasks.

Let's embark on your journey to eXplainable AI with the XAI library!

For more information, feel free to visit Zeed Almelhem's [portfolio](https://www.zeed-almelhem.com/) and connect on [LinkedIn](https://www.linkedin.com/in/zeed-almelhem), [Kaggle](https://www.kaggle.com/zeeda1melhem), [GitHub](https://github.com/Zeed-Almelhem), [Medium](https://medium.com/@zeed.almelhem), [Instagram](https://www.instagram.com/zeed_almelhem/), [Twitter](https://twitter.com/Zeed_almelhem), and [blog](https://www.zeed-almelhem.com/blog).
