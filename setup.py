from setuptools import setup, find_packages

dependencies = [
    "matplotlib",
    "pandas",
    "seaborn",
    "dash",
    "dash-core-components",
    "dash-html-components",
    "plotly",
    "numpy",
    "scikit-learn",
    "scipy",
    "statsmodels",
    "shap",
]

keywords=[
    'xai','xaiz', 'explainable ai', 'interpretable ai', 'machine learning',
    'deep learning', 'artificial intelligence', 'data science',
    'model interpretability', 'model transparency', 'model explanation',
    'model visualization', 'regression analysis', 'classification',
    'supervised learning', 'unsupervised learning', 'reinforcement learning',
    'data visualization', 'feature importance', 'regression metrics',
    'model evaluation', 'machine learning library', 'python library',
    'machine learning explainability', 'ai transparency', 'ai ethics',
    'ai accountability', 'model explainability techniques', 'ai insights',
    'open source', 'data scientists', 'machine learning practitioners'
]

long_description = """
# XAI (eXplainable Artificial Intelligence) Library

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

The XAI (eXplainable Artificial Intelligence) Library is a powerful toolkit designed to empower data scientists and machine learning practitioners in their journey to understand, interpret, and visualize complex machine learning models. In an era of advanced AI algorithms, transparency and interpretability are paramount. XAI addresses these needs by offering a suite of advanced visualizations tailored for regression analysis, curated datasets, pre-trained regression models, and comprehensive documentation.

## Features

- **Advanced Visualizations**: Explore a rich collection of visualizations in the [XAI Regression Visualizations Module](xai/regression/visualizations/__init__.py) designed to unravel the intricacies of your regression models.
- **Datasets and Models**: Access curated datasets and pre-trained regression models to streamline your regression analysis process.
- **Model-Agnostic Explanations**: Enjoy model-agnostic explanations for compatibility with a wide array of machine learning models.
- **Interpretability for Diverse Data Types**: XAI is built to cater to diverse data types, including tabular data, image data, and natural language processing (NLP) models.
- **Continuous Improvement**: Expect regular updates with additional features and support for various machine learning tasks.

## Getting Started

To harness the power of the XAI Library, follow these steps:

1. **Install the Library**: Begin by including the library in your Python environment.
2. **Explore Documentation**: Refer to the [Getting Started Guide](https://github.com/Zeed-Almelhem/XAI/blob/main/docs/getting_started.md) to understand the library's objectives and scope.
3. **Usage Guide**: Dive into the [Usage Guide](https://github.com/Zeed-Almelhem/XAI/blob/main/docs/usage_guide.md) for detailed instructions on using the library's features.

## Examples

Explore practical examples demonstrating the library's capabilities in the [Example Notebooks](examples/) directory.

## Documentation

- [Getting Started Guide](https://github.com/Zeed-Almelhem/XAI/blob/main/docs/getting_started.md): Learn about the library's objectives and scope.
- [Usage Guide](https://github.com/Zeed-Almelhem/XAI/blob/main/docs/usage_guide.md): Detailed instructions on using the library's features.
- [Index](https://github.com/Zeed-Almelhem/XAI/blob/main/docs/index.md): The main page with information about the library and its creator.

## Author

- Zeed Almelhem ([GitHub](https://github.com/Zeed-Almelhem))

## Contact

For inquiries and feedback, please feel free to contact the author:

- Email (z@zeed-almelhem.com)
- [Personal Website](https://www.zeed-almelhem.com/)
- [Kaggle](https://www.kaggle.com/zeeda1melhem)
- [GitHub](https://github.com/Zeed-Almelhem)
- [Medium](https://medium.com/@zeed.almelhem)
- [Instagram](https://www.instagram.com/zeed_almelhem/)
- [LinkedIn](https://www.linkedin.com/in/zeed-almelhem/)
- [Twitter](https://twitter.com/Zeed_almelhem)
- [Blog](https://www.zeed-almelhem.com/blog)
- [Projects](https://www.zeed-almelhem.com/projects)
- [Contact](https://www.zeed-almelhem.com/contact)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

"""

setup(
    name="xai-kit",
    version="0.0.5",
    packages=find_packages(),
    package_data={
        'xai.datasets.regression_data': ['*.csv'],
        'xai.regression.models': ['*.pkl', '*.csv'],
    },
    install_requires=dependencies,
    author="Zeed Almelhem",
    author_email="info@zeed-almelhem.com",
    description="Explainable Artificial Intelligence (XAI) Library",
    long_description_content_type="text/markdown",
    long_description=long_description,
    keywords= keywords,
    url="https://github.com/Zeed-Almelhem/XAI",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

# python setup.py sdist bdist_wheel
# pip install twine
# twine upload dist/*