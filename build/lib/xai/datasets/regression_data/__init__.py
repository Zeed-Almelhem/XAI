import pandas as pd

def load_admission_prediction_data():
    """
    Load the Admission Prediction dataset from the regression_data directory.
    
    Returns:
    DataFrame: A Pandas DataFrame containing the Admission Prediction dataset.
    """
    data_path = r"https://raw.githubusercontent.com/Zeed-Almelhem/XAI/main/xai/datasets/regression_data/Admission_Prediction.csv"
    df = pd.read_csv(data_path)
    return df

def load_bike_sharing_demand_data():
    """
    Load the Bike Sharing Demand dataset from the regression_data directory.
    
    Returns:
    DataFrame: A Pandas DataFrame containing the Bike Sharing Demand dataset.
    """
    data_path = r"https://raw.githubusercontent.com/Zeed-Almelhem/XAI/main/xai/datasets/regression_data/bike_sharing_demand.csv"
    df = pd.read_csv(data_path)
    return df