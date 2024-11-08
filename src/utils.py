import pandas as pd
import numpy as np

def load_housing_data():
    """
    Load the California housing data from local CSV file.
    Returns a pandas DataFrame with the housing data.
    """
    try:
        # Read the CSV file
        df = pd.read_csv('data/housing.csv')
        
        print("Data loaded successfully!")
        return df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def verify_data():
    """Verify that the housing data was loaded correctly."""
    try:
        df = pd.read_csv('data/housing.csv')
        print("\nData verification:")
        print(f"Number of records: {len(df)}")
        print("\nFirst few records:")
        print(df.head())
        print("\nData statistics:")
        print(df.describe())
        print("\nMissing values:")
        print(df.isnull().sum())
        print("\nUnique values in ocean_proximity:")
        print(df['ocean_proximity'].value_counts())
        return True
    except Exception as e:
        print(f"Error verifying data: {str(e)}")
        return False

def format_price(price):
    """Format house price in a readable format."""
    return f"${price:,.2f}"