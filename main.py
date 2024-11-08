import os
import pandas as pd
import numpy as np
import joblib  # Added this import
from src.data_processing import clean_data, prepare_features
from src.model import train_model, evaluate_model, make_prediction
from src.utils import load_housing_data, verify_data, format_price
from sklearn.model_selection import train_test_split

def predict_house_price(model, preprocessor):
    """Function to get user input and make predictions"""
    try:
        print("\nEnter housing data for prediction:")
        longitude = float(input("Longitude: "))
        latitude = float(input("Latitude: "))
        housing_age = float(input("Housing median age: "))
        total_rooms = float(input("Total rooms: "))
        total_bedrooms = float(input("Total bedrooms: "))
        population = float(input("Population: "))
        households = float(input("Households: "))
        median_income = float(input("Median income (in tens of thousands): "))
        ocean_proximity = input("Ocean proximity (NEAR BAY/INLAND/NEAR OCEAN/<1H OCEAN/ISLAND): ").upper()
        
        # Validate ocean proximity
        valid_locations = ['NEAR BAY', 'INLAND', 'NEAR OCEAN', '<1H OCEAN', 'ISLAND']
        if ocean_proximity not in valid_locations:
            raise ValueError(f"Ocean proximity must be one of: {', '.join(valid_locations)}")
        
        # Create a DataFrame with the input data
        new_data = pd.DataFrame({
            'longitude': [longitude],
            'latitude': [latitude],
            'housing_median_age': [housing_age],
            'total_rooms': [total_rooms],
            'total_bedrooms': [total_bedrooms],
            'population': [population],
            'households': [households],
            'median_income': [median_income],
            'ocean_proximity': [ocean_proximity]
        })
        
        # Make prediction
        predicted_price = make_prediction(model, preprocessor, new_data)
        
        print(f"\nPredicted house price: {format_price(predicted_price)}")
        
    except ValueError as e:
        print(f"Invalid input: {str(e)}")
    except Exception as e:
        print(f"Error making prediction: {str(e)}")

def main():
    try:
        # Verify the data file exists
        if not os.path.exists('data/housing.csv'):
            print("Error: housing.csv not found in data directory!")
            return
            
        # Load data
        print("Loading housing data...")
        df = load_housing_data()
        
        if df is None:
            print("Failed to load data!")
            return
            
        # Verify the data
        if not verify_data():
            print("Data verification failed. Please check the data.")
            return
        
        # Clean data
        print("Cleaning data...")
        df_cleaned = clean_data(df)
        
        # Prepare features
        print("Preparing features...")
        X, y, preprocessor, feature_names = prepare_features(df_cleaned)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        print("Training model...")
        model = train_model(X_train, y_train)
        
        # Evaluate model
        print("Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test)
        print("\nModel Performance:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model and preprocessor
        print("\nSaving model and preprocessor...")
        joblib.dump(model, 'models/house_price_model.joblib')
        joblib.dump(preprocessor, 'models/preprocessor.joblib')
        
        # Interactive prediction loop
        while True:
            predict_house_price(model, preprocessor)
            if input("\nMake another prediction? (y/n): ").lower() != 'y':
                break
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()