from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import joblib
import pandas as pd

def train_model(X_train, y_train):
    """Train the housing price prediction model."""
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model's performance."""
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Calculate percentage error
    mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
    
    return {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }

def make_prediction(model, preprocessor, new_data):
    """
    Make house price predictions for new data.
    
    Parameters:
    - model: Trained GradientBoostingRegressor model
    - preprocessor: Fitted ColumnTransformer
    - new_data: DataFrame with required features
    
    Returns:
    - Predicted house price
    """
    # Process the new data similar to training data
    new_data = new_data.copy()
    
    # Create derived features
    new_data['rooms_per_household'] = new_data['total_rooms'] / new_data['households']
    new_data['bedrooms_per_room'] = new_data['total_bedrooms'] / new_data['total_rooms']
    new_data['population_per_household'] = new_data['population'] / new_data['households']
    new_data['median_income'] = np.log1p(new_data['median_income'])
    
    # Transform features
    X_new_processed = preprocessor.transform(new_data)
    
    # Make prediction
    prediction = model.predict(X_new_processed)
    
    return prediction[0]