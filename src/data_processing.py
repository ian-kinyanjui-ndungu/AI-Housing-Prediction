import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def clean_data(df):
    """Clean the housing dataset."""
    # Drop rows with missing values or fill them
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    
    # Create derived features
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']
    
    # Log transform skewed numerical features
    df['median_income'] = np.log1p(df['median_income'])
    
    return df

def prepare_features(df):
    """Prepare features for model training."""
    # Split features and target
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    
    # Define numeric and categorical columns
    numeric_features = ['longitude', 'latitude', 'housing_median_age', 
                       'total_rooms', 'total_bedrooms', 'population', 
                       'households', 'median_income', 'rooms_per_household',
                       'bedrooms_per_room', 'population_per_household']
    categorical_features = ['ocean_proximity']
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False))  # Changed sparse to sparse_output
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after transformation
    numeric_features_out = numeric_features
    categorical_features_out = (preprocessor
                              .named_transformers_['cat']
                              .named_steps['onehot']
                              .get_feature_names_out(categorical_features))
    feature_names = numeric_features_out + list(categorical_features_out)
    
    return X_processed, y, preprocessor, feature_names