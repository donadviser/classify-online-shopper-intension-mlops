import pandas as pd
import numpy as np
import inflection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

from utils.preprocess import (
    OutlierTransformer, RenameColumnsTransformer, FeatureEngineeringTransformer, DataFrameTransformer, DropColumnsTransformer)


def create_preprocessing_pipeline():
    numeric_features = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                        'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']
    
    categorical_features = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    return preprocessor


def preprocess_training_data(X, y):
    preprocess_pipeline = create_preprocessing_pipeline()
    X_preprocessed = preprocess_pipeline.fit_transform(X).toarray()
    
    # Ensure X_preprocessed is at least 2D
    if X_preprocessed.ndim < 2:
        X_preprocessed = X_preprocessed.reshape(-1, 1)

    # Retrieve feature names after transformation
    numeric_features = preprocess_pipeline.transformers_[0][2]
    categorical_features = preprocess_pipeline.transformers_[1][2]
    
    num_features = preprocess_pipeline.transformers_[0][1].named_steps['scaler'].get_feature_names_out(numeric_features)
    cat_features = preprocess_pipeline.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
    
    feature_names = np.concatenate([num_features, cat_features])
    
    print(f"Expected number of features: {len(feature_names)}")
    print(f"Shape of X_preprocessed: {X_preprocessed.shape}")

    df_transformer = DataFrameTransformer(feature_names)
    X_preprocessed = df_transformer.transform(X_preprocessed)
    
    print(f"Shape of X_preprocessed after DataFrameTransformer: {X_preprocessed.shape}")
    
    return X_preprocessed, y, preprocess_pipeline, feature_names


def preprocess_inference_data(preprocess_pipeline, X):
    X_preprocessed = preprocess_pipeline.transform(X)
    
    # Retrieve feature names after transformation
    numeric_features = preprocess_pipeline.named_steps['preprocessor'].transformers_[0][2]
    categorical_features = preprocess_pipeline.named_steps['preprocessor'].transformers_[1][2]
    num_features = preprocess_pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler'].get_feature_names_out(numeric_features)
    cat_features = preprocess_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
    feature_names = np.concatenate([num_features, cat_features, ['InteractionRate']])
    
    df_transformer = DataFrameTransformer(feature_names)
    X_preprocessed = df_transformer.transform(X_preprocessed)
    
    return X_preprocessed

if __name__ == '__main__':
    # Example of loading the dataset
    data = pd.read_csv("https://raw.githubusercontent.com/donadviser/datasets/master/data-don/online_shoppers_intention.csv")

    # Define the target variable y
    y = data['Revenue']

    # Define the features X
    X = data.drop(columns=['Revenue'])     

    # Preprocess the data
    X_preprocessed, y, preprocess_pipeline, feature_names = preprocess_training_data(X, y)