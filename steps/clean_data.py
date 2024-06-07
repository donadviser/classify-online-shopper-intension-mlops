from logger import logging
from exceptions import CustomException
import pandas as pd
import numpy as np
from typing import Tuple
from typing_extensions import Annotated

from sklearn.pipeline import Pipeline
from src.data_cleaning import create_preprocessing_pipeline
from utils.preprocess import DataFrameTransformer


def clean_df(X_data: pd.DataFrame, target_col: str) -> Tuple[
    Annotated[np.ndarray, "X_preprocessed"],
    Annotated[Pipeline, "preprocess_pipeline"]
]:
    
    numeric_features = ['administrative', 'administrative_duration', 'informational', 'informational_duration',
                        'product_related', 'product_related_duration', 'bounce_rates', 'exit_rates', 'page_values', 'special_day']
    categorical_features = ['operating_systems', 'browser', 'region', 'traffic_type', 'visitor_type', 'weekend', 'month']

    
    try:
        drop_na = None
        drop_columns = []#['month', 'visitor_type']


        preprocess_pipeline, numeric_features = create_preprocessing_pipeline(
            numeric_features, 
            categorical_features,
            drop_na=drop_na,
            drop_columns=drop_columns,
            )      
    
        # Fit and transform the data
        X_preprocessed = preprocess_pipeline.fit_transform(X_data).toarray()

        # Retrieve feature names after transformation
        num_features = preprocess_pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler'].get_feature_names_out(numeric_features)
        cat_features = preprocess_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)

        feature_names = np.concatenate([num_features, cat_features])    

        # Cast the data to a DataFrame
        df_transformer = DataFrameTransformer(feature_names)        
        X_preprocessed = df_transformer.transform(X_preprocessed)
        
        return X_preprocessed, preprocess_pipeline

    except Exception as e:
        raise CustomException(e, f"Error in clean_df")