from logger import logging
from exceptions import CustomException
import pandas as pd
import re
import numpy as np
from typing import Tuple, Union, Optional, List
from typing_extensions import Annotated

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.data_cleaning import create_preprocessing_pipeline

from utils.preprocess import DataFrameTransformer


def preprocess_training_data(X_data: pd.DataFrame, target_col: str) -> Tuple[
    Annotated[np.ndarray, "X_preprocessed"],
    Annotated[Pipeline, "preprocess_pipeline"]
]:
    
    numeric_features = ['administrative', 'administrative_duration', 'informational', 'informational_duration',
                        'product_related', 'product_related_duration', 'bounce_rates', 'exit_rates', 'page_values', 'special_day']
    categorical_features = ['operating_systems', 'browser', 'region', 'traffic_type', 'visitor_type', 'weekend', 'month']

    """numeric_features = ['Administrative', 'Administrative_Duration', 'Informational', 'Informational_Duration',
                        'ProductRelated', 'ProductRelated_Duration', 'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay']
    
    categorical_features = ['Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend']"""

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
    
def rename_to_snake_case(df):
    """
    Renames the columns of a DataFrame to snake_case, handling camel case, acronyms, Pascal case, hyphens, and multiple spaces.
    Args:
        df: The DataFrame to rename.
    Returns:
        A new DataFrame with the columns renamed to snake_case.
    """
    return df.rename(columns=lambda s: '_'.join([word.lower() for word in re.findall(r'[A-Z][a-z]*', s.replace('-', '_').replace('  ', '_'))]))
    
if __name__ == "__main__":
    df = pd.read_csv("https://raw.githubusercontent.com/donadviser/datasets/master/data-don/online_shoppers_intention.csv")
    print("Before Cleaning")
    '''df = (df
          .pipe(rename_to_snake_case)
          )'''

    print(df.columns)
    print(df.head())
    print(df.shape)

    print(df['SpecialDay'].value_counts())

    target_col = "Revenue"

    #Define the target variable y
    y = df[target_col]

    # Define the features X
    X = df.drop(columns=[target_col])

    # Preprocess the training data
    X_preprocessed, preprocess_pipeline = preprocess_training_data(X, target_col=target_col)
   
     
    print(X_preprocessed.describe().T)

