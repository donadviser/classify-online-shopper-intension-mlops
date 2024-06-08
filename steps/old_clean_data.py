from src.logger import logging
from src.exceptions import CustomException
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
    
    categorical_features = ["operating_systems", "browser", "region", "traffic_type", "visitor_type", 'special_day', "month", "weekend"]
    numeric_features = X_data.columns.difference(categorical_features)

    try:
        drop_na = None
        drop_columns = []#['month', 'visitor_type']


        preprocess_pipeline = create_preprocessing_pipeline(
            numeric_features, 
            categorical_features,
            drop_na=drop_na,
            drop_columns=drop_columns,
            target_col=target_col
            )
     

        # Fit and transform the data
        X_preprocessed = preprocess_pipeline.fit_transform(X)

        """num_features = preprocess_pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler'].get_feature_names_out(numeric_features)
        cat_features = preprocess_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        feature_names = np.concatenate([num_features, cat_features, []])

        print(f"Length of features names: {len(feature_names)}")
        
        df_transformer = DataFrameTransformer(feature_names)
        X_preprocessed = df_transformer.transform(X_preprocessed)"""
    
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
    df = (df
          .pipe(rename_to_snake_case)
          )

    print(df.columns)
    print(df.head())
    print(df.shape)

    target_col = "revenue"

    #Define the target variable y
    y = df[target_col]

    # Define the features X
    X = df.drop(columns=[target_col])

    # Preprocess the training data
    X_preprocessed, preprocess_pipeline = preprocess_training_data(X, target_col=target_col)

    print("\nAfter Cleaning")
    #print(X_preprocessed.head())
    print(X_preprocessed.shape)
    print(preprocess_pipeline)
    #print(X_preprocessed.describe().T)

