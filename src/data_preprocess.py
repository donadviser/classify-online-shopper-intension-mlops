from src.logger import logging
from src.exceptions import CustomException
import pandas as pd
import numpy as np
from typing import Tuple, Union, Optional, List
from typing_extensions import Annotated

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler



def create_prepprocessor_transformers(numeric_features: str, categorical_features: str) -> ColumnTransformer:
    # Preprocessing pipeline for numeric features
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
        #('scaler', RobustScaler())
    ])

    # Preprocessing pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ("scaler",StandardScaler(with_mean=False))
    ])

    # Combine the transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])    
    
    return preprocessor


def create_preprocessing_pipeline(numeric_features, categorical_features) -> Tuple[
            Annotated[Pipeline, "preprocess_pipeline"],
            ]:
    
    """
    Create a preprocessing pipeline and define the columns for numeric and categorical transformers.
    To avoid X.shape and y.shape inconsistencies, this step should not involved dropping of rows.
    All rows dropping should be done before before slitting the datasets into X and y.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Tuple[Pipeline, list, list]: A tuple containing the preprocessing pipeline,
                                     list of numerical columns, and list of categorical columns.
    """
    try:
        # We use the sklearn pipeline to chain together multiple preprocessing steps
        preprocess_pipeline = Pipeline([("passthrough", "passthrough")])

        preprocessor = create_prepprocessor_transformers(numeric_features, categorical_features)

        
        preprocess_pipeline.steps.append(('preprocessor', preprocessor))


        logging.info("Preprocessing pipeline created successfully.")
        return preprocess_pipeline
    except Exception as e:
        logging.error(f"Error in create_preprocessing_pipeline: {e}")
        raise CustomException(e, f"Error in create_preprocessing_pipeline")