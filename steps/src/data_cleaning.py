from logger import logging
from exceptions import CustomException
import pandas as pd
import numpy as np
from typing import Tuple, Union, Optional, List
from typing_extensions import Annotated

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler


from utils.preprocess import (
    OutlierTransformer,
    FeatureEngineeringTransformer,
    RenameColumnsTransformer
    )

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
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine the transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])    
    
    return preprocessor


def create_preprocessing_pipeline(
        numeric_features, 
        categorical_features,
        drop_na: Optional[bool] = None,
        drop_columns: Optional[List[str]] = None,
        ) -> Tuple[
            Annotated[Pipeline, "preprocess_pipeline"],
            Annotated[List, "numeric_features"]
            ]:
    
    """
    Create a preprocessing pipeline and define the columns for numeric and categorical transformers.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Tuple[Pipeline, list, list]: A tuple containing the preprocessing pipeline,
                                     list of numerical columns, and list of categorical columns.
    """
    try:
        # We use the sklearn pipeline to chain together multiple preprocessing steps
        preprocess_pipeline = Pipeline([("passthrough", "passthrough")])
        if drop_na:
            preprocess_pipeline.steps.append(('drop_na', FunctionTransformer(lambda x: x.dropna(), validate=False)))

        if drop_columns:
            preprocess_pipeline.steps.append(("drop_columns", FunctionTransformer(lambda x: x.drop(columns=drop_columns), validate=False)))

        preprocess_pipeline.steps.append(('rename_columns', RenameColumnsTransformer()))

        preprocess_pipeline.steps.append(('feature_engineering', FeatureEngineeringTransformer()))

        new_features = ['interaction_rate']
        numeric_features = numeric_features+new_features

        preprocessor = create_prepprocessor_transformers(numeric_features, categorical_features)

        
        preprocess_pipeline.steps.append(('preprocessor', preprocessor))


        logging.info("Preprocessing pipeline created successfully.")
        return preprocess_pipeline, numeric_features
    except Exception as e:
        logging.error(f"Error in create_preprocessing_pipeline: {e}")
        raise CustomException(e, f"Error in create_preprocessing_pipeline")