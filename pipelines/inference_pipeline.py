import pandas as pd
import numpy as np
import joblib
import os
from typing import Tuple
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from src.data_preparation import (
    DataCleaning,
    RenameColumnsStrategy, 
    DropDuplicatesStrategy, 
    DropMissingThreshold, 
    FeatureEngineeringStrategy
)
 


from src.logger import logging
from src.exceptions import CustomException

def load_preprocessor(artefact_path: str = 'preprocessor.joblib') -> Tuple[Pipeline, LabelEncoder]:
    """
    Load the pre-trained preprocessor from a specified file path.

    Args:
        artefact_path (str): The file path to load the preprocessor from.

    Returns:
        ColumnTransformer: The loaded preprocessor.

    """
    try:
        preprocess_pipeline = joblib.load(os.path.join(artefact_path, 'preprocess_pipeline.pkl'))
        label_encoder = joblib.load(os.path.join(artefact_path, 'label_encoder.pkl'))
        logging.info(f'Preprocessor loaded from {artefact_path}')

        return preprocess_pipeline, label_encoder
    except Exception as e:
        raise CustomException(e, f"Error in load_preprocessor")

def preprocess_new_data(
        inference_df: pd.DataFrame, 
        preprocess_pipeline: Pipeline, 
        label_encoder: LabelEncoder, 
        target_col:str='revenue'
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess new data using the loaded preprocessor and custom transformations.

    Args:
        inference_df (pd.DataFrame): The new data to preprocess.
        preprocess_pipeline (Pipeline): The pre-trained preprocessor for X.
        label_encoder (LabelEncoder): The pre-trained preprocessor for y.
        target_col (str): The name of the target column.

    Returns:
        np.ndarray: The preprocessed X data.
        np.ndarray: The preprocessed y data.
    """
    try:
        # Obtain the preprocessing pipeline
        #Stage 2 Cleaning Step: Preprocessing step
        logging.info("Start the cleaning part: data preparation and data processing")
        data_prepared = DataCleaning(inference_df, RenameColumnsStrategy())
        data_prepared = data_prepared.handle_data()
        

        data_prepared = DataCleaning(data_prepared, DropDuplicatesStrategy())
        data_prepared = data_prepared.handle_data()
        
        data_prepared = DataCleaning(data_prepared, DropMissingThreshold())
        data_prepared = data_prepared.handle_data()
        
        data_prepared = DataCleaning(data_prepared, FeatureEngineeringStrategy())
        data_prepared = data_prepared.handle_data()

        X_test_clean = data_prepared.drop(columns=target_col)
        y_test_clean = data_prepared[target_col]
        
        # Apply column-specific transformations
        X_test_preprocessed = preprocess_pipeline.transform(X_test_clean) 
        y_test_preprocessed = label_encoder.transform(y_test_clean) 

        logging.info("New data preprocessed successfully.")
        return X_test_preprocessed, y_test_preprocessed
    except Exception as e:
        raise CustomException(e, f"Error in preprocess_new_data")