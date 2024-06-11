import logging
import os
import sys

# Get the path of the parent folder
parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent folder to the Python path
sys.path.append(parent_folder_path)

import pandas as pd
import numpy as np
import joblib
from src.logger import logging
from src.exceptions import CustomException
from src.data_preparation import (
    DataCleaning,
    RenameColumnsStrategy, 
    DropDuplicatesStrategy, 
    DropMissingThreshold, 
    FeatureEngineeringStrategy
)

def get_data_for_test(target_col: str='revenue'):
    try:
        logging.info("Starting to get encoded data for test")
        inference_data_path = "https://raw.githubusercontent.com/donadviser/datasets/master/data-don/online_shoppers_intention.csv"
        #inference_data_path = "/Users/don/github-projects/classify-online-shopper-intension-mlops/data/market_response_data.csv"
        inference_df = pd.read_csv(inference_data_path)
        inference_df = inference_df.sample(n=100)

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


        # Load the preprocessing pipeline and label encoder
        artefact_path = "/Users/don/github-projects/classify-online-shopper-intension-mlops/artefacts"
        preprocess_pipeline = joblib.load(os.path.join(artefact_path, 'preprocess_pipeline.pkl'))
        label_encoder = joblib.load(os.path.join(artefact_path, 'label_encoder.pkl'))

        # Preprocess the test features
        X_test_preprocessed = preprocess_pipeline.transform(X_test_clean).toarray()
        
        
        # Encode the test labels
        y_test_preprocessed = label_encoder.transform(y_test_clean)

        # Verify the shapes before concatenation
        logging.info(f"X_test_preprocessed.shape={X_test_preprocessed.shape}")
        logging.info(f"y_test_preprocessed.shape={y_test_preprocessed.shape}")

        if X_test_preprocessed.shape[0] != y_test_preprocessed.shape[0]:
            raise CustomException(
                f"Shape mismatch: X_test_preprocessed has {X_test_preprocessed.shape[0]} rows, "
                f"but y_test_preprocessed has {y_test_preprocessed.shape[0]} rows"
            )

        # Concatenate the preprocessed features and labels
        test_array = np.c_[X_test_preprocessed, np.array(y_test_preprocessed)]

        logging.info("Completed preprocessing pipeline for the test dataset")
        
        return test_array
    except Exception as e:
        raise CustomException(e, "Failed to load preprocessing pipeline to clean the inference dataset")
    

"""if __name__ == "__main__":
    X_test_preprocessed, y_test_preprocessed = get_data_for_test()
    print(X_test_preprocessed.shape)
    print(y_test_preprocessed[:20])"""