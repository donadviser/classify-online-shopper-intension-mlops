from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.prepare_data import prepare_df
from steps.clean_data import clean_df
from steps.split_data import data_splitter

from src.logger import logging
from src.exceptions import CustomException
import pandas as pd
import joblib


#@pipeline(enable_cache=False)
def train_pipeline(data_path: str)->None:
    """A pipeline to train a model.

    Args:
        data_path (str): The path to the data file.
    """
    try:
        logging.info("Starting Data Collection Step")
        data_df = ingest_df(data_path)
        logging.info("Data Collection Step completed successfully")
        X_train_clean, X_test_clean, y_train_clean, y_test_clean = prepare_df(data_df)
        logging.info("Starting Data Cleaning Steps [Data Preparation step]")
        train_array, test_array = clean_df(X_train_clean, X_test_clean, y_train_clean, y_test_clean)
        logging.info("Data Cleaning Step [Data Preprocessing Step] completed successfully")

        """classifier_model = train_model(X_train_encoded, y_train, X_test_encoded, y_test)
        accuracy, precision_score, recall_score, f1_score, confusion_matrix, classification_report  = evaluate_model(classifier_model, X_test_encoded, y_test)

        logging.info(f"accuracy: {accuracy}")
        logging.info(f"precision_score: {precision_score}")
        logging.info(f"recall_score: {recall_score}")
        logging.info(f"f1_score: {f1_score}")
        logging.info(f"confusion_matrix: {confusion_matrix}")
        logging.info(f"classification_report: {classification_report}")"""
        

        logging.info("Training pipeline completed successfully")      

    except Exception as e:
        raise CustomException(e, "Error while training the model")