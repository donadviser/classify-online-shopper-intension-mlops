from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.prepare_data import prepare_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model


from src.logger import logging
from src.exceptions import CustomException


@pipeline(enable_cache=False)
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

        classifier_model = train_model(train_array, test_array)
        accuracy, precision_score, recall_score, f1_score, confusion_matrix, classification_report  = evaluate_model(classifier_model, test_array)

        logging.info("Training pipeline completed successfully")      

    except Exception as e:
        raise CustomException(e, "Error while training the model")