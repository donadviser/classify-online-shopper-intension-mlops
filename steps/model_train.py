import sys
from typing import Annotated

import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin

from zenml import step, ArtifactConfig
from src.exceptions import CustomException
from src.logger import logging
from src.model_dev import ModelTrainer
from steps.config import ModelNameConfig
import mlflow
import mlflow.gluon


from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker

# Enable autologging for the relevant framework
#mlflow.sklearn.autolog()

#@step(enable_cache=False, experiment_tracker=experiment_tracker.name)
@step()
def train_model(
        train_array: np.ndarray,  
        ) -> Annotated[ClassifierMixin, ArtifactConfig(name="classifier_model", is_model_artifact=True)]:
    """
    Trains the model

    Args:
        X_train (pd.DataFrame): Training data
        y_train (pd.DataFrame): Training labels
        model_type (str, optional): Model type. Defaults to 'randomforest'.
        do_fine_tuning (bool, optional): Fine tuning. Defaults to True.

    Returns:
        Annotated[ClassifierMixin, "sklearn_regressor_model"]: Trained model
    """
    logging.info("Split training and test input data")
    X_train,y_train,X_test,y_test=(
        train_array[:,:-1],
        train_array[:,-1]
    )


    model_config = ModelNameConfig()
    logging.info(f"Model name: {model_config.model_name}")
    try:
        model_name = model_config.model_name
        do_fine_tuning= model_config.do_fine_tuning

        model_training  = ModelTrainer(X_train, y_train, X_test, y_test)
        if model_name == "randomforest":
            mlflow.sklearn.autolog()
            logging.info("Training random forest model")
            rf_model = model_training.random_forest_trainer(fine_tuning=do_fine_tuning)
            return rf_model
        
        elif model_name == "lightgbm":
            mlflow.sklearn.autolog()
            logging.info("Training lightgbm model")
            lgm_model = model_training.lightgbm_trainer(
                fine_tuning=do_fine_tuning
            )
            return lgm_model
        
        elif model_name == "xgboost":
            mlflow.sklearn.autolog()
            logging.info("Training xgboost model")
            xgb_model = model_training.xgboost_trainer(
                fine_tuning=do_fine_tuning
            )
            return xgb_model
        else:
            raise ValueError('Model type not supported')

    except Exception as e:
        CustomException(e, "Error in train_model")