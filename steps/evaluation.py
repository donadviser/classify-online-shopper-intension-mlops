from src.logger import logging
from src.exceptions import CustomException
from zenml import step
from src.evaluator import Accuracy, ConfusionMatrix, PrecisionScore, RecallScore, F1Score, ClassificationReport
from sklearn.base import ClassifierMixin
import mlflow
import mlflow.gluon
import pandas as pd
import numpy as np
from typing import Tuple
from typing_extensions import Annotated
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker
 
# Enable autologging for the relevant framework
mlflow.sklearn.autolog()

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model: ClassifierMixin,
                   test_array: np.ndarray,
                   )-> Tuple[
                       Annotated[float, "accuracy"],
                       Annotated[float, "precision_score"],
                       Annotated[float, "recall_score"],
                       Annotated[float, "f1_score"],
                       Annotated[np.ndarray, "confusion_matrix"],
                       Annotated[dict, "classification_report"]
                   ]:
    
    """
    Evaluates the model

    Args:
        model (ClassifierMixin): Trained model
    
    Returns:
        accuracy (float):
    """
    try:
        X_test, y_test = (
        test_array[:,:-1],
        test_array[:,-1]
    )


        y_pred = model.predict(X_test)

        evaluation = Accuracy()
        accuracy = evaluation.calculate_scores(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        logging.info(f"Model evaluation {accuracy=}") 

        confusion_matrix = ConfusionMatrix()
        confusion_matrix = confusion_matrix.calculate_scores(y_test, y_pred)
        #mlflow.log_metric("confusion_matrix", confusion_matrix)
        logging.info(f"Model evaluation {confusion_matrix=}")

        precision_score = PrecisionScore()
        precision_score = precision_score.calculate_scores(y_test, y_pred)
        mlflow.log_metric("precision_score", precision_score)
        logging.info(f"Model evaluation {precision_score=}")

        recall_score = RecallScore()
        recall_score = recall_score.calculate_scores(y_test, y_pred)
        mlflow.log_metric("recall_score", recall_score)
        logging.info(f"Model evaluation {recall_score=}")
        logging.info(f"Model evaluation completed successfully")

        f1_score = F1Score()
        f1_score = f1_score.calculate_scores(y_test, y_pred)
        mlflow.log_metric("f1_score", f1_score)
        logging.info(f"Model evaluation {f1_score=}")
        logging.info(f"Model evaluation completed successfully")

        classification_report = ClassificationReport()
        classification_report = classification_report.calculate_scores(y_test, y_pred)
        #mlflow.log_metric("classification_report", classification_report)
        logging.info(f"Model evaluation {classification_report=}")
        logging.info(f"Model evaluation completed successfully")

        return accuracy, precision_score, recall_score, f1_score, confusion_matrix, classification_report
    except Exception as e:
        raise CustomException(e, "Error while evaluating the model")
    