from src.logger import logging
from src.exceptions import CustomException
from abc import ABC, abstractmethod
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    precision_score, 
    recall_score,
    f1_score,
    classification_report
    )
import numpy as np

class Evaluation(ABC):
    """
    Abstract base defingin strategy for handling data
    """
    @abstractmethod
    def calculate_scores(self, y_test: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the scores

        Args:
            y_test (np.ndarray): Test data
            y_pred (np.ndarray): Predicted data

        Returns:
            None
        pass
        """

class Accuracy(Evaluation):
    """
    Strategy for evaluating mean squared error
    """

    def calculate_scores(self, y_test: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating Accuracy")
            logging.info(f"y_pred: {y_pred}")
            accuracy = accuracy_score(y_test, y_pred)
            logging.info(f"accuracy: {accuracy}")
            return accuracy
        except Exception as e:
            raise CustomException(e, "Error while calculating accuracy")
        
class ConfusionMatrix(Evaluation):
    """
    Strategy for evaluating the confusion matrix
    """
    def calculate_scores(self, y_test: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        try:
            logging.info("Calculating Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            logging.info(f"confusion_matrix: {cm}")
            return cm
        except Exception as e:
            raise CustomException(e, "Error while calculating confusion matrix")
        
class PrecisionScore(Evaluation):
    """
    Strategy for evaluating the precision score
    """

    def calculate_scores(self, y_test: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating Precision Score")
            precision = precision_score(y_test, y_pred)
            logging.info(f"precision: {precision}")
            return precision
        except Exception as e:
            raise CustomException(e, "Error while calculating precision score")
        

class RecallScore(Evaluation):
    """
    Strategy for evaluating the recall score
    """

    def calculate_scores(self, y_test: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating Recall Score")
            recall = recall_score(y_test, y_pred)
            logging.info(f"recall: {recall}")
            return recall
        except Exception as e:
            raise CustomException(e, "Error while calculating recall score")
        

class F1Score(Evaluation):
    """
    Strategy for evaluating the f1 score
    """

    def calculate_scores(self, y_test: np.ndarray, y_pred: np.ndarray) -> float:
        try:
            logging.info("Calculating F1 Score")
            f1 = f1_score(y_test, y_pred)
            logging.info(f"f1: {f1}")
            return f1
        except Exception as e:
            raise CustomException(e, "Error while calculating f1 score")
        
        
class ClassificationReport(Evaluation):
    """
    Strategy for evaluating the classification report
    """
    
    def calculate_scores(self,y_true: np.ndarray, y_pred: np.ndarray)-> dict:
        try:
            logging.info("Calculate Classification Report")
            report = classification_report(y_true, y_pred, output_dict=True)
            logging.info(f"Classification Report:\n{report}")
            return report
        except Exception as e:
            raise CustomException(e, "Error while calculating classification report")        