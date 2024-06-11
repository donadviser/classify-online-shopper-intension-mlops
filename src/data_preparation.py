import pandas as pd
import numpy as np
import re
from scipy.stats import zscore
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exceptions import CustomException
from abc import ABC, abstractmethod
from typing import Tuple, Union
from typing_extensions import Annotated
from feature_engine.encoding import MeanEncoder


class DataStrategy(ABC):
    """
    Abstract base defining strategy for handling data.
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, str]:
        pass

class RenameColumnsStrategy(DataStrategy):
    """
    Strategyto rename columns of a DataFrame to snake_case.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, str]:
        try:
            data = data.copy()
            return data.rename(
                columns=lambda s: '_'.join([word.lower() for word in re.findall(r'[A-Z][a-z]*', s.replace('-', '_').replace('  ', '_'))])
                )
        except Exception as e:
            raise CustomException(e, "Error renaming columns in RenameColumnsStrategy")
        
class OutlierHandlingStrategy(DataStrategy):
    """
    Strategy to handle outliers in a DataFrame.
    """
    def __init__(self, method='zscore', threshold=3):
        self.method = method
        self.threshold = threshold
        self.outlier_indices_ = None

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            if self.method == 'zscore':
                self.outlier_indices_ = np.abs(zscore(data)) > self.threshold
            elif self.method == 'iqr':
                q1 = np.percentile(data, 25, axis=0)
                q3 = np.percentile(data, 75, axis=0)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                self.outlier_indices_ = (data < lower_bound) | (data > upper_bound)
            else:
                raise ValueError("Invalid method. Please choose either method='zscore' or method='iqr'.")
            
            # Treat outliers by replacing them with NaN
            return data.mask(self.outlier_indices_)
        except Exception as e:
            raise CustomException(e, "Error handling outliers in OutlierHandlingStrategy")

class DataDateTimeConverter(DataStrategy):
    """
    Strategy for converting datetime columns.
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, str]:
        """
        Converts datetime columns to datetime type.

        Args:
            data (pd.DataFrame): Data to be converted.

        Returns:
            Union[pd.DataFrame, pd.Series]: Converted data.
        """
        datetime_cols = ['order_purchase_timestamp', 'order_approved_at', 
                         'order_delivered_carrier_date', 'order_delivered_customer_date', 
                         'order_estimated_delivery_date', 'shipping_limit_date']
    
        
        try:
            data = data.assign(**{col: pd.to_datetime(data[col], errors='coerce') for col in datetime_cols})
            logging.info(f"Converting datetime columns completed successfully")
            return data
        except Exception as e:
            raise CustomException(e, "Error converting datetime columns")



class FeatureEngineeringStrategy(DataStrategy):
    """
    Strategy for preprocessing data.
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, str]:
        """
        Preprocess data.

        Args:
            data (pd.DataFrame): Data to be preprocessed.

        Returns:
            Union[pd.DataFrame, pd.Series]: Preprocessed data.
        """
        try:
            data = data.copy()
            data = (data
                    .assign(interaction_rate = lambda x: x['page_values']/(x['product_related']+1)
                    )
                    .dropna()
                   )
            print(f"DataPreProcessStrategy completed successfully")
            return data
        except Exception as e:
            raise CustomException(e, "Error during data preparation in FeatureEngineeringStrategy")


class DropMissingThreshold(DataStrategy):
    """
    Strategy for dropping columns with missing threshold.
    """
    def handle_data(self, data: pd.DataFrame, threshold = 0.5) -> Union[pd.DataFrame, pd.Series, str]:
        try:
            data = data.copy()
            data = data.dropna(thresh=len(data) * threshold, axis=1)
            logging.info(f"Dropping Missing Threshod of {threshold} completed successfully")
            return data
        except Exception as e:
            raise CustomException(e, f"Error in DropMissingThreshold")
        
class DropDuplicatesStrategy(DataStrategy):
    """
    Strategy for dropping columns with missing threshold.
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, str]:
        try:
            data = data.copy()
            data = data.drop_duplicates()
            logging.info(f"Dropping Duplicate rows completed successfully")
            return data
        except Exception as e:
            raise CustomException(e, f"Error in DropMissingThreshold")
        

class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data.

    """
    def __init__(self, target_col:str):
        self.target_col = target_col
        
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, str]:
        """
        Divide data.cl

        Args:
            data (pd.DataFrame): Data to be divided.

        Returns:
            Union[pd.DataFrame, pd.Series]: Divided data.
        """
        try:
            print(f"DataDivideStrategy started")
            X = data.drop(columns=self.target_col)
            y = data[self.target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            print(f"DataDivideStrategy completed successfully")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, "Error while dividing data")


class DataEncodeStrategy(DataStrategy):
    """
    Strategy for encoding data.
    """
    def handle_data(self, data: Tuple[pd.DataFrame, pd.DataFrame], target_col: str) -> Annotated[str, "preprocessor"]:
        """
        Encode data.

        Args:
            data (Tuple[pd.DataFrame, pd.DataFrame]): Tuple of training data

        Returns:
                Annotated[ColumnTransformer, "preprocessor"]: Encoded training and testing data, and the preprocessor.
        """
        try:
            logging.info("Data encoded started.")
            X_train = data
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('mean_encoder', MeanEncoder())
            ])

            numerical_cols = X_train.select_dtypes(include=[np.number]).columns.difference([target_col]).tolist()
            categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

            print(f"numerical_cols: {numerical_cols}")
            print(f"categorical_cols: {categorical_cols}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols),
                ])
            

            logging.info("Data encoded successfully.")
            return preprocessor
        except Exception as e:
            raise CustomException(e, "Error in DataEncodeStrategy")
        

class DataCleaning:
    """
    Class for cleaning data: Clean, Preprocess and Divide the data.
    """
    def __init__(self, data: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]], strategy: DataStrategy, target_col: str = None):
        self.data = data
        self.strategy = strategy
        self.target_col = target_col

    def handle_data(self) -> Union[pd.DataFrame, pd.Series, Tuple]:
        """
        Clean, Preprocess and Divide the data.

        Returns:
            Union[pd.DataFrame, pd.Series, Tuple]: Clean, Preprocessed and Divided data.
        """
        try:
            if self.target_col:
                return self.strategy.handle_data(self.data, self.target_col)
            return self.strategy.handle_data(self.data)
        except Exception as e:
            raise CustomException(e, "Error while cleaning, preprocessing and dividing data")