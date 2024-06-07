from typing import Union
import pandas as pd
import numpy as np
import re

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import zscore

from logger import logging
from exceptions import CustomException


# Custom transformers
class DropMissingThreshold(BaseEstimator, TransformerMixin):
    """
    Transformer to drop columns with missing values above a certain threshold.
    """
    def __init__(self, threshold: float):
        self.threshold = threshold
    
    def fit(self, X: pd.DataFrame, y: None = None) -> 'DropMissingThreshold':
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            return X.dropna(thresh=len(X) * self.threshold, axis=1)
        except Exception as e:
            raise CustomException(e, f"Error in DropMissingThreshold")
        
    
class DateTimeConverter(BaseEstimator, TransformerMixin):
    """
    Transformer to convert specified columns to datetime.
    """
    def __init__(self, datetime_cols: list):
        self.datetime_cols = datetime_cols
        
    def fit(self, X: pd.DataFrame, y: None = None) -> 'DateTimeConverter':
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            X=(X.assign(**{col: pd.to_datetime(X[col], errors='coerce') for col in self.datetime_cols}))
            return X
        except Exception as e:
            raise CustomException(e, f"Error in DateTimeConverter")
        

    
class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to drop columns from the dataset.
    """
    def __init__(self, drop_columns: list):
        self.drop_columns = drop_columns
    
    def fit(self, X: pd.DataFrame, y: None = None) -> 'DropColumnsTransformer':
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            return X.drop(columns=self.drop_columns)
        except Exception as e:
            raise CustomException(e, f"Error in DropColumnsTransformer")
        

class OutlierTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to remove outliers from the dataset.
    """
    def __init__(self, method='zscore', threshold=3):
        self.method = method
        self.threshold = threshold
        self.outlier_indices_ = None

    def fit(self, X:pd.DataFrame, y:None=None) -> 'OutlierTransformer':
        if self.method == 'zscore':
            self.outlier_indices_ = np.abs(zscore(X)) > self.threshold
        elif self.method == 'iqr':
            q1 = np.percentile(X, 25, axis=0)
            q3 = np.percentile(X, 75, axis=0)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            self.outlier_indices_ = (X < lower_bound) | (X > upper_bound)
        else:
            raise ValueError("Invalid method. Please choose either method='zscore' or method='iqr'.")

        return self

    def transform(self, X:pd.DataFrame, y:None=None) -> pd.DataFrame:
        try:
            if self.outlier_indices_ is None:
                raise ValueError(f"The transformer must be fit before transforming data.")
            
            # Treat outliers based on the chosen method
            if self.method == 'zscore' or self.method == 'iqr':
                return np.where(self.outlier_indices_, np.nan, X)
        except Exception as e:
            raise CustomException(e, f"Error in OutlierTransformer")



class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = (X
             .drop_duplicates()
             .assign(interaction_rate = lambda x: x['page_values']/(x['product_related']+1)
                    )
            )
        return X
        

class DataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Ensure X is a numpy array and has the correct shape
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"Shape of passed values is {X.shape}, indices imply {len(self.feature_names)}")
        return pd.DataFrame(X, columns=self.feature_names)
    
    
class RenameColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to rename columns of a DataFrame to snake_case.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        return X.rename(
            columns=lambda s: '_'.join([word.lower() for word in re.findall(r'[A-Z][a-z]*', s.replace('-', '_').replace('  ', '_'))])
            )

def cast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast columns to the correct data type.
    """

    data = df.copy()

    data["datetime_utc"] = pd.to_datetime(data["datetime_utc"])
    data["area"] = data["area"].astype("string")
    data["consumer_type"] = data["consumer_type"].astype("int32")
    data["energy_consumption"] = data["energy_consumption"].astype("float64")

    return data