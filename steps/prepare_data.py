from src.logger import logging
from src.exceptions import CustomException
from sklearn.compose import ColumnTransformer
from src.data_preparation import (
    DataCleaning,
    RenameColumnsStrategy,
    DropMissingThreshold,
    DropDuplicatesStrategy,
    FeatureEngineeringStrategy,
    DataDivideStrategy,
    DataEncodeStrategy
    )
import pandas as pd
import numpy as np
import joblib
from typing import Tuple
from typing_extensions import Annotated
from zenml import step


@step
def prepare_df(data_df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train_clean"],
    Annotated[pd.DataFrame, "X_test_clean"],
    Annotated[pd.Series, "y_train_clean"],
    Annotated[pd.Series, "y_test_clean"],
]:
    
    try:
        print(f"{data_df.columns=}")
        data_prepared = DataCleaning(data_df, RenameColumnsStrategy())
        data_prepared = data_prepared.handle_data()
        
        print(f"{data_prepared.columns=}")
        data_prepared = DataCleaning(data_prepared, DropDuplicatesStrategy())
        data_prepared = data_prepared.handle_data()
        
        data_prepared = DataCleaning(data_prepared, DropMissingThreshold())
        data_prepared = data_prepared.handle_data()
        
        data_prepared = DataCleaning(data_prepared, FeatureEngineeringStrategy())
        data_prepared = data_prepared.handle_data()
        
        
        data_divider = DataCleaning(data_prepared, DataDivideStrategy(target_col='revenue'))
        X_train_clean, X_test_clean, y_train_clean, y_test_clean = data_divider.handle_data()
        

        joblib.dump(X_train_clean, 'artefacts/X_train_clean.joblib')
        joblib.dump(X_test_clean, 'artefacts/X_test_clean.joblib')
        joblib.dump(y_train_clean, 'artefacts/y_train_clean.joblib')
        joblib.dump(y_test_clean, 'artefacts/y_test_clean.joblib')

        return X_train_clean, X_test_clean, y_train_clean, y_test_clean
    except Exception as e:
        raise CustomException(e, f"Error in clean_df")