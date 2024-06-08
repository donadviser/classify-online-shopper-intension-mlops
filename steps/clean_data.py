from src.logger import logging
from src.exceptions import CustomException
import pandas as pd
import numpy as np
from typing import Tuple
from typing_extensions import Annotated
import joblib

from sklearn.pipeline import Pipeline
from src.data_cleaning import create_preprocessing_pipeline
from steps.split_data import data_splitter
from utils.preprocess import DataFrameTransformer
from zenml import step

#@step
def clean_df(data_df: pd.DataFrame, target_col: str) -> Tuple[
    Annotated[pd.DataFrame, "X_train_preprocessed"],
    Annotated[pd.DataFrame, "X_test_preprocessed"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
    Annotated[Pipeline, "preprocess_pipeline"]
]:
    
    numeric_features = ['administrative', 'administrative_duration', 'informational', 'informational_duration',
                        'product_related', 'product_related_duration', 'bounce_rates', 'exit_rates', 'page_values', 'special_day']
    categorical_features = ['operating_systems', 'browser', 'region', 'traffic_type', 'visitor_type', 'weekend', 'month']

    
    try:
        y = data_df[target_col]

        # Define the features X
        X = data_df.drop(columns=[target_col])

        X_train, X_test, y_train, y_test = data_splitter(X, y, test_size = 0.2, random_state = 42, stratify=y)
        #Define the target variable y
        
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        print(X_train.head())
        print(y_train.head())

        drop_na = None
        drop_columns = []#['month', 'visitor_type']


        preprocess_pipeline, numeric_features = create_preprocessing_pipeline(
            numeric_features, 
            categorical_features,
            drop_na=drop_na,
            drop_columns=drop_columns,
            )      
    
        # Fit and transform the data
        X_train_preprocessed = preprocess_pipeline.fit_transform(X_train).toarray()
        X_test_preprocessed = preprocess_pipeline.transform(X_test).toarray()

        # Retrieve feature names after transformation
        num_features = preprocess_pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler'].get_feature_names_out(numeric_features)
        cat_features = preprocess_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)

        feature_names = np.concatenate([num_features, cat_features])    

        # Cast the data to a DataFrame
        df_transformer = DataFrameTransformer(feature_names)        
        #X_train_preprocessed = df_transformer.transform(X_train_preprocessed)
        #X_test_preprocessed = df_transformer.transform(X_test_preprocessed)

        print(f"{X_train_preprocessed.shape=}")
        print(f"{X_test_preprocessed.shape=}")
        print(f"{y_train.shape=}")
        print(f"{y_test.shape=}")

        train_array = np.c_[X_train_preprocessed, np.array(y_train)]
        test_array = np.c_[X_test_preprocessed, np.array(y_test)]

        logging.info(f"Saving the preprocessing objects.")

        # Save artefacts
        joblib.dump(preprocess_pipeline, 'artefacts/preprocess_pipeline.pkl')
        joblib.dump(train_array, 'artefacts/data_train_preprocessed.pkl')
        joblib.dump(test_array, 'artefacts/data_test_preprocessed.pkl')
        logging.info("Preprocessed artefacts saved")

        #print(X_train_preprocessed.describe().T)
        
        return train_array, test_array

    except Exception as e:
        raise CustomException(e, f"Error in clean_df")