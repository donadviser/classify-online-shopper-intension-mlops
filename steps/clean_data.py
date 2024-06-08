from src.logger import logging
from src.exceptions import CustomException
import pandas as pd
import numpy as np
from typing import Tuple
from typing_extensions import Annotated
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from src.data_preprocess import create_preprocessing_pipeline
from steps.split_data import data_splitter
from utils.preprocess import DataFrameTransformer
from zenml import step

#@step
def clean_df(
        X_train_clean: pd.DataFrame,
        X_test_clean: pd.DataFrame,
        y_train_clean: pd.Series,
        y_test_clean: pd.Series) -> Tuple[
                                        Annotated[pd.DataFrame, "X_train_preprocessed"],
                                        Annotated[pd.DataFrame, "X_test_preprocessed"],
                                        Annotated[pd.Series, "y_train_processed"],
                                        Annotated[pd.Series, "y_test_processed"],
]:
    

    try:
        # Identify numeric and categorical features
        
        #categorical_features = X_train_clean.select_dtypes(include=['object']).columns.tolist()
        categorical_features = ["operating_systems", "browser", "region", "traffic_type", "visitor_type", 'special_day', "month", "weekend"]
        #numeric_features = X_train_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_features = X_train_clean.columns.difference(categorical_features)
        print(f"\nnumeric_features: {numeric_features}, \ncategorical_features: {categorical_features}")

        print(X_train_clean[categorical_features])
        print(X_train_clean.info())


        preprocess_pipeline = create_preprocessing_pipeline(numeric_features, categorical_features)      
    
        # Fit and transform the data
        X_train_preprocessed = preprocess_pipeline.fit_transform(X_train_clean).toarray()
        X_test_preprocessed = preprocess_pipeline.transform(X_test_clean).toarray()



        # Retrieve feature names after transformation
        num_features = preprocess_pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler'].get_feature_names_out(numeric_features)
        cat_features = preprocess_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)

        feature_names = np.concatenate([num_features, cat_features])    

        # Cast the data to a DataFrame
        df_transformer = DataFrameTransformer(feature_names)        
        #X_train_preprocessed = df_transformer.transform(X_train_preprocessed)
        #X_test_preprocessed = df_transformer.transform(X_test_preprocessed)

        # Label encode the target variable
        label_encoder = LabelEncoder()
        y_train_processed = label_encoder.fit_transform(y_train_clean)
        y_test_processed = label_encoder.fit_transform(y_test_clean)

        print(f"{X_train_preprocessed.shape=}")
        print(f"{X_test_preprocessed.shape=}")
        print(f"{y_train_processed.shape=}")
        print(f"{y_test_processed.shape=}")

        print(f"{y_train_processed[20:41]=}")
        print(f"{y_test_processed[20:41]=}")

        train_array = np.c_[X_train_preprocessed, np.array(y_train_processed)]
        test_array = np.c_[X_test_preprocessed, np.array(y_test_processed)]

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