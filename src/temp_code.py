from src.logger import logging
from src.exceptions import CustomException
import pandas as pd
import numpy as np
from typing import Tuple, Union, Optional, List
from typing_extensions import Annotated
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

from utils.preprocess import (
    OutlierTransformer,
    FeatureEngineeringTransformer,
    RenameColumnsTransformer,
    DataFrameTransformer
)

def create_preprocessor_transformers(numeric_features: str, categorical_features: str) -> ColumnTransformer:
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ("scaler", StandardScaler(with_mean=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])    

    return preprocessor

def create_preprocessing_pipeline(
        numeric_features, 
        categorical_features,
        drop_na: Optional[bool] = None,
        drop_columns: Optional[List[str]] = None,
        ) -> Tuple[
            Annotated[Pipeline, "preprocess_pipeline"],
            Annotated[List, "numeric_features"]
            ]:

    try:
        preprocess_pipeline = Pipeline([("passthrough", "passthrough")])
        if drop_na:
            preprocess_pipeline.steps.append(('drop_na', FunctionTransformer(lambda x: x.dropna(), validate=False)))

        if drop_columns:
            preprocess_pipeline.steps.append(("drop_columns", FunctionTransformer(lambda x: x.drop(columns=drop_columns), validate=False)))

        preprocess_pipeline.steps.append(('rename_columns', RenameColumnsTransformer()))
        preprocess_pipeline.steps.append(('feature_engineering', FeatureEngineeringTransformer()))

        new_features = ['interaction_rate']
        numeric_features = numeric_features + new_features

        preprocessor = create_preprocessor_transformers(numeric_features, categorical_features)
        preprocess_pipeline.steps.append(('preprocessor', preprocessor))

        logging.info("Preprocessing pipeline created successfully.")
        return preprocess_pipeline, numeric_features
    except Exception as e:
        logging.error(f"Error in create_preprocessing_pipeline: {e}")
        raise CustomException(e, f"Error in create_preprocessing_pipeline")

def preprocess_training_data(X, y, numeric_features, categorical_features, drop_na=False, drop_columns=None, test_size=0.2, random_state=None):
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    data_train = X_train_raw.copy()
    data_train['Revenue'] = y_train

    if drop_na:
        data_train.dropna(inplace=True)

    if drop_columns:
        data_train.drop(columns=drop_columns, inplace=True)

    data_train.drop_duplicates(inplace=True)
    y_train = data_train['Revenue']
    X_train = data_train.drop(columns=['Revenue'])

    preprocess_pipeline, numeric_features = create_preprocessing_pipeline(
        numeric_features, categorical_features, drop_na, drop_columns)

    X_train_preprocessed = preprocess_pipeline.fit_transform(X_train)
    X_test_preprocessed = preprocess_pipeline.transform(X_test_raw)

    numeric_transformer = preprocess_pipeline.named_steps['preprocessor'].named_transformers_['num'].named_steps['scaler']
    num_features = numeric_transformer.get_feature_names_out(numeric_features)
    cat_transformer = preprocess_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    cat_features = cat_transformer.get_feature_names_out(categorical_features)
    feature_names = np.concatenate([num_features, cat_features])

    df_transformer = DataFrameTransformer(feature_names)
    X_train_preprocessed = df_transformer.transform(X_train_preprocessed)
    X_test_preprocessed = df_transformer.transform(X_test_preprocessed)

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test, preprocess_pipeline, feature_names

# Example of loading the dataset
data = pd.read_csv('online_shoppers_intention.csv')

# Define the target variable y
y = data['Revenue']

# Define the features X
X = data.drop(columns=['Revenue'])

# Identify numeric and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Preprocess the training data and split into train and test sets
X_train_preprocessed, X_test_preprocessed, y_train, y_test, preprocess_pipeline, feature_names = preprocess_training_data(
    X, y, numeric_features, categorical_features)

# Save the preprocessed train and test data
data_train = pd.concat([X_train_preprocessed, y_train.reset_index(drop=True)], axis=1)
data_test = pd.concat([X_test_preprocessed, y_test.reset_index(drop=True)], axis=1)

data_train.to_csv('data_train_preprocessed.csv', index=False)
data_test.to_csv('data_test_preprocessed.csv', index=False)

joblib.dump(preprocess_pipeline, 'preprocess_pipeline.pkl')

# For training and evaluation steps
X_train = data_train.drop(columns=['Revenue'])
y_train = data_train['Revenue']
X_test = data_test.drop(columns=['Revenue'])
y_test = data_test['Revenue']

# Training the model (example function)
# model = train_and_save_model(X_train, y_train)  # Uncomment and define this function as needed

# Evaluate the model (example function)
# y_pred = model.predict(X_test)  # Uncomment and define this function as needed
# print("Accuracy on test data:", accuracy_score(y_test, y_pred))  # Uncomment and define this function as needed
# print(classification_report(y_test, y_pred))  # Uncomment and define this function as needed
