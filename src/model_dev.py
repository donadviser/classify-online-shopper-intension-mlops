import sys

import numpy as np
import pandas as pd
import optuna

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.base import ClassifierMixin

from src.exceptions import CustomException
from src.logger import logging

class HyperparameterOptimisation:
    """Hyperparameter optimisation class which optimises the hyperparameters of the model
    Args:
        X_train (pd.DataFrame): Training data
        y_train (pd.DataFrame): Training labels
        model_type (str, optional): Model type. Defaults to 'RandomForestClassifier'.
    """

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test        
        self.y_test = y_test

    def optimise_randomforest(self, trial: optuna.Trial) -> float:
        """
        Optimises the hyperparameters of the model: Random Forest parameters

        Args:
            trial (optinua.Trial): Trial object

        Returns:
            Accurancy (str): metric depending on the problem type
        """
         
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
        )
        
        model.fit(self.X_train, self.y_train)
        val_accuracy = model.score(self.X_test, self.y_test)
        
        return val_accuracy
    
    def optimise_lightgbm(self, trial: optuna.Trial) -> float:
        """
        Method for Optimizing LightGBM.

        """
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 20)
        num_leaves = trial.suggest_int("num_leaves", 15, 50),
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-3, 1e-1)
        model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            #num_leaves=num_leaves,
        )
        model.fit(self.X_train, self.y_train)
        val_accuracy = model.score(self.X_test, self.y_test)
        return val_accuracy
    
    def optimize_xgboost_classifier(self, trial: optuna.Trial) -> float:
        """
        Optimises the hyperparameters of the model: Xgboost parameters

        Args:
            trial (optinua.Trial): Trial object

        Returns:
            Accurancy (str): metric like R-squared or mean squared error (depending on the problem type)
        """
        
        param = {
            'verbosity': 0,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 1, 200)
        }

        model = XGBClassifier(**param)
        model.fit(self.X_train, self.y_train)
        val_accuracy = model.score(self.X_test, self.y_test)
        logging.info("val_accuracy = {}".format(val_accuracy))
        return val_accuracy

class ModelTrainer:
    """Model trainer class which trains the model
    Args:
        X_train (pd.DataFrame): Training data
        y_train (pd.DataFrame): Training labels
        model_type (str, optional): Model type. Defaults to 'lightgbm'.
        do_fine_tuning (bool, optional): Fine tuning. Defaults to True.

    """

    def __init__(self, X_train: np.ndarray,  y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    
    def random_forest_trainer(self, fine_tuning: bool = True) -> ClassifierMixin:
        """
        Fucntion that trains the RandomForest model

        Args:
            fine_tuning (bool, optional): Fine tuning. Defaults to True. If True, hyperparameter optimisation is performed
        """
        logging.info("Starting training Random Forest model")
        try:
            if fine_tuning:
                hyper_opt = HyperparameterOptimisation(
                    self.X_train,
                    self.y_train,
                    self.X_test,
                    self.y_test,
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimise_randomforest, n_trials=20)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                max_depth = trial.params["max_depth"]
                min_samples_split = trial.params["min_samples_split"]
                logging.info(f"Random Forest Best Parameters: {trial.params}")
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                    )
                model.fit(self.X_train, self.y_train)
                                 
                logging.info("Training Random Forest model with hyperparameter tunning completed successfully")
                return model
            else:
                model = RandomForestClassifier(
                    n_estimators=185,
                    max_depth=20,
                    min_samples_split=2,
                    random_state=42)
                model.fit(self.X_train, self.y_train)
                logging.info("Training Random Forest model completed successfully")
                return model                

        except Exception as e:
            logging.info("Error training Random Forest model")
            CustomException(e, "Error in model_dev while training Random Forest")
            return None
        

    def lightgbm_trainer(self, fine_tuning: bool = True) -> ClassifierMixin:
        """
        It trains the LightGBM model.

        Args:
            fine_tuning: If True, hyperparameter optimization is performed. If False, the default
            parameters are used, Defaults to True (optional).
        """

        logging.info("Started training LightGBM model.")
        try:
            if fine_tuning:
                hyper_opt = HyperparameterOptimisation(
                    self.X_train, self.y_train, self.X_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hyper_opt.optimise_lightgbm, n_trials=20)
                trial = study.best_trial
                n_estimators = trial.params["n_estimators"]
                #max_depth = trial.params["max_depth"]
                num_leaves = trial.params["num_leaves"]
                learning_rate = trial.params["learning_rate"]
                model = LGBMClassifier(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    #max_depth=max_depth,
                    num_leaves=num_leaves,
                )
                model.fit(self.X_train, self.y_train)
                return model
            else:
                model = LGBMClassifier(
                    n_estimators=200, learning_rate=0.01, max_depth=20
                )
                model.fit(self.X_train, self.y_train)
                logging.info("Training LightGBM model with hyperparameter tunning completed successfully")
                return model
        except Exception as e:
            logging.info("Error training LGBM model with hyperparameter tunning")
            CustomException(e, "Error in model_dev while training LGBM model")
            return None
            
        
    def xgboost_trainer(self, fine_tuning: bool = True) -> ClassifierMixin:
        """
        It trains the xgboost model.

        Args:
            fine_tuning: If True, hyperparameter optimization is performed. If False, the default
            parameters are used, Defaults to True (optional).
        """

        logging.info("Started training XGBoost model.")
        try:
            if fine_tuning:
                hy_opt = HyperparameterOptimisation(
                    self.X_train, self.y_train, self.X_test, self.y_test
                )
                study = optuna.create_study(direction="maximize")
                study.optimize(hy_opt.optimize_xgboost_classifier, n_trials=20)
                # Print the best hyperparameters
                if study.trials:
                    best_params = study.best_params
                    print('Best hyperparameters: ', best_params)
                    print('Best accuracy: ', study.best_value)
                # Train final model with best hyperparameters
                
                model = XGBClassifier(**best_params)
                model.fit(self.X_train, self.y_train)
                logging.info("Training Xgboost with hyperparameter tunning completed successfully")
                return model

            else:
                params = {
                    'booster': 'dart', 
                    'learning_rate': 0.016130258092904054, 
                    'max_depth': 8, 
                    'min_child_weight': 8, 
                    'subsample': 0.7405184982356865, 
                    'colsample_bytree': 0.6286302376190893, 
                    'n_estimators': 183,
                    'random_state': 42
                    }
                model = XGBClassifier(**params)
                model.fit(self.X_train, self.y_train)
                logging.info("Training Xgboost completed successfully")
                return model
        except Exception as e:
            logging.info("Error training Xgboost")
            CustomException(e, "Error in model_dev while training Xgboost")
            return None




        
