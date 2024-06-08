MODEL_NAME = "shopper_intension"
SERVICE_NAME = "shopper_intension_service"
PIPELINE_STEP_NAME = "bentoml_model_deployer_step"
PIPELINE_NAME = "training_shopper"
DataPreparationInput = Union[pd.DataFrame, pd.Series, np.ndarray]
DataPreparationOutput = Union[pd.DataFrame, pd.Series, np.ndarray]