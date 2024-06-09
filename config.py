
class ModelNameConfig:
    model_name = "xgboost"
    do_fine_tuning = False

class ServiceNameConfig:
    project_name = "shopper_intension_project"
    MODEL_NAME = "shopper_intension"
    SERVICE_NAME = "shopper_intension_service"
    PIPELINE_STEP_NAME = "bentoml_model_deployer_step"
    PIPELINE_NAME = "training_shopper_mlops"