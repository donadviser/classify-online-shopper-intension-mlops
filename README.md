# Predict Online Shoppers Purchasing Behaviour with Deployment
## (Predicting Responses to Marketing Campaigns)

## Introduction
This study aims to classify online shoppers' purchasing behaviour using the Online Shoppers Intention __[Kaggle](https://www.kaggle.com/datasets/henrysue/online-shoppers-intention)__ dataset. This dataset consists of 18 features, including 10 numerical and 8 categorical attributes. The primary goal is to predict whether a customer made a purchase (purchase or no purchase) based on their browsing behavior and website interaction characteristics.

## Dataset

The dataset consists of feature vectors belonging to 12,330 sessions. The dataset was formed so that each session would belong to a different user in a 1-year period to avoid any tendency to a specific campaign, special day, user profile, or period.

## Attribute

- **Revenue:** class whether it can make a revenue or not
- **Administrative, Administrative Duration, Informational, Informational Duration, Product Related, and Product Related Duration:** represent the number of different types of pages visited by the visitor in that session and total time spent in each of these page categories.
- **Bounce Rate:** percentage of visitors who enter the site from that page and then leave (“bounce”) without triggering any other requests to the analytics server during that session
- **Exit Rate:** the percentage that were the last in the session
- **Page Value:** feature represents the average value for a web page that a user visited before completing an e-commerce transaction
- **Special Day:** indicates the closeness of the site visiting time to a specific special day (e.g. Mother’s Day, Valentine’s Day) in which the sessions are more likely to be finalized with transaction. For example, for Valentine’s day, this value takes a nonzero value between February 2 and February 12, zero,before and after this date unless it is close to another special day, and its maximum value of 1 on February 8
- **Operating system, browser, region, traffic type:** Different types of operating systems, browser, region and traffic type used to visit the website
- **Visitor type:** Whether the customer is a returning or new visitor
- **Weekend:** A Boolean value indicating whether the date of the visit is weekend
- **Month:** Month of the year


## Steps
To create a modular, production-ready machine learning code that includes custom transformers and encoding for both training and inference pipelines, we'll follow these steps:

1. **Data Cleaning and Preprocessing:** Handle missing values, encode categorical features, and scale numerical features.
2. **Feature Engineering:** Create custom features if necessary.
3. **Pipeline Construction:** Construct pipelines for training and inference.
4. **Model Training and Evaluation:** Train the model and evaluate its performance.
5. **Saving and Loading Pipelines and Models:** Save the preprocessing pipeline and the trained model.
6. **Model Deployment:** Deploy the model to the cloud
7. **Inference:** Load the pipeline and model to make predictions on new data.


## Modelling
1. Random Forest  
2. XGBoost  
3. Light GBM


In order to achieve this in a real-world scenario, we will be using [ZenML](https://zenml.io/) to
build a production-ready pipeline to predict the customer intention for the next order or purchase.


## Install Packages
- create a virtual environment: I use pyenv
  `pyenv virtualenv 3.11 venv311_zenml`
  `pyenv local venv311_zenml`
  
`pip install "zenml["server"]"`

`zenml up`

To integrate zenml with mflow, run the following command
`zenml integration install mlflow -y`

The project can only be executed with a ZenML stack that has an MLflow
experiment tracker and model deployer as a component. Configuring a new stack
with the two components are as follows:

```bash
zenml integration install mlflow -y
zenml experiment-tracker register mlflow_tracker --flavor=mlflow
zenml model-deployer register mlflow --flavor=mlflow
zenml stack register local-mlflow-stack -a default -o default -d mlflow -e mlflow_tracker --set
```

This should give you the following stack to work with. 

![mlflow_stack](_assets/mlflow_stack.png)


Add this line before callint the training pipeline
`print(Client().active_stack.experiment_tracker.get_tracking_uri())`

use the output to as parameter below:
`mlflow ui --backend-store-uri "file:/Users/don/Library/Application Support/zenml/local_stores/afe9b1d8-06da-4d66-bcd1-d75224ab95f4/mlruns"`

for large file for macOS
`brew install git-lfs` 
after installing, you need to initialize the Git LFS for your repository
`git lfs install`


__References:__

* https://medium.com/analytics-vidhya/marketing-analytics-significance-of-feature-engineering-model-selection-and-hyper-parameter-53d34b57bc55
* https://github.com/Charlotte-1987/openclassroomP7/blob/main/P7_MODELLING_STREAMLIT_VF.ipynb