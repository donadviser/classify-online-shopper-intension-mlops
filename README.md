# Predict Online Shoppers Purchasing Behaviour with Deployment

## Introduction
This study aims to classify online shoppers' purchasing behavior using the Online Shoppers Intention __[Kaggle](https://www.kaggle.com/datasets/henrysue/online-shoppers-intention)__ dataset. This dataset consists of 18 features, including 10 numerical and 8 categorical attributes. The primary goal is to predict whether a customer made a purchase (purchase or no purchase) based on their browsing behavior and website interaction characteristics.

## Dataset

The dataset consists of feature vectors belonging to 12,330 sessions. The dataset was formed so that each session would belong to a different user in a 1-year period to avoid any tendency to a specific campaign, special day, user profile, or period.

## Attribute

- **Revenue:** class whether it can make a revenue or not
- **Administrative, Administrative Duration, Informational, Informational Duration, Product Related and Product Related Duration:** represent the number of different types of pages visited by the visitor in that session and total time spent in each of these page categories.
- **Bounce Rate:** percentage of visitors who enter the site from that page and then leave (“bounce”) without triggering any other requests to the analytics server during that session
- **Exit Rate:** the percentage that were the last in the session
- **Page Value:** feature represents the average value for a web page that a user visited before completing an e-commerce transaction
- **Special Day:** indicates the closeness of the site visiting time to a specific special day (e.g. Mother’s Day, Valentine’s Day) in which the sessions are more likely to be finalized with transaction. For example, for Valentine’s day, this value takes a nonzero value between February 2 and February 12, zero,before and after this date unless it is close to another special day, and its maximum value of 1 on February 8
- **Operating system,browser, region, traffic type:** Different types of operating systems, browser, region and traffic type used to visit the website
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

__References:__

* https://medium.com/analytics-vidhya/marketing-analytics-significance-of-feature-engineering-model-selection-and-hyper-parameter-53d34b57bc55
* https://github.com/Charlotte-1987/openclassroomP7/blob/main/P7_MODELLING_STREAMLIT_VF.ipynb