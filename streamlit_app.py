import json

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from run_deployment import run_main
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer


def main():
    st.title("End to End Online Shopper Intension Pipeline with ZenML")

    high_level_image = Image.open("_assets/high_level_overview.png")
    st.image(high_level_image, caption="High Level Pipeline")

    whole_pipeline_image = Image.open(
        "_assets/training_and_deployment_pipeline_updated.png"
    )

    st.markdown(
        """ 
    #### Problem Statement 
     The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. I will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict the customer satisfaction score for the next order or purchase.    """
    )
    st.image(whole_pipeline_image, caption="Whole Pipeline")
    st.markdown(
        """ 
    Above is a figure of the whole pipeline, we first ingest the data, clean it, train the model, and evaluate the model, and if data source changes or any hyperparameter values changes, deployment will be triggered, and (re) trains the model and if the model meets minimum accuracy requirement, the model will be deployed.
    """
    )

    st.markdown(
        """ 
    #### Description of Features 
    This app is designed to predict online shoppers purchasing behaviour. You can input the features of the product listed below and get the customer shopping intension. 
    | Features        | Description   | 
    | ------------- | -     | 
    | Revenue | Class whether it can make a revenue or not. | 
    | Administrative   | represent the number of different types of pages visited by the visitor in that session and total time spent in each of these page categories. |
    | Administrative Duration   | Number of installments chosen by the customer. |  
    | Informational   | Number of installments chosen by the customer. |  
    | Informational Duration   | Number of installments chosen by the customer. |  
    | Product Related   | Number of installments chosen by the customer. |  
    | Product Related Duration   | Number of installments chosen by the customer. |  
    | Bounce Rate |       percentage of visitors who enter the site from that page and then leave (“bounce”) without triggering any other requests to the analytics server during that session. | 
    | Exit Rate |       the percentage that were the last in the session. |
    | Page Value |    feature represents the average value for a web page that a user visited before completing an e-commerce transaction.  | 
    | Special Day |    indicates the closeness of the site visiting time to a specific special day (e.g. Mother’s Day, Valentine’s Day) in which the sessions are more likely to be finalized with transaction. For example, for Valentine’s day, this value takes a nonzero value between February 2 and February 12, zero,before and after this date unless it is close to another special day, and its maximum value of 1 on February 8. |
    | Operating system |    Length of the product description. |
    | browser |    Number of product published photos |
    | region |    Weight of the product measured in grams. | 
    | traffic type |    Length of the product measured in centimeters. |
    | Visitor type |    Whether the customer is a returning or new visitor. |
    | Weekend |    A Boolean value indicating whether the date of the visit is weekend. |
    | Month |    Month of the year. |
    """
    )
    Administrative = st.sidebar.slider("Administrative")
    Administrative_Duration = st.sidebar.slider("Payment Installments")
    Informational = st.number_input("Informational")
    Informational_Duration = st.number_input("Informational_Duration")
    ProductRelated = st.number_input("ProductRelated")
    ProductRelated_Duration = st.number_input("ProductRelated_Duration")
    BounceRates = st.number_input("BounceRates")
    ExitRates = st.number_input("ExitRates")
    PageValues = st.number_input("PageValues")
    SpecialDay = st.number_input("SpecialDay")
    Month = st.number_input("Month")
    OperatingSystems = st.number_input("OperatingSystems")
    Browser = st.number_input("Browser")
    Region = st.number_input("Region")
    TrafficType = st.number_input("TrafficType")
    VisitorType = st.number_input("VisitorType")
    Weekend = st.number_input("Weekend")
    Revenue = st.number_input("Revenue")

    if st.button("Predict"):
        model_deployer = MLFlowModelDeployer.get_active_model_deployer()

        service = model_deployer.find_model_server(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
        )[0]
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            run_main()

        df = pd.DataFrame(
            {
                "Administrative": [Administrative],
                "Administrative_Duration": [Administrative_Duration],
                "Informational": [Informational],
                "ProductRelated": [ProductRelated],
                "Informational_Duration": [Informational_Duration],
                "ProductRelated_Duration": [ProductRelated_Duration],
                "BounceRates": [BounceRates],
                "ExitRates": [ExitRates],
                "PageValues": [PageValues],
                "SpecialDay": [SpecialDay],
                "Month": [Month],
                "OperatingSystems": [OperatingSystems],
                "Browser": [Browser],
                "Region": [Region],
                "TrafficType": [TrafficType],
                "VisitorType": [VisitorType],
                "Weekend": [Weekend],
                "Revenue": [Revenue],
            }
        )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        st.success(
            "Your Customer Satisfactory rate(range between 0 - 5) with given product details is :-{}".format(
                pred
            )
        )
    if st.button("Results"):
        st.write(
            "We have experimented with two ensemble and tree based models and compared the performance of each model. The results are as follows:"
        )

        df = pd.DataFrame(
            {
                "Models": ["LightGBM", "Xgboost"],
                "MSE": [1.804, 1.781],
                "RMSE": [1.343, 1.335],
            }
        )
        st.dataframe(df)

        st.write(
            "Following figure shows how important each feature is in the model that contributes to the target variable or contributes in predicting customer satisfaction rate."
        )
        image = Image.open("_assets/feature_importance_xgboost.png")
        st.image(image, caption="Feature Importance Gain")


if __name__ == "__main__":
    main()