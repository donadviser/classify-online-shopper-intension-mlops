from pipelines.training_pipeline import train_pipeline
from zenml.client import Client


if __name__ == "__main__":
    print("Training Pipeline Starts")
    #print(Client().active_stack.experiment_tracker.get_tracking_uri())
    data_path = "https://raw.githubusercontent.com/donadviser/datasets/master/data-don/online_shoppers_intention.csv"
    train_pipeline(data_path)