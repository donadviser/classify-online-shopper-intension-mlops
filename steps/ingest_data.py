from src.logger import logging
from src.exceptions import CustomException
import pandas as pd
from zenml import step
from zenml.config.retry_config import StepRetryConfig



class IngestData:
    """
    Ingests data from a source.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path
        print(self.data_path)

    def get_data(self) -> pd.DataFrame:
        """
        Gets data from a source.
        """
        logging.info(f"Ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)
    

    
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingests data from a source.

    Args:
        data_path (str): The path to the data source.

    Returns:
        pd.DataFrame: The ingested data.
    """

    try:
        logging.info(f"Starting data ingestion STEP")
        data_ingest = IngestData(data_path)
        df_raw = data_ingest.get_data()
        logging.info(f"Data ingestion STEP completed successfully")
        return df_raw
    
    except Exception as e:
        raise CustomException(e, "Error while ingesting data")
    
if __name__ == "__main__":
    df = ingest_df(data_path="/Users/don/github-projects/classify-online-shopper-intension-mlops/data/market_response_data.csv")
    print(df.head())
    
   