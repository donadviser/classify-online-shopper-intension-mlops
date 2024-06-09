import os
import logging
from datetime import datetime
from pathlib import Path 


logs_dir = "logs"
log_dir_path = "/Users/don/github-projects/classify-online-shopper-intension-mlops/logs"
print(log_dir_path)

# Ensure the logs directory exists
os.makedirs(log_dir_path, exist_ok=True) 

log_file_name =f"log_{datetime.now().strftime('%Y_%m_%d')}.log"
log_file_path = os.path.join(log_dir_path, log_file_name)
print(log_file_path)

logging.basicConfig(
    filename = log_file_path,
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO,
)