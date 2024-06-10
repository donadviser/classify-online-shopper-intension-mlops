import os
import logging
from datetime import datetime
from pathlib import Path 


logs_dir = "logs"
log_dir_path = Path(__file__).parent.parent / logs_dir

# Ensure the logs directory exists
log_dir_path.mkdir(parents=True, exist_ok=True) 

log_file_name =f"log_{datetime.now().strftime('%Y_%m_%d')}.log"
log_file_path = os.path.join(log_dir_path, log_file_name)

logging.basicConfig(
    filename = log_file_path,
    format = "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO,
)