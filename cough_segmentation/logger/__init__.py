import logging
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


LOG_FILE = f"{datetime.now().strftime('%m_%d_%y_%H_%M_%S')}.log"
dir_logs = os.path.join(os.getcwd(),"logs")
os.makedirs(dir_logs,exist_ok=True)
LOG_FILE_PATH = os.path.join(dir_logs,LOG_FILE)

# creating formatter for the log entries
formatter = logging.Formatter("[%(asctime)s] - %(lineno)d - %(name)s - %(levelname)s - %(message)s")

# creating a file hanlder to write logs to the file
logs_file_handler = logging.FileHandler(LOG_FILE_PATH)
logs_file_handler.setFormatter(formatter)

# creating a file hanlder to write logs to the console
logs_stream_handler = logging.StreamHandler(sys.stdout)
logs_stream_handler.setFormatter(formatter)

# creating loggers and handlers
logging.basicConfig(
    level=logging.INFO,
    handlers = [logs_file_handler,logs_stream_handler]
)