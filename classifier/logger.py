import logging
import os  # Required for creating directories and checking file paths
from datetime import datetime  # For getting the current date for log naming
from logging.handlers import RotatingFileHandler  # For rotating log files

# Configure logging
def setup_logger():
    # Create a directory for logs based on the current date
    current_date = datetime.now().strftime("%m_%d_%Y")
    log_dir = f"logs/log_{current_date}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Define the log file with a counter (log_1, log_2, etc.)
    log_file = f"{log_dir}/log_{current_date}_"
    
    # Set up the logger with rotating file handler (approx. 1000 lines ~ 10KB)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # File handler that rotates log files after they reach approx. 10000 lines (~10KB)
    file_handler = RotatingFileHandler(f"{log_file}", maxBytes=102400, backupCount=5)  # Adjust maxBytes based on average line length
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Define the log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger