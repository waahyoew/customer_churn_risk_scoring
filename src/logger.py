"""
Logging module for the Churn Prediction Pipeline.
Sets up standard logging to both the console and a file.
"""

import logging
import sys
from pathlib import Path
from src.config import settings

def setup_logger(logger_name: str = 'ChurnPipeline') -> logging.Logger:
    """
    Configure and return a structured logger instance.
    The logger writes to both standard output and a file named 'pipeline.log'.

    Args:
        logger_name (str): The name of the logger instance. Defaults to 'ChurnPipeline'.

    Returns:
        logging.Logger: The configured Python logger.

    Raises:
        PermissionError: If the system cannot write to the logs directory.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Prevent adding multiple handlers if setup_logger is called multiple times
    if logger.hasHandlers():
        return logger

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Stream handler (console output)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_file_path = Path(settings.logs_dir) / 'pipeline.log'
    try:
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except PermissionError as error:
        # Fallback to console only if file writing is blocked
        logger.warning(f"Cannot write logs to {log_file_path} (file might be locked). Falling back to console only. Error: {error}")

    return logger
