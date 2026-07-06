"""
Data loader module.
Handles data ingestion from local paths or remote URLs, and performs initial target separation.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple
import urllib.error

from src.config import (
    TARGET_COLUMN_NAME, 
    FEATURES_TO_DROP, 
    TARGET_CLASS_MAPPING,
    settings
)
from src.logger import setup_logger

logger = setup_logger()

def load_data() -> pd.DataFrame:
    """
    Load dataset from a local path, falling back to a remote URL if the local file is missing.
    
    Returns:
        pd.DataFrame: The loaded dataset.
        
    Raises:
        FileNotFoundError: If the local file is not found and remote URL fails.
        ValueError: If the loaded dataframe is empty.
    """
    dataframe: pd.DataFrame = pd.DataFrame()
    
    local_path = Path(settings.local_data_path)
    
    if local_path.exists():
        logger.info(f"Loading data from local path: {local_path}")
        try:
            dataframe = pd.read_excel(local_path)
        except Exception as error:
            logger.error(f"Error reading local excel file: {error}")
            raise
    else:
        logger.warning(f"Local data not found at {local_path}. Attempting to download from URL: {settings.data_url}")
        try:
            dataframe = pd.read_excel(settings.data_url)
        except urllib.error.URLError as error:
            logger.error(f"Failed to fetch data from URL: {error}")
            raise FileNotFoundError(f"Data not found locally and URL fetch failed: {error}")
        except Exception as error:
            logger.error(f"Unexpected error loading data from URL: {error}")
            raise

    if dataframe.empty:
        logger.error("Loaded dataframe is completely empty.")
        raise ValueError("The loaded dataset contains no data.")

    logger.info(f"Successfully loaded dataset with shape: {dataframe.shape}")
    return dataframe

def clean_and_split_features_target(dataframe: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Clean the dataset by dropping irrelevant columns and splitting the features from the target variable.
    Also maps the target variable to numerical values.

    Args:
        dataframe (pd.DataFrame): The raw input dataframe.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - X (pd.DataFrame): The feature set.
            - y (pd.Series): The numeric target variable.

    Raises:
        KeyError: If the target column is missing from the dataset.
        ValueError: If mapping the target variable introduces NaN values.
    """
    logger.info("Cleaning data and extracting target variable...")
    dataframe_clean = dataframe.copy()

    # Drop unwanted columns if they exist
    for column_name in FEATURES_TO_DROP:
        if column_name in dataframe_clean.columns:
            dataframe_clean.drop(columns=column_name, inplace=True)
            logger.info(f"Dropped column: {column_name}")

    if TARGET_COLUMN_NAME not in dataframe_clean.columns:
        logger.error(f"Target column '{TARGET_COLUMN_NAME}' is missing.")
        raise KeyError(f"Target column '{TARGET_COLUMN_NAME}' not found in dataset.")

    target_raw_series = dataframe_clean.pop(TARGET_COLUMN_NAME)

    # Map target categories to numerical values
    target_numeric_series = target_raw_series.map(TARGET_CLASS_MAPPING)
    
    if target_numeric_series.isna().any():
        logger.error(f"Target mapping resulted in NaN. Original unique values: {target_raw_series.unique()}")
        raise ValueError(
            f"Target mapping produced missing values. Ensure TARGET_CLASS_MAPPING covers all categories. "
            f"Expected keys: {list(TARGET_CLASS_MAPPING.keys())}"
        )
    
    logger.info(f"Remaining feature columns count: {len(dataframe_clean.columns)}")
    return dataframe_clean, target_numeric_series
