"""
Preprocessing module.
Contains logic for splitting data and building the Scikit-Learn ColumnTransformer pipeline.
"""

import pandas as pd
from typing import Tuple
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

from src.config import settings
from src.logger import setup_logger

logger = setup_logger()

def split_data(
    features_dataframe: pd.DataFrame, 
    target_series: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into stratified training and testing sets to prevent data leakage.

    Args:
        features_dataframe (pd.DataFrame): The input features.
        target_series (pd.Series): The target variable.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            - X_train: Training features.
            - X_test: Testing features.
            - y_train: Training target.
            - y_test: Testing target.
    """
    logger.info("Splitting data into train and test sets (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        features_dataframe, 
        target_series,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=target_series
    )
    logger.info(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
    return X_train, X_test, y_train, y_test

def build_preprocessor(train_features: pd.DataFrame) -> ColumnTransformer:
    """
    Build a Scikit-Learn ColumnTransformer for imputing and encoding features dynamically based on data types.

    Args:
        train_features (pd.DataFrame): The training feature dataframe used to detect column types.

    Returns:
        ColumnTransformer: A preprocessor ready to be fit or added to a Pipeline.
    
    Raises:
        ValueError: If no features are detected for preprocessing.
    """
    logger.info("Building preprocessing pipeline...")
    
    categorical_columns = train_features.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_columns = train_features.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    if not categorical_columns and not numeric_columns:
        raise ValueError("No numeric or categorical features detected in the dataset.")

    logger.info(f"Detected {len(categorical_columns)} categorical features.")
    logger.info(f"Detected {len(numeric_columns)} numeric features.")

    # Median imputation for numeric columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
    ])

    # Mode imputation and One-Hot Encoding for categorical columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_columns),
            ('cat', categorical_transformer, categorical_columns),
        ],
        remainder='drop',
        verbose_feature_names_out=False
    )
    
    return preprocessor

def create_full_pipeline(preprocessor: ColumnTransformer, model: BaseEstimator) -> Pipeline:
    """
    Combine the preprocessor and an estimator into a single Scikit-Learn Pipeline.

    Args:
        preprocessor (ColumnTransformer): The data preprocessing steps.
        model (BaseEstimator): The Scikit-Learn model/estimator.

    Returns:
        Pipeline: A complete pipeline object.
    """
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model),
    ])
