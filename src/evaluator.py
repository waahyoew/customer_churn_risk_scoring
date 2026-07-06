"""
Evaluator module.
Handles model selection, evaluation, and saving artifacts to disk.
Includes SHA-256 checksum generation and class distribution tracking.
"""

import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Any
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
)

from src.config import settings
from src.logger import setup_logger

logger = setup_logger()

def select_best_model(
    tuned_models: Dict[str, GridSearchCV], 
    test_features: pd.DataFrame, 
    test_target: pd.Series
) -> Tuple[str, BaseEstimator]:
    """
    Select the best performing model based on a hierarchical criteria: Test F1 -> ROC-AUC -> Accuracy.

    Args:
        tuned_models (Dict[str, GridSearchCV]): A dictionary of tuned models.
        test_features (pd.DataFrame): Testing feature set.
        test_target (pd.Series): Testing target variable.

    Returns:
        Tuple[str, BaseEstimator]: The name and the estimator instance of the best model.
        
    Raises:
        ValueError: If tuned_models dictionary is empty.
    """
    if not tuned_models:
        raise ValueError("The tuned_models dictionary is empty. Cannot select a model.")

    logger.info("Selecting the best model based on Test F1 -> ROC-AUC -> Accuracy...")
    model_summary_list: list[dict[str, Any]] = []
    
    for model_name, grid_search_object in tuned_models.items():
        estimator = grid_search_object.best_estimator_
        predictions = estimator.predict(test_features)
        probabilities = estimator.predict_proba(test_features)[:, 1]
        
        f1_metric = f1_score(test_target, predictions, zero_division=0)
        roc_auc_metric = roc_auc_score(test_target, probabilities)
        accuracy_metric = accuracy_score(test_target, predictions)
        
        model_summary_list.append({
            'model_name': model_name,
            'f1_score': f1_metric,
            'roc_auc': roc_auc_metric,
            'accuracy': accuracy_metric,
            'estimator': estimator
        })
        logger.info(
            f"[{model_name}] Test F1: {f1_metric:.4f}, "
            f"ROC-AUC: {roc_auc_metric:.4f}, Acc: {accuracy_metric:.4f}"
        )
        
    summary_dataframe = pd.DataFrame(model_summary_list)
    summary_dataframe.sort_values(
        by=['f1_score', 'roc_auc', 'accuracy'], 
        ascending=False, 
        inplace=True
    )
    
    best_model_row = summary_dataframe.iloc[0]
    best_model_name: str = str(best_model_row['model_name'])
    best_model_estimator: BaseEstimator = best_model_row['estimator']
    
    logger.info(f"Final selected model: {best_model_name}")
    return best_model_name, best_model_estimator

def evaluate_model(
    model_name: str, 
    estimator: BaseEstimator, 
    test_features: pd.DataFrame, 
    test_target: pd.Series
) -> Dict[str, float]:
    """
    Evaluate the final model and return a dictionary of metrics.

    Args:
        model_name (str): The name of the model being evaluated.
        estimator (BaseEstimator): The fitted Scikit-Learn pipeline/model.
        test_features (pd.DataFrame): Testing feature set.
        test_target (pd.Series): Testing target variable.

    Returns:
        Dict[str, float]: Computed classification metrics.
    """
    logger.info(f"Evaluating final model ({model_name}) on test set...")
    
    predictions = estimator.predict(test_features)
    probabilities = estimator.predict_proba(test_features)[:, 1]
    
    confusion_mat = confusion_matrix(test_target, predictions)
    mcc_score = matthews_corrcoef(test_target, predictions)
    
    computed_metrics: Dict[str, float] = {
        'accuracy': float(accuracy_score(test_target, predictions)),
        'precision': float(precision_score(test_target, predictions, zero_division=0)),
        'recall': float(recall_score(test_target, predictions, zero_division=0)),
        'f1_score': float(f1_score(test_target, predictions, zero_division=0)),
        'roc_auc': float(roc_auc_score(test_target, probabilities)),
        'mcc': float(mcc_score),
    }
    
    logger.info(f"Final Model Metrics: {computed_metrics}")
    logger.info(f"Confusion Matrix:\n{confusion_mat}")
    logger.info(f"Classification Report:\n{classification_report(test_target, predictions, zero_division=0)}")
    
    return computed_metrics

def save_model_artifacts(
    model_name: str, 
    estimator: BaseEstimator, 
    metrics: Dict[str, float],
    y_train: pd.Series
) -> None:
    """
    Save the model estimator and its metadata to the disk.
    Includes SHA-256 checksum and training class distribution.

    Args:
        model_name (str): The name of the model.
        estimator (BaseEstimator): The fitted Scikit-Learn pipeline.
        metrics (Dict[str, float]): The evaluation metrics to store in metadata.
        y_train (pd.Series): Training target for computing class distribution.
    """
    logger.info("Saving model and metadata artifacts...")
    
    # Save the physical model
    model_file_path = settings.model_path
    joblib.dump(estimator, model_file_path)
    logger.info(f"Model successfully saved to: {model_file_path}")
    
    # Compute SHA-256 checksum
    checksum = _compute_model_checksum(model_file_path)
    logger.info(f"Model checksum (SHA-256): {checksum}")
    
    # Compute class distribution for drift baseline
    class_distribution = _compute_class_distribution(y_train)
    logger.info(f"Training class distribution: {class_distribution}")
    
    # Safely extract feature count if available in the pipeline
    try:
        feature_names = estimator.named_steps['preprocessor'].get_feature_names_out()
        feature_count = len(feature_names)
    except (AttributeError, KeyError) as warning:
        logger.warning(f"Could not extract feature names for metadata: {warning}")
        feature_count = 0
        
    metadata: Dict[str, Any] = {
        'model_name': model_name,
        'model_version': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'trained_at': datetime.now().isoformat(),
        'random_state': settings.random_state,
        'test_size_ratio': settings.test_size,
        'best_hyperparameters': estimator.named_steps['model'].get_params(),
        'metrics': metrics,
        'feature_count': feature_count,
        'model_checksum': checksum,
        'class_distribution': class_distribution,
    }
    
    metadata_file_path = settings.metadata_path
    with open(metadata_file_path, 'w', encoding='utf-8') as file_handler:
        json.dump(metadata, file_handler, indent=2, ensure_ascii=False)
        
    logger.info(f"Metadata successfully saved to: {metadata_file_path}")


def _compute_model_checksum(filepath: str) -> str:
    """
    Compute SHA-256 checksum of a file for integrity verification.

    Args:
        filepath: Absolute path to the file.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _compute_class_distribution(y: pd.Series) -> Dict[str, float]:
    """
    Compute the proportion of each class in the target series.

    Used as the drift detection baseline in monitoring.

    Args:
        y: Target labels series.

    Returns:
        Dict mapping class label (str) to proportion (float).
    """
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    return {str(int(cls)): round(count / total, 4) for cls, count in zip(unique, counts)}

