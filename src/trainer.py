"""
Trainer module.
Handles hyperparameter tuning using GridSearchCV for multiple model architectures.
"""

import pandas as pd
from typing import Dict, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer

from src.config import (
    DECISION_TREE_PARAM_GRID, 
    RANDOM_FOREST_PARAM_GRID, 
    settings
)
from src.preprocessing import create_full_pipeline
from src.logger import setup_logger

logger = setup_logger()

def tune_models(
    train_features: pd.DataFrame, 
    train_target: pd.Series, 
    preprocessor: ColumnTransformer
) -> Dict[str, GridSearchCV]:
    """
    Train and tune multiple models (Decision Tree and Random Forest) using GridSearchCV.

    Args:
        train_features (pd.DataFrame): Training feature dataset.
        train_target (pd.Series): Training target variable.
        preprocessor (ColumnTransformer): Fitted preprocessor to include in the pipeline.

    Returns:
        Dict[str, GridSearchCV]: A dictionary mapping model names to their fitted GridSearchCV objects.
        
    Raises:
        ValueError: If training features or targets are empty.
    """
    if train_features.empty or train_target.empty:
        raise ValueError("Cannot tune models on empty training data.")

    logger.info("Starting hyperparameter tuning process...")
    cross_validator = StratifiedKFold(
        n_splits=settings.cv_splits, 
        shuffle=True, 
        random_state=settings.random_state
    )
    
    models_to_tune: Dict[str, Dict[str, Any]] = {
        'Decision Tree': {
            'base_model': DecisionTreeClassifier(random_state=settings.random_state),
            'param_grid': DECISION_TREE_PARAM_GRID
        },
        'Random Forest': {
            'base_model': RandomForestClassifier(random_state=settings.random_state),
            'param_grid': RANDOM_FOREST_PARAM_GRID
        }
    }
    
    tuned_models_dict: Dict[str, GridSearchCV] = {}
    
    for model_name, configuration in models_to_tune.items():
        logger.info(f"Initiating GridSearch for {model_name}...")
        pipeline = create_full_pipeline(preprocessor, configuration['base_model'])
        
        search_engine = GridSearchCV(
            estimator=pipeline,
            param_grid=configuration['param_grid'],
            scoring=settings.scoring_metric,
            cv=cross_validator,
            n_jobs=-1,
            refit=True,
            verbose=1
        )
        search_engine.fit(train_features, train_target)
        
        logger.info(f"[{model_name}] Best Parameters: {search_engine.best_params_}")
        logger.info(f"[{model_name}] Best CV {settings.scoring_metric}: {search_engine.best_score_:.4f}")
        
        tuned_models_dict[model_name] = search_engine
        
    return tuned_models_dict
