"""
Pipeline orchestration module.
Coordinates the execution of the entire machine learning pipeline.
"""

from src.logger import setup_logger
from src.data_loader import load_data, clean_and_split_features_target
from src.preprocessing import split_data, build_preprocessor
from src.trainer import tune_models
from src.evaluator import select_best_model, evaluate_model, save_model_artifacts

logger = setup_logger()

def run_pipeline() -> None:
    """
    Execute the end-to-end churn prediction pipeline.
    This includes loading data, preprocessing, model tuning, evaluation, and saving artifacts.

    Raises:
        Exception: Re-raises any exception encountered during the pipeline execution 
                   so the calling process (e.g., CI/CD) can detect the failure.
    """
    logger.info("=== Starting Churn Prediction Pipeline ===")
    try:
        # Step 1: Load Raw Data
        raw_dataframe = load_data()
        
        # Step 2: Clean Data and Extract Target Variable
        features_dataframe, target_series = clean_and_split_features_target(raw_dataframe)
        
        # Step 3: Train-Test Split (Prevent Data Leakage)
        X_train, X_test, y_train, y_test = split_data(features_dataframe, target_series)
        
        # Step 4: Build Preprocessor dynamically
        preprocessor = build_preprocessor(X_train)
        
        # Step 5: Tune Models via GridSearchCV
        tuned_models_dictionary = tune_models(X_train, y_train, preprocessor)
        
        # Step 6: Select Best Model based on hierarchical criteria
        best_model_name, best_estimator = select_best_model(
            tuned_models=tuned_models_dictionary, 
            test_features=X_test, 
            test_target=y_test
        )
        
        # Step 7: Evaluate the chosen model
        final_metrics = evaluate_model(
            model_name=best_model_name, 
            estimator=best_estimator, 
            test_features=X_test, 
            test_target=y_test
        )
        
        # Step 8: Save Model and Metadata Artifacts (with class distribution)
        save_model_artifacts(
            model_name=best_model_name, 
            estimator=best_estimator, 
            metrics=final_metrics,
            y_train=y_train
        )
        
        logger.info("=== Pipeline Completed Successfully ===")
        
    except Exception as error:
        logger.error(f"Pipeline failed with error: {str(error)}", exc_info=True)
        raise
