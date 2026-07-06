"""
train.py — Standalone training pipeline for Churn Prediction Model.

This script provides a clean CLI interface for training the model
without needing to use the main.py orchestrator.

Responsibilities:
- Load and preprocess the churn dataset.
- Train the best performing model using GridSearchCV.
- Save the trained model artifact + metadata (including SHA-256 checksum).
- Compute and save training class distribution for drift detection baseline.

Usage:
    python train.py
"""

import sys
from pathlib import Path

# Add project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent))

from src.pipeline import run_pipeline
from src.logger import setup_logger


def main() -> None:
    """
    Main training entrypoint: orchestrates the full pipeline execution.
    """
    logger = setup_logger('TrainingScript')
    
    try:
        logger.info("=" * 60)
        logger.info("Starting Churn Prediction Training Pipeline")
        logger.info("=" * 60)
        
        run_pipeline()
        
        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info("Model saved to: models/final_churn_model.joblib")
        logger.info("Metadata saved to: models/final_churn_model_metadata.json")
        logger.info("=" * 60)
        
        sys.exit(0)
        
    except Exception as error:
        logger.critical(f"Training failed with error: {error}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
