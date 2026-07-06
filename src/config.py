"""
config.py — Centralized configuration for the Churn Prediction system.

Single source of truth for:
- Feature column names and ordering (prevents train/serve skew).
- Encoding mappings used during preprocessing.
- Churn label definitions.
- File paths (model, logs, data).
- Environment-driven settings (CORS, API keys, ports).

Both train.py and the FastAPI app import from this module,
ensuring preprocessing logic can never diverge.
"""

import os
from pathlib import Path
from typing import Dict, List
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Project Root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── ML Constants (shared between training and serving) ────────────────────────
TARGET_COLUMN_NAME: str = 'churn'
CUSTOMER_ID_COLUMN_NAME: str = 'ID_Customer'
FEATURES_TO_DROP: List[str] = [CUSTOMER_ID_COLUMN_NAME]

# Target Variable Mapping to Numeric
TARGET_CLASS_MAPPING: Dict[str, int] = {'No': 0, 'Yes': 1}

# Churn Labels (for API response)
CHURN_LABELS: Dict[int, str] = {
    0: "No Churn",
    1: "Churn"
}

# Hyperparameter Grids for GridSearchCV
DECISION_TREE_PARAM_GRID: Dict[str, List] = {
    'model__max_depth': [4, 8, 16, 32, 64, None],
    'model__min_samples_split': [2, 4, 8, 16, 32],
}

RANDOM_FOREST_PARAM_GRID: Dict[str, List] = {
    'model__n_estimators': [50, 100, 150],
    'model__max_depth': [4, 8, 16, None],
    'model__min_samples_split': [2, 4, 8, 16],
}


# ── Environment-Driven Settings ──────────────────────────────────────────────
class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.

    Attributes:
        model_path: Path to the trained model .joblib artifact.
        metadata_path: Path to the model metadata JSON file.
        log_path: Path to the prediction log (JSONL).
        cors_origins: Comma-separated list of allowed CORS origins.
        metrics_api_key: API key required to access /metrics endpoint.
        data_url: Remote URL for the training dataset (fallback).
    """

    # Data Configuration
    data_url: str = "https://storage.googleapis.com/dqlab-dataset/cth_churn_analysis_train.xlsx"
    data_dir: str = str(PROJECT_ROOT / "data")
    local_data_path: str = str(PROJECT_ROOT / "data" / "cth_churn_analysis_train.xlsx")

    # Model Configuration
    models_dir: str = str(PROJECT_ROOT / "models")
    model_path: str = str(PROJECT_ROOT / "models" / "churn_model_v2.joblib")
    metadata_path: str = str(PROJECT_ROOT / "models" / "churn_model_v2_metadata.json")

    # Logging Configuration
    logs_dir: str = str(PROJECT_ROOT / "logs")
    log_path: str = str(PROJECT_ROOT / "logs" / "predictions.jsonl")

    # Training Configuration
    random_state: int = 57
    test_size: float = 0.2
    cv_splits: int = 5
    scoring_metric: str = "f1"

    # API Configuration
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    metrics_api_key: str = "dev-key-change-in-production"

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="CHURN_",
        extra="allow"
    )

    def get_cors_origins_list(self) -> List[str]:
        """Parse comma-separated CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]


# ── Module-level singleton ────────────────────────────────────────────────────
settings = Settings()

# Create directories if they don't exist
Path(settings.models_dir).mkdir(parents=True, exist_ok=True)
Path(settings.logs_dir).mkdir(parents=True, exist_ok=True)
Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
