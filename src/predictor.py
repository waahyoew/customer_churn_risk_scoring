"""
predictor.py — Model loader and inference engine.

Loads the trained churn prediction model from disk at application startup,
verifies its integrity via SHA-256 checksum, and exposes a clean
predict() interface for the API layer.

Feature ordering is guaranteed by using pandas DataFrame with proper column names.
"""

import hashlib
import json
import logging
import os

import joblib
import pandas as pd

from src.schemas import ChurnPredictionInput, PredictionResponse
from src.config import CHURN_LABELS, settings

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """
    Singleton predictor that wraps the trained churn prediction model.

    Handles model loading, integrity verification, feature ordering,
    and probability-based response construction.
    Designed to be instantiated once at app startup.
    """

    def __init__(self) -> None:
        self._model = None
        self._metadata: dict = {}
        self._is_loaded: bool = False

    def load(self) -> None:
        """
        Load the trained model and its metadata from disk.

        Performs SHA-256 integrity check if checksum is available
        in model_metadata.json.

        Raises:
            FileNotFoundError: If model file does not exist.
            RuntimeError: If model file fails integrity check.
        """
        model_path = settings.model_path
        metadata_path = settings.metadata_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}. "
                "Please run `python train.py` first."
            )

        # Load metadata first (for integrity check)
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)
            logger.info("Model version: %s", self._metadata.get("model_version", "unknown"))

        # Integrity check: verify SHA-256 before loading
        expected_checksum = self._metadata.get("model_checksum")
        if expected_checksum:
            actual_checksum = self._compute_checksum(model_path)
            if actual_checksum != expected_checksum:
                raise RuntimeError(
                    f"Model integrity check FAILED.\n"
                    f"Expected: {expected_checksum}\n"
                    f"Actual:   {actual_checksum}\n"
                    f"The model file may be corrupted or tampered with."
                )
            logger.info("Model integrity check passed (SHA-256 match).")

        logger.info("Loading model from %s ...", model_path)
        self._model = joblib.load(model_path)
        self._is_loaded = True
        logger.info("Model loaded successfully.")

    @staticmethod
    def _compute_checksum(filepath: str) -> str:
        """
        Compute SHA-256 checksum of a file.

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

    @property
    def is_loaded(self) -> bool:
        """Returns True if the model is loaded and ready."""
        return self._is_loaded

    @property
    def model_version(self) -> str:
        """Returns the version string from model metadata."""
        return self._metadata.get("model_version", "unknown")

    @property
    def metadata(self) -> dict:
        """Returns the full model metadata dictionary."""
        return self._metadata

    def _build_feature_dataframe(self, data: ChurnPredictionInput) -> pd.DataFrame:
        """
        Convert a validated input schema into a pandas DataFrame.

        This ensures feature order consistency between training and serving.

        Args:
            data: Validated input from the API request.

        Returns:
            DataFrame with a single row containing all features.
        """
        input_dict = data.model_dump()
        # Convert to DataFrame to ensure proper column ordering and handling
        return pd.DataFrame([input_dict])

    def predict(self, data: ChurnPredictionInput) -> PredictionResponse:
        """
        Run inference on a single churn prediction input.

        Args:
            data: Validated ChurnPredictionInput from the API layer.

        Returns:
            PredictionResponse with churn_prediction, label, confidence,
            and per-class probability breakdown.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        features_df = self._build_feature_dataframe(data)

        # Predict using the pipeline (which includes preprocessing)
        predicted_class = int(self._model.predict(features_df)[0])
        probabilities_raw = self._model.predict_proba(features_df)[0]
        classes = self._model.classes_

        confidence = float(max(probabilities_raw))
        probabilities_dict = {
            str(int(cls)): round(float(prob), 4)
            for cls, prob in zip(classes, probabilities_raw)
        }

        return PredictionResponse(
            churn_prediction=predicted_class,
            churn_label=CHURN_LABELS.get(predicted_class, "Unknown"),
            confidence=round(confidence, 4),
            probabilities=probabilities_dict,
            model_version=self.model_version,
        )


# ── Module-level singleton ────────────────────────────────────────────────────
predictor = ChurnPredictor()
