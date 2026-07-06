"""
schemas.py — Pydantic data contracts for Churn Prediction API.

Defines strict input validation and structured response models
to ensure all API requests are type-safe and well-documented.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, Optional


class ChurnPredictionInput(BaseModel):
    """
    Input schema for a single churn prediction request.

    All fields are required. Validation rules mirror the training data constraints.
    Features should match the columns from the training dataset (excluding ID_Customer and churn).
    """

    Jenis_kelamin: str = Field(
        ...,
        description="Gender of the customer (e.g., 'Laki-laki', 'Perempuan').",
        examples=["Laki-laki"]
    )
    umur: int = Field(
        ...,
        ge=0,
        le=120,
        description="Age of the customer. Must be between 0 and 120.",
        examples=[31]
    )
    membership_program: int = Field(
        ...,
        ge=0,
        description="Months in membership program. Must be 0 or more.",
        examples=[12]
    )
    using_reward: str = Field(
        ...,
        description="Whether customer uses reward program ('Yes' or 'No').",
        examples=["No"]
    )
    pembayaran: str = Field(
        ...,
        description="Payment method (e.g., 'Credit Card', 'Bank Transfer', 'E-Wallet').",
        examples=["Credit Card"]
    )
    Subscribe_brochure: str = Field(
        ...,
        description="Brochure subscription channel (e.g., 'Email', 'SMS', 'WhatsApp').",
        examples=["Email"]
    )
    harga_per_bulan: int = Field(
        ...,
        gt=0,
        description="Monthly subscription price. Must be greater than 0.",
        examples=[50000]
    )
    jumlah_harga_langganan: int = Field(
        ...,
        ge=0,
        description="Total accumulated subscription charges. Must be 0 or more.",
        examples=[600000]
    )

    @field_validator("using_reward")
    @classmethod
    def validate_using_reward(cls, value: str) -> str:
        """Ensure using_reward is either 'Yes' or 'No'."""
        if value not in ("Yes", "No"):
            raise ValueError("using_reward must be 'Yes' or 'No'.")
        return value

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "Jenis_kelamin": "Laki-laki",
                "umur": 31,
                "membership_program": 12,
                "using_reward": "No",
                "pembayaran": "Credit Card",
                "Subscribe_brochure": "Email",
                "harga_per_bulan": 50000,
                "jumlah_harga_langganan": 600000
            }
        }
    )


class PredictionResponse(BaseModel):
    """
    Response schema returned after a successful prediction.

    Includes the predicted churn status, confidence score,
    and per-class probability breakdown.
    """

    churn_prediction: int = Field(
        ..., 
        description="Predicted churn status (0 = No Churn, 1 = Churn)."
    )
    churn_label: str = Field(
        ..., 
        description="Human-readable churn status (e.g., 'No Churn', 'Churn')."
    )
    confidence: float = Field(
        ..., 
        description="Confidence score of the top prediction (0.0–1.0)."
    )
    probabilities: Dict[str, float] = Field(
        ..., 
        description="Predicted probability for each churn class."
    )
    model_version: str = Field(
        ..., 
        description="Version of the model used for prediction."
    )


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""

    status: str = Field(..., description="Overall API status (ok/degraded/error).")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded.")
    model_version: str = Field(..., description="Current model version identifier.")
    uptime_seconds: float = Field(..., description="API uptime in seconds.")


class MetricsResponse(BaseModel):
    """Response schema for the /metrics endpoint."""

    total_requests: int = Field(..., description="Total number of prediction requests.")
    predictions_distribution: Dict[str, int] = Field(
        ..., 
        description="Count of predictions per class."
    )
    average_confidence: float = Field(
        ..., 
        description="Average confidence score across all predictions."
    )
    last_request_at: Optional[str] = Field(
        None, 
        description="Timestamp of the last prediction request."
    )
