"""
test_api.py — Automated tests for the Churn Prediction API endpoints.

Tests cover:
- Happy path: valid input returns correct structure.
- Edge cases: boundary values, extreme ages.
- Validation errors: missing fields, out-of-range values.
- System endpoints: /health and /metrics.

Usage:
    pytest tests/test_api.py -v
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from app.main import app
from src.predictor import predictor
from src.config import settings

client = TestClient(app)


# ── Fixtures ──────────────────────────────────────────────────────────────────
VALID_INPUT = {
    "Jenis_kelamin": "Laki-laki",
    "umur": 31,
    "membership_program": 12,
    "using_reward": "No",
    "pembayaran": "Credit Card",
    "Subscribe_brochure": "Email",
    "harga_per_bulan": 50000,
    "jumlah_harga_langganan": 600000
}

API_KEY_HEADER = {"X-API-Key": settings.metrics_api_key}


def requires_model(func):
    """Decorator that skips test if model is not loaded (instead of false-passing)."""
    return pytest.mark.skipif(
        not predictor.is_loaded,
        reason="Model not loaded — run `python main.py` to train first.",
    )(func)


# ── /health Endpoint Tests ────────────────────────────────────────────────────
class TestHealthEndpoint:
    """Health endpoint should always be available regardless of model state."""

    def test_health_returns_200(self):
        """Health check endpoint should always return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_required_fields(self):
        """Health response must include status, model_loaded, model_version, uptime."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "uptime_seconds" in data

    def test_health_uptime_is_positive(self):
        """Uptime should be a non-negative number."""
        response = client.get("/health")
        data = response.json()
        assert data["uptime_seconds"] >= 0


# ── /metrics Endpoint Tests ───────────────────────────────────────────────────
class TestMetricsEndpoint:
    """Metrics endpoint requires valid API key."""

    def test_metrics_without_key_returns_403(self):
        """Metrics without API key should return HTTP 403 Forbidden."""
        response = client.get("/metrics")
        assert response.status_code == 403

    def test_metrics_with_wrong_key_returns_403(self):
        """Metrics with incorrect API key should return HTTP 403."""
        response = client.get("/metrics", headers={"X-API-Key": "wrong-key"})
        assert response.status_code == 403

    def test_metrics_with_valid_key_returns_200(self):
        """Metrics with valid API key should return HTTP 200."""
        response = client.get("/metrics", headers=API_KEY_HEADER)
        assert response.status_code == 200

    def test_metrics_has_required_fields(self):
        """Metrics response must include aggregation fields."""
        response = client.get("/metrics", headers=API_KEY_HEADER)
        data = response.json()
        assert "total_requests" in data
        assert "predictions_distribution" in data
        assert "average_confidence" in data

    def test_metrics_total_requests_non_negative(self):
        """Total request count must be zero or more."""
        response = client.get("/metrics", headers=API_KEY_HEADER)
        data = response.json()
        assert data["total_requests"] >= 0


# ── /predict Endpoint — Validation Tests (no model required) ──────────────────
class TestPredictValidation:
    """Input validation tests — these work regardless of model state."""

    def test_predict_missing_field_returns_422(self):
        """Request missing required fields must return HTTP 422."""
        incomplete_input = {"Jenis_kelamin": "Laki-laki", "umur": 31}
        response = client.post("/predict", json=incomplete_input)
        assert response.status_code == 422

    def test_predict_invalid_using_reward_returns_422(self):
        """using_reward value of 'Maybe' is invalid — must return HTTP 422."""
        invalid_input = {**VALID_INPUT, "using_reward": "Maybe"}
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422

    def test_predict_negative_age_returns_422(self):
        """Negative age value must fail validation."""
        invalid_input = {**VALID_INPUT, "umur": -5}
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422

    def test_predict_zero_monthly_price_returns_422(self):
        """Monthly price of 0 is invalid (must be > 0)."""
        invalid_input = {**VALID_INPUT, "harga_per_bulan": 0}
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422

    def test_predict_age_over_120_returns_422(self):
        """Age over 120 should fail validation."""
        invalid_input = {**VALID_INPUT, "umur": 150}
        response = client.post("/predict", json=invalid_input)
        assert response.status_code == 422


# ── /predict Endpoint — Inference Tests (model required) ──────────────────────
class TestPredictInference:
    """Inference tests that require a loaded model. Skipped if model is absent."""

    @requires_model
    def test_predict_valid_input_returns_200(self):
        """Valid input should return HTTP 200."""
        response = client.post("/predict", json=VALID_INPUT)
        assert response.status_code == 200

    @requires_model
    def test_predict_response_has_all_fields(self):
        """Response must have all prediction fields."""
        response = client.post("/predict", json=VALID_INPUT)
        data = response.json()
        assert "churn_prediction" in data
        assert "churn_label" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "model_version" in data

    @requires_model
    def test_predict_churn_in_valid_range(self):
        """Churn prediction must be either 0 or 1."""
        response = client.post("/predict", json=VALID_INPUT)
        data = response.json()
        assert data["churn_prediction"] in [0, 1]

    @requires_model
    def test_predict_confidence_between_0_and_1(self):
        """Confidence score must be between 0.0 and 1.0."""
        response = client.post("/predict", json=VALID_INPUT)
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    @requires_model
    def test_predict_probabilities_sum_to_one(self):
        """Probabilities across all classes should sum to approximately 1.0."""
        response = client.post("/predict", json=VALID_INPUT)
        data = response.json()
        prob_sum = sum(data["probabilities"].values())
        assert 0.99 <= prob_sum <= 1.01  # Allow small floating point error

    @requires_model
    def test_predict_female_customer_valid(self):
        """Female customer (Perempuan) is a valid input."""
        valid_input_female = {**VALID_INPUT, "Jenis_kelamin": "Perempuan"}
        response = client.post("/predict", json=valid_input_female)
        assert response.status_code == 200

    @requires_model
    def test_predict_zero_membership_valid(self):
        """Zero months membership program is valid (new customer)."""
        valid_input_new = {**VALID_INPUT, "membership_program": 0}
        response = client.post("/predict", json=valid_input_new)
        assert response.status_code == 200
