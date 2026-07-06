"""
main.py — FastAPI application: Churn Prediction REST API.

Endpoints:
    POST /predict     → Run churn prediction inference on a single customer.
    GET  /health      → System health check with model status.
    GET  /metrics     → MLOps metrics (API-key protected).
    GET  /docs        → Auto-generated Swagger UI (built-in FastAPI).
"""

import logging
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project root to Python path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.monitoring import monitoring
from src.predictor import predictor
from src.schemas import (
    ChurnPredictionInput,
    HealthResponse,
    MetricsResponse,
    PredictionResponse,
)
from src.config import settings

# ── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan: load model on startup ──────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the ML model and initialize monitoring baseline on startup."""
    logger.info("Starting up Churn Prediction API ...")
    try:
        predictor.load()
        monitoring.set_baseline_from_metadata(predictor.metadata)
        logger.info("Model ready. API is accepting requests.")
    except FileNotFoundError as exc:
        logger.error("STARTUP FAILED: %s", exc)
        logger.error("Run `python main.py` to train the model first.")
    except RuntimeError as exc:
        logger.error("MODEL INTEGRITY CHECK FAILED: %s", exc)
    yield
    logger.info("Shutting down API ...")


# ── App Initialization ────────────────────────────────────────────────────────
app = FastAPI(
    title="Churn Prediction API",
    description=(
        "Real-time customer churn prediction API for subscription-based businesses. "
        "Predicts churn status (0 = No Churn, 1 = Churn) based on customer profile. "
        "Built with FastAPI + scikit-learn."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: restricted to configured origins (not wildcard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins_list(),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict customer churn",
    tags=["Inference"],
)
async def predict(input_data: ChurnPredictionInput) -> PredictionResponse:
    """
    Classify churn risk for a single customer.

    Accepts customer profile features and returns:
    - `churn_prediction` (int 0 or 1): Predicted churn status.
    - `churn_label` (str): Human-readable churn label.
    - `confidence` (float): Top-class probability score.
    - `probabilities` (dict): Full probability breakdown per class.

    Raises:
        503: If model has not been loaded successfully.
        500: If an unexpected inference error occurs.
    """
    if not predictor.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please contact the administrator.",
        )

    try:
        response = predictor.predict(input_data)
        monitoring.log_prediction(
            input_data=input_data.model_dump(),
            churn_prediction=response.churn_prediction,
            confidence=response.confidence,
            model_version=response.model_version,
        )
        logger.info(
            "Prediction: churn=%d, confidence=%.4f",
            response.churn_prediction,
            response.confidence,
        )
        return response

    except Exception as exc:
        logger.exception("Inference failed: %s", exc)
        raise HTTPException(status_code=500, detail="Inference error. Check server logs.")


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health() -> HealthResponse:
    """
    Return the current health status of the API and model.

    Returns model_loaded status, version, and API uptime.
    """
    return HealthResponse(
        status="ok" if predictor.is_loaded else "degraded",
        model_loaded=predictor.is_loaded,
        model_version=predictor.model_version,
        uptime_seconds=monitoring.uptime_seconds,
    )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="MLOps prediction metrics",
    tags=["Monitoring"],
)
async def metrics(x_api_key: str = Header(default=None)) -> MetricsResponse:
    """
    Return aggregated MLOps metrics since last server restart.

    Requires `X-API-Key` header matching the configured METRICS_API_KEY.
    This prevents unauthorized access to business-sensitive prediction data.
    """
    if x_api_key != settings.metrics_api_key:
        raise HTTPException(
            status_code=403,
            detail="Forbidden: invalid or missing X-API-Key header.",
        )
    data = monitoring.get_metrics()
    return MetricsResponse(**data)
