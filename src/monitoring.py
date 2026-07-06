"""
monitoring.py — MLOps: request logging, metrics aggregation, and drift detection.

Features:
- Buffered file writes to prevent blocking the async event loop.
- Recovers in-memory metrics from log file on startup.
- Loads actual training class distribution from model metadata.
- Drift detection with configurable threshold.
"""

import atexit
import json
import logging
import os
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Optional

from src.config import settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
WRITE_BUFFER_SIZE = 20  # Flush to disk every N predictions
DRIFT_ALERT_THRESHOLD = 0.25  # Alert if any class drifts >25% from baseline


class MonitoringService:
    """
    In-memory metrics store with buffered JSON-lines logging and drift detection.

    - Predictions are buffered in memory and flushed to disk periodically.
    - On startup, replays existing log file to recover metrics.
    - Drift baseline is loaded from model_metadata.json (actual training distribution).
    """

    def __init__(self) -> None:
        self._total_requests: int = 0
        self._class_counts: Dict[str, int] = defaultdict(int)
        self._confidence_sum: float = 0.0
        self._last_request_at: Optional[str] = None
        self._app_start_time: float = time.time()
        self._write_buffer: list = []
        self._baseline_distribution: Dict[str, float] = {}

        os.makedirs(settings.logs_dir, exist_ok=True)
        self._recover_from_log()

        # Ensure buffer is flushed if process exits unexpectedly
        atexit.register(self._flush_buffer)

    def set_baseline_from_metadata(self, metadata: dict) -> None:
        """
        Load the drift detection baseline from model metadata.

        Falls back to uniform distribution if metadata is missing.

        Args:
            metadata: Model metadata dict (from model_metadata.json).
        """
        dist = metadata.get("class_distribution", {})
        if dist:
            self._baseline_distribution = {str(k): float(v) for k, v in dist.items()}
            logger.info("Drift baseline loaded from metadata: %s", self._baseline_distribution)
        else:
            # Fallback: uniform across 2 classes (No Churn, Churn)
            self._baseline_distribution = {"0": 0.50, "1": 0.50}
            logger.warning(
                "No class_distribution in metadata. Using uniform baseline."
            )

    def _recover_from_log(self) -> None:
        """
        Replay the prediction log file to recover in-memory metrics.

        Called on startup so metrics survive server restarts.
        """
        log_path = settings.log_path
        if not os.path.exists(log_path):
            return

        recovered = 0
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    prediction = entry.get("prediction", {})

                    self._total_requests += 1
                    self._class_counts[str(prediction.get("churn_prediction", 0))] += 1
                    self._confidence_sum += prediction.get("confidence", 0.0)
                    self._last_request_at = entry.get("timestamp")
                    recovered += 1
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Partial log recovery (corrupted entry): %s", exc)

        if recovered > 0:
            logger.info("Recovered %d predictions from log file.", recovered)

    def log_prediction(
        self,
        input_data: Dict[str, Any],
        churn_prediction: int,
        confidence: float,
        model_version: str,
    ) -> None:
        """
        Record a prediction event to buffer and update in-memory metrics.

        Buffer is flushed to disk every WRITE_BUFFER_SIZE predictions.

        Args:
            input_data: Raw input values as dict.
            churn_prediction: Predicted class label (0 or 1).
            confidence: Model confidence score for the top class.
            model_version: Version of the model that made this prediction.
        """
        timestamp = datetime.now().isoformat()

        log_entry = {
            "timestamp": timestamp,
            "model_version": model_version,
            "input": input_data,
            "prediction": {
                "churn_prediction": churn_prediction,
                "confidence": confidence,
            },
        }

        # Buffer the write instead of opening file on every request
        self._write_buffer.append(json.dumps(log_entry))

        if len(self._write_buffer) >= WRITE_BUFFER_SIZE:
            self._flush_buffer()

        # Update in-memory aggregates
        self._total_requests += 1
        self._class_counts[str(churn_prediction)] += 1
        self._confidence_sum += confidence
        self._last_request_at = timestamp

        # Check for distribution drift after accumulating enough data
        if self._total_requests >= 20 and self._baseline_distribution:
            self._check_drift()

    def _flush_buffer(self) -> None:
        """Write all buffered log entries to disk in a single I/O operation."""
        if not self._write_buffer:
            return

        try:
            with open(settings.log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(self._write_buffer) + "\n")
            self._write_buffer.clear()
        except OSError as exc:
            logger.error("Failed to flush prediction log: %s", exc)

    def _check_drift(self) -> None:
        """
        Compare current prediction distribution against training baseline.

        Uses the actual class_distribution from model metadata (not uniform).
        Logs WARNING if deviation exceeds threshold.
        """
        for cls, baseline_pct in self._baseline_distribution.items():
            current_count = self._class_counts.get(cls, 0)
            current_pct = current_count / self._total_requests

            deviation = abs(current_pct - baseline_pct)
            if deviation > DRIFT_ALERT_THRESHOLD:
                logger.warning(
                    "[DRIFT ALERT] Class '%s': current=%.2f%%, baseline=%.2f%%, "
                    "deviation=%.2f%% (threshold=%.2f%%)",
                    cls,
                    current_pct * 100,
                    baseline_pct * 100,
                    deviation * 100,
                    DRIFT_ALERT_THRESHOLD * 100,
                )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return aggregated prediction metrics for the /metrics API endpoint.

        Returns:
            Dict with total_requests, class distribution, avg confidence,
            and last request timestamp.
        """
        avg_confidence = (
            round(self._confidence_sum / self._total_requests, 4)
            if self._total_requests > 0
            else 0.0
        )

        return {
            "total_requests": self._total_requests,
            "predictions_distribution": dict(self._class_counts),
            "average_confidence": avg_confidence,
            "last_request_at": self._last_request_at,
        }

    @property
    def uptime_seconds(self) -> float:
        """Returns elapsed seconds since the monitoring service started."""
        return round(time.time() - self._app_start_time, 2)


# ── Module-level singleton ────────────────────────────────────────────────────
monitoring = MonitoringService()
