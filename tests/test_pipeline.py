"""
Automated Integration and Model Quality Tests
Verifies data pipeline integrity, feature store validity, and ML inference performance.

Author: oddproblem (https://github.com/oddproblem/ecom-analytics)
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import PROCESSED_ORDERS_PATH, MODEL_ARTIFACT_PATH
from src.predictor import DeliveryDelayPredictor


def test_feature_store_integrity():
    """Verify processed feature store exists and satisfies schema requirements."""
    assert PROCESSED_ORDERS_PATH.exists(), f"Feature store missing at {PROCESSED_ORDERS_PATH}"
    df = pd.read_parquet(PROCESSED_ORDERS_PATH)
    
    assert len(df) > 50000, f"Expected >50k rows, got {len(df)}"
    assert "is_delayed" in df.columns, "Target column 'is_delayed' missing"
    assert "distance_km" in df.columns, "Feature 'distance_km' missing"
    assert "carrier_handoff_lag_days" in df.columns, "Feature 'carrier_handoff_lag_days' missing"
    
    # Target distribution sanity check (delay rate between 5% and 15%)
    delay_rate = df["is_delayed"].mean()
    assert 0.05 <= delay_rate <= 0.15, f"Delay rate {delay_rate:.3f} outside expected 5%-15% bounds"


def test_model_artifact_and_performance():
    """Verify trained model satisfies performance thresholds."""
    assert MODEL_ARTIFACT_PATH.exists(), f"Model artifact missing at {MODEL_ARTIFACT_PATH}"
    predictor = DeliveryDelayPredictor()
    
    # Test ROC-AUC benchmark for Amazon Data Science standard (> 0.80)
    roc_auc = predictor.metrics.get("roc_auc", 0)
    assert roc_auc >= 0.80, f"Model ROC-AUC {roc_auc} fell below 0.80 benchmark"


def test_inference_latency_and_output():
    """Verify real-time prediction output and sub-100ms latency."""
    predictor = DeliveryDelayPredictor()
    sample_order = {
        "distance_km": 850.0,
        "freight_value": 28.50,
        "price": 140.0,
        "product_weight_kg": 2.2,
        "product_volume_liters": 12.0,
        "estimated_window_days": 18.0,
        "approval_delay_hours": 2.0,
        "carrier_handoff_lag_days": 3.0,
        "seller_historical_delay_rate": 0.08,
        "customer_state": "RJ",
        "seller_state": "SP",
        "purchase_dow": 1,
        "purchase_month": 8,
        "is_holiday_season": 0,
    }
    
    # Warm-up call to initialize internal buffers
    predictor.predict(sample_order)

    # Measure steady-state latency
    start_time = time.time()
    result = predictor.predict(sample_order)
    latency_ms = (time.time() - start_time) * 1000.0
    
    assert latency_ms < 100.0, f"Inference latency {latency_ms:.1f}ms exceeds 100ms threshold"
    assert "delay_probability_pct" in result
    assert "risk_level" in result
    assert result["risk_level"] in ["LOW", "MODERATE", "ELEVATED", "CRITICAL"]
    assert "operational_action" in result
    assert isinstance(result["risk_flags"], list)


if __name__ == "__main__":
    test_feature_store_integrity()
    test_model_artifact_and_performance()
    test_inference_latency_and_output()
    print("[PASS] All automated quality tests passed successfully!")
