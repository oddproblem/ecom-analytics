"""
Configuration and constants for Amazon Fulfillment & Delivery SLA Intelligence Platform.
Author: oddproblem (https://github.com/oddproblem/ecom-analytics)
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Processed Cache Files
PROCESSED_ORDERS_PATH = DATA_PROCESSED_DIR / "orders_sla_features.parquet"
PROCESSED_KPIS_PATH = DATA_PROCESSED_DIR / "executive_kpis.parquet"
STATE_LOGISTICS_PATH = DATA_PROCESSED_DIR / "state_logistics.parquet"
MODEL_ARTIFACT_PATH = MODELS_DIR / "delivery_delay_classifier.joblib"

# Amazon Business & SLA Assumptions
ESTIMATED_CONCESSION_COST_USD = 12.50  # Average carrier concession / customer appeasement coupon per late order
PROACTIVE_NOTIFICATION_COST_USD = 0.85  # Cost to send automated proactive SMS/email with credit before complaint
CHURN_RISK_MULTIPLIER = 3.4            # Dissatisfaction multiplier for late deliveries vs on-time
RANDOM_STATE = 42

# Key Target & Feature definitions
TARGET_COL = "is_delayed"
NUMERIC_FEATURES = [
    "distance_km",
    "freight_value",
    "price",
    "freight_ratio",
    "product_weight_kg",
    "product_volume_liters",
    "estimated_window_days",
    "approval_delay_hours",
    "carrier_handoff_lag_days",
    "seller_historical_delay_rate",
    "items_count",
    "payment_installments",
]

CATEGORICAL_FEATURES = [
    "is_same_state",
    "customer_state",
    "seller_state",
    "purchase_dow",
    "purchase_month",
    "is_holiday_season",
]
