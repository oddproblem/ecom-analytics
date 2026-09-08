"""
Real-Time Inference Engine for Delivery SLA Risk Prediction
Author: oddproblem (https://github.com/oddproblem/ecom-analytics)
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.config import MODEL_ARTIFACT_PATH


class DeliveryDelayPredictor:
    def __init__(self, model_path: Path = MODEL_ARTIFACT_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {model_path}. Train the model first using src/model.py.")
        
        self.payload = joblib.load(model_path)
        self.pipeline = self.payload["pipeline"]
        self.metrics = self.payload["metrics"]
        self.optimal_threshold = self.payload.get("optimal_threshold", 0.75)
        self.num_cols = self.payload["num_cols"]
        self.cat_cols = self.payload["cat_cols"]

    def predict(self, input_data: dict) -> dict:
        """
        Takes order features dict and returns prediction, probability, risk level, and mitigation actions.
        """
        # Build DataFrame with single row
        df_input = pd.DataFrame([input_data])

        # Auto-compute derived features if missing
        if "freight_ratio" not in df_input.columns:
            freight = df_input.get("freight_value", [20.0])[0]
            price = df_input.get("price", [100.0])[0]
            df_input["freight_ratio"] = round(freight / (price + 1.0), 4)

        if "is_same_state" not in df_input.columns:
            c_state = df_input.get("customer_state", ["SP"])[0]
            s_state = df_input.get("seller_state", ["SP"])[0]
            df_input["is_same_state"] = int(c_state == s_state)

        # Default fallbacks
        defaults = {
            "distance_km": 450.0,
            "freight_value": 18.50,
            "price": 95.0,
            "freight_ratio": 0.19,
            "product_weight_kg": 1.2,
            "product_volume_liters": 8.0,
            "estimated_window_days": 20.0,
            "approval_delay_hours": 1.5,
            "carrier_handoff_lag_days": 2.5,
            "seller_historical_delay_rate": 0.07,
            "items_count": 1,
            "is_same_state": 1,
            "customer_state": "SP",
            "seller_state": "SP",
            "purchase_dow": 2,
            "purchase_month": 6,
            "is_holiday_season": 0,
        }

        for col in self.num_cols + self.cat_cols:
            if col not in df_input.columns:
                df_input[col] = defaults.get(col, 0)

        # Predict probability
        prob = float(self.pipeline.predict_proba(df_input[self.num_cols + self.cat_cols])[0, 1])

        # Risk level categorization
        if prob < 0.25:
            risk_level = "LOW"
            risk_color = "#10B981"  # Emerald Green
            action = "Standard fulfillment pipeline. No manual intervention required."
        elif prob < 0.50:
            risk_level = "MODERATE"
            risk_color = "#F59E0B"  # Amber
            action = "Monitor tracking events. Trigger automated carrier ping if dispatch exceeds 48 hours."
        elif prob < 0.75:
            risk_level = "ELEVATED"
            risk_color = "#F97316"  # Orange
            action = "Expedite carrier pickup. Assign priority line-haul transit to avoid SLA breach."
        else:
            risk_level = "CRITICAL"
            risk_color = "#EF4444"  # Red
            action = "High probability of breach! Reroute to express carrier and trigger proactive customer notification."

        # Operational risk factors detected
        flags = []
        if df_input["carrier_handoff_lag_days"].iloc[0] > 4.0:
            flags.append(f"Excessive seller handoff lag ({df_input['carrier_handoff_lag_days'].iloc[0]:.1f} days)")
        if df_input["distance_km"].iloc[0] > 1200:
            flags.append(f"Long-haul cross-country transit ({df_input['distance_km'].iloc[0]:.0f} km)")
        if df_input["is_same_state"].iloc[0] == 0:
            flags.append(f"Interstate border crossing ({df_input['seller_state'].iloc[0]} -> {df_input['customer_state'].iloc[0]})")
        if df_input["seller_historical_delay_rate"].iloc[0] > 0.15:
            flags.append(f"Underperforming seller (historical delay rate: {df_input['seller_historical_delay_rate'].iloc[0]*100:.1f}%)")
        if df_input["estimated_window_days"].iloc[0] < 12:
            flags.append(f"Aggressive SLA window ({df_input['estimated_window_days'].iloc[0]:.0f} days)")

        return {
            "delay_probability": round(prob, 4),
            "delay_probability_pct": round(prob * 100, 1),
            "is_delayed_predicted": int(prob >= self.optimal_threshold),
            "risk_level": risk_level,
            "risk_color": risk_color,
            "operational_action": action,
            "risk_flags": flags if flags else ["Normal routing profile"],
            "optimal_threshold": self.optimal_threshold,
        }


# Quick test helper
if __name__ == "__main__":
    predictor = DeliveryDelayPredictor()
    test_order = {
        "distance_km": 1600.0,
        "freight_value": 45.0,
        "price": 120.0,
        "product_weight_kg": 4.5,
        "product_volume_liters": 25.0,
        "estimated_window_days": 10.0,
        "approval_delay_hours": 12.0,
        "carrier_handoff_lag_days": 5.0,
        "seller_historical_delay_rate": 0.22,
        "customer_state": "BA",
        "seller_state": "SP",
        "purchase_dow": 4,
        "purchase_month": 11,
        "is_holiday_season": 1,
    }
    result = predictor.predict(test_order)
    print("Inference Test Result:")
    print(result)
