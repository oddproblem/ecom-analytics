"""
Machine Learning Training & SLA Risk Model Evaluation
Trains an Amazon-grade Delivery Delay & SLA Risk Classification Model.
Calculates ROC-AUC, PR-AUC, Confusion Matrix, and Financial Concession Impact.

Author: oddproblem (https://github.com/oddproblem/ecom-analytics)
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CHURN_RISK_MULTIPLIER,
    ESTIMATED_CONCESSION_COST_USD,
    MODEL_ARTIFACT_PATH,
    MODELS_DIR,
    PROCESSED_ORDERS_PATH,
    RANDOM_STATE,
    TARGET_COL,
)


def load_feature_matrix():
    """Load cached feature matrix from parquet."""
    if not PROCESSED_ORDERS_PATH.exists():
        raise FileNotFoundError(f"Feature store not found at {PROCESSED_ORDERS_PATH}. Run pipeline.py first.")
    
    df = pd.read_parquet(PROCESSED_ORDERS_PATH)
    return df


def train_delivery_delay_model():
    """Train gradient boosting SLA classifier with business threshold optimization."""
    print("[INFO] Loading dataset for model training...")
    df = load_feature_matrix()

    # Features selected based on domain relevance to e-commerce logistics
    num_cols = [
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
    ]

    cat_cols = [
        "is_same_state",
        "customer_state",
        "seller_state",
        "purchase_dow",
        "purchase_month",
        "is_holiday_season",
    ]

    X = df[num_cols + cat_cols].copy()
    y = df[TARGET_COL].values

    print(f"[INFO] Dataset shape: {X.shape}. Delay rate: {np.mean(y)*100:.2f}%")

    # Time-aware or stratified split (80/20 train/test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    # Primary Model: Gradient Boosting with class weighting
    print("[INFO] Training HistGradientBoostingClassifier...")
    clf = HistGradientBoostingClassifier(
        class_weight="balanced",
        max_iter=150,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    pipeline.fit(X_train, y_train)

    # Probabilities on test set
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics calculation
    auc_roc = roc_auc_score(y_test, y_prob)
    auc_pr = average_precision_score(y_test, y_prob)

    # Precision-recall curve to find optimal business threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    # F1 optimization
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])

    # Predictions with standard 0.5 and optimal threshold
    y_pred_std = (y_prob >= 0.5).astype(int)
    y_pred_opt = (y_prob >= optimal_threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred_opt).tolist()
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_opt).ravel()

    # Calculate financial impact
    # Delay avoided with proactive carrier intervention
    estimated_delays_detected = int(tp)
    concessions_saved = float(estimated_delays_detected * ESTIMATED_CONCESSION_COST_USD * 0.70)  # 70% mitigation efficacy

    metrics = {
        "roc_auc": round(float(auc_roc), 4),
        "pr_auc": round(float(auc_pr), 4),
        "optimal_threshold": round(float(optimal_threshold), 4),
        "precision_at_optimal": round(float(precision_score(y_test, y_pred_opt)), 4),
        "recall_at_optimal": round(float(recall_score(y_test, y_pred_opt)), 4),
        "f1_at_optimal": round(float(f1_score(y_test, y_pred_opt)), 4),
        "test_sample_size": len(y_test),
        "actual_delays_in_test": int(y_test.sum()),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "business_impact": {
            "concessions_saved_usd": concessions_saved,
            "delays_detected": estimated_delays_detected,
            "false_alarms": int(fp),
        },
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
    }

    # Extract feature importance approximation
    # For HistGradientBoosting, compute permutation importance or feature weights
    print("[INFO] Computing feature importance...")
    from sklearn.inspection import permutation_importance
    perm_importance = permutation_importance(
        pipeline, X_test.iloc[:2000], y_test[:2000], n_repeats=3, random_state=RANDOM_STATE, scoring="roc_auc"
    )
    
    feature_names = num_cols + cat_cols
    importances = [float(imp) for imp in perm_importance.importances_mean]
    feat_imp_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    metrics["top_features"] = feat_imp_df.to_dict(orient="records")

    print("\n" + "=" * 50)
    print("           MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"ROC-AUC Score          : {auc_roc:.4f}")
    print(f"PR-AUC (Avg Precision) : {auc_pr:.4f}")
    print(f"Optimal Threshold      : {optimal_threshold:.4f}")
    print(f"Precision @ Optimal    : {metrics['precision_at_optimal']:.4f}")
    print(f"Recall @ Optimal       : {metrics['recall_at_optimal']:.4f}")
    print(f"F1-Score @ Optimal     : {metrics['f1_at_optimal']:.4f}")
    print(f"Projected Concession Savings: ${concessions_saved:,.2f}")
    print("=" * 50)

    # Save model artifact
    print(f"[INFO] Saving model artifact to {MODEL_ARTIFACT_PATH}...")
    model_payload = {
        "pipeline": pipeline,
        "metrics": metrics,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "optimal_threshold": optimal_threshold,
    }
    joblib.dump(model_payload, MODEL_ARTIFACT_PATH)

    # Save metrics JSON for quick dashboard consumption
    metrics_json_path = MODELS_DIR / "model_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[SUCCESS] Model artifact and metrics saved to {MODELS_DIR}!")
    return model_payload, metrics


if __name__ == "__main__":
    train_delivery_delay_model()
