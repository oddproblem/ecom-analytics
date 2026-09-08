"""
End-to-End Data Pipeline & Feature Store Generator
Loads raw e-commerce CSVs, computes geospatial distances, operational lead times,
and produces analytical feature stores for model training and Streamlit dashboard.

Author: oddproblem (https://github.com/oddproblem/ecom-analytics)
"""

import math
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import (
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    PROCESSED_ORDERS_PATH,
    PROCESSED_KPIS_PATH,
    STATE_LOGISTICS_PATH,
)


def haversine_np(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees) using vectorized NumPy.
    """
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    km = 6367 * c
    return km


def load_raw_datasets():
    """Load raw CSV datasets from data/raw/."""
    print("[INFO] Loading raw CSV files...")
    orders = pd.read_csv(DATA_RAW_DIR / "olist_orders_dataset.csv")
    items = pd.read_csv(DATA_RAW_DIR / "olist_order_items_dataset.csv")
    customers = pd.read_csv(DATA_RAW_DIR / "olist_customers_dataset.csv")
    sellers = pd.read_csv(DATA_RAW_DIR / "olist_sellers_dataset.csv")
    products = pd.read_csv(DATA_RAW_DIR / "olist_products_dataset.csv")
    reviews = pd.read_csv(DATA_RAW_DIR / "olist_order_reviews_dataset.csv")
    translations = pd.read_csv(DATA_RAW_DIR / "product_category_name_translation.csv")
    
    # Geolocation: deduplicate by zipcode prefix using median coordinates
    print("[INFO] Processing geolocation data...")
    geo = pd.read_csv(DATA_RAW_DIR / "olist_geolocation_dataset.csv")
    geo_clean = (
        geo.groupby("geolocation_zip_code_prefix")
        .agg(
            lat=("geolocation_lat", "median"),
            lng=("geolocation_lng", "median"),
        )
        .reset_index()
    )

    return orders, items, customers, sellers, products, reviews, translations, geo_clean


def build_analytical_feature_store():
    """Build unified analytical dataset and feature store."""
    (
        orders,
        items,
        customers,
        sellers,
        products,
        reviews,
        translations,
        geo_clean,
    ) = load_raw_datasets()

    print("[INFO] Translating product categories...")
    products = products.merge(translations, on="product_category_name", how="left")
    products["category_english"] = products["product_category_name_english"].fillna(
        products["product_category_name"].fillna("other")
    )

    # Convert timestamps
    ts_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for c in ts_cols:
        orders[c] = pd.to_datetime(orders[c])

    # Filter to delivered orders with valid delivery and estimated dates
    delivered_orders = orders[
        (orders["order_status"] == "delivered")
        & orders["order_delivered_customer_date"].notna()
        & orders["order_estimated_delivery_date"].notna()
    ].copy()

    # Aggregate item level to order level (primary product, sums, item counts)
    print("[INFO] Aggregating item & product features...")
    items_enriched = items.merge(
        products[
            [
                "product_id",
                "category_english",
                "product_weight_g",
                "product_length_cm",
                "product_height_cm",
                "product_width_cm",
            ]
        ],
        on="product_id",
        how="left",
    )

    # Product volume in cm3
    items_enriched["product_volume_cm3"] = (
        items_enriched["product_length_cm"].fillna(20)
        * items_enriched["product_height_cm"].fillna(15)
        * items_enriched["product_width_cm"].fillna(15)
    )
    items_enriched["product_weight_g"] = items_enriched["product_weight_g"].fillna(1000)

    order_items_agg = (
        items_enriched.groupby("order_id")
        .agg(
            seller_id=("seller_id", "first"),
            items_count=("order_item_id", "count"),
            price=("price", "sum"),
            freight_value=("freight_value", "sum"),
            product_weight_g=("product_weight_g", "sum"),
            product_volume_cm3=("product_volume_cm3", "sum"),
            primary_category=("category_english", "first"),
        )
        .reset_index()
    )

    # Aggregate reviews to order level
    reviews_agg = (
        reviews.groupby("order_id")
        .agg(
            review_score=("review_score", "mean"),
            has_comment=("review_comment_message", lambda s: int(s.notna().any())),
        )
        .reset_index()
    )

    # Merge everything together
    print("[INFO] Merging orders, items, customers, sellers, and reviews...")
    df = delivered_orders.merge(order_items_agg, on="order_id", how="inner")
    df = df.merge(
        customers[["customer_id", "customer_unique_id", "customer_zip_code_prefix", "customer_city", "customer_state"]],
        on="customer_id",
        how="left",
    )
    df = df.merge(
        sellers[["seller_id", "seller_zip_code_prefix", "seller_city", "seller_state"]],
        on="seller_id",
        how="left",
    )
    df = df.merge(reviews_agg, on="order_id", how="left")
    df["review_score"] = df["review_score"].fillna(df["review_score"].median())

    # Merge geolocation for customer and seller
    df = df.merge(
        geo_clean.rename(
            columns={
                "geolocation_zip_code_prefix": "customer_zip_code_prefix",
                "lat": "customer_lat",
                "lng": "customer_lng",
            }
        ),
        on="customer_zip_code_prefix",
        how="left",
    )
    df = df.merge(
        geo_clean.rename(
            columns={
                "geolocation_zip_code_prefix": "seller_zip_code_prefix",
                "lat": "seller_lat",
                "lng": "seller_lng",
            }
        ),
        on="seller_zip_code_prefix",
        how="left",
    )

    # Compute Haversine distance
    df["distance_km"] = haversine_np(
        df["seller_lat"].values,
        df["seller_lng"].values,
        df["customer_lat"].values,
        df["customer_lng"].values,
    )
    # Fill missing distances with regional median or heuristic (approx 600km)
    median_dist = df["distance_km"].dropna().median()
    df["distance_km"] = df["distance_km"].fillna(median_dist).clip(lower=5.0, upper=4000.0)

    # Compute operational time intervals
    print("[INFO] Engineering temporal & delivery metrics...")
    df["actual_delivery_days"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 3600)

    df["estimated_window_days"] = (
        df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 3600)

    # Approval delay
    df["approval_delay_hours"] = (
        df["order_approved_at"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600
    df["approval_delay_hours"] = df["approval_delay_hours"].fillna(0.5).clip(lower=0.0, upper=168.0)

    # Carrier handoff lag
    df["carrier_handoff_lag_days"] = (
        df["order_delivered_carrier_date"] - df["order_purchase_timestamp"]
    ).dt.total_seconds() / (24 * 3600)
    df["carrier_handoff_lag_days"] = (
        df["carrier_handoff_lag_days"].fillna(df["carrier_handoff_lag_days"].median()).clip(lower=0.1, upper=30.0)
    )

    # Amazon Target: Is the delivery delayed past estimated date?
    df["delay_days"] = (
        df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
    ).dt.total_seconds() / (24 * 3600)
    df["is_delayed"] = (df["delay_days"] > 0).astype(int)

    # Derived physical & economic features
    df["product_weight_kg"] = (df["product_weight_g"] / 1000.0).round(3)
    df["product_volume_liters"] = (df["product_volume_cm3"] / 1000.0).round(3)
    df["freight_ratio"] = (df["freight_value"] / (df["price"] + 1.0)).round(4)
    df["is_same_state"] = (df["customer_state"] == df["seller_state"]).astype(int)

    # Temporal calendar features
    df["purchase_dow"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour
    df["is_holiday_season"] = df["purchase_month"].isin([11, 12]).astype(int)
    df["payment_installments"] = 1  # default baseline

    # Historical seller SLA reliability rate (leave-one-out or expanding mean)
    seller_delay_stats = (
        df.groupby("seller_id")["is_delayed"]
        .agg(seller_orders="count", seller_delay_sum="sum")
        .reset_index()
    )
    # Smoothed Bayesian delay rate: (delays + 1) / (orders + 10)
    seller_delay_stats["seller_historical_delay_rate"] = (
        (seller_delay_stats["seller_delay_sum"] + 1)
        / (seller_delay_stats["seller_orders"] + 15)
    ).round(4)

    df = df.merge(
        seller_delay_stats[["seller_id", "seller_historical_delay_rate"]],
        on="seller_id",
        how="left",
    )
    df["seller_historical_delay_rate"] = df["seller_historical_delay_rate"].fillna(0.065)

    # Sort by purchase timestamp
    df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)

    print(f"[OK] Processed {len(df):,} delivered orders. Delay rate: {df['is_delayed'].mean()*100:.2f}%")

    # Generate aggregated state logistics data for interactive geospatial views
    print("[INFO] Building state-level logistics summary...")
    state_logistics = (
        df.groupby(["customer_state"])
        .agg(
            total_orders=("order_id", "count"),
            delayed_orders=("is_delayed", "sum"),
            avg_delivery_days=("actual_delivery_days", "mean"),
            avg_freight=("freight_value", "mean"),
            avg_review=("review_score", "mean"),
            avg_distance_km=("distance_km", "mean"),
        )
        .reset_index()
    )
    state_logistics["delay_rate_pct"] = (
        (state_logistics["delayed_orders"] / state_logistics["total_orders"]) * 100
    ).round(2)

    # Save to compressed Parquet
    print(f"[INFO] Caching processed parquet files to {DATA_PROCESSED_DIR}...")
    df.to_parquet(PROCESSED_ORDERS_PATH, index=False, engine="pyarrow")
    state_logistics.to_parquet(STATE_LOGISTICS_PATH, index=False, engine="pyarrow")

    # Build executive KPIs
    kpi_summary = pd.DataFrame(
        [
            {
                "total_orders": len(df),
                "total_gmv": df["price"].sum() + df["freight_value"].sum(),
                "total_revenue": df["price"].sum(),
                "total_freight": df["freight_value"].sum(),
                "overall_on_time_rate_pct": (1.0 - df["is_delayed"].mean()) * 100,
                "overall_delay_rate_pct": df["is_delayed"].mean() * 100,
                "avg_delivery_days": df["actual_delivery_days"].mean(),
                "avg_estimated_days": df["estimated_window_days"].mean(),
                "avg_order_value": df["price"].mean(),
                "avg_freight_cost": df["freight_value"].mean(),
                "avg_review_score": df["review_score"].mean(),
                "delayed_order_avg_review": df[df["is_delayed"] == 1]["review_score"].mean(),
                "on_time_order_avg_review": df[df["is_delayed"] == 0]["review_score"].mean(),
            }
        ]
    )
    kpi_summary.to_parquet(PROCESSED_KPIS_PATH, index=False, engine="pyarrow")

    print("[SUCCESS] Feature store pipeline completed successfully!")
    return df, state_logistics, kpi_summary


if __name__ == "__main__":
    build_analytical_feature_store()
