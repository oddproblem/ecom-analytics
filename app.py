"""
Amazon Fulfillment & Delivery SLA Intelligence Platform
Interactive Streamlit Web Application

Author: oddproblem (https://github.com/oddproblem/ecom-analytics)
"""

import os
import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    PROCESSED_ORDERS_PATH,
    PROCESSED_KPIS_PATH,
    STATE_LOGISTICS_PATH,
    MODEL_ARTIFACT_PATH,
    ESTIMATED_CONCESSION_COST_USD,
)
from src.predictor import DeliveryDelayPredictor
from src.llm_advisor import generate_llm_analysis

# Page Configuration
st.set_page_config(
    page_title="Amazon Delivery SLA Intelligence | oddproblem",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Amazon Dark/Navy Theme & Polished Typography
st.markdown(
    """
    <style>
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2.0rem;
        font-weight: 700;
        color: #FF9900;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.95rem;
        font-weight: 600;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 15px;
    }
    .badge-author {
        display: inline-block;
        background-color: #232F3E;
        color: #FF9900;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #FF9900;
        margin-bottom: 12px;
    }
    .status-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .risk-critical { background-color: rgba(239, 68, 68, 0.2); color: #EF4444; border: 1px solid #EF4444; }
    .risk-elevated { background-color: rgba(249, 115, 22, 0.2); color: #F97316; border: 1px solid #F97316; }
    .risk-moderate { background-color: rgba(245, 158, 11, 0.2); color: #F59E0B; border: 1px solid #F59E0B; }
    .risk-low { background-color: rgba(16, 185, 129, 0.2); color: #10B981; border: 1px solid #10B981; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_app_data():
    """Load cached analytical feature store and metrics."""
    df_orders = pd.read_parquet(PROCESSED_ORDERS_PATH)
    df_kpis = pd.read_parquet(PROCESSED_KPIS_PATH)
    df_state = pd.read_parquet(STATE_LOGISTICS_PATH)
    with open(BASE_DIR / "models" / "model_metrics.json", "r") as f:
        metrics = json.load(f)
    return df_orders, df_kpis, df_state, metrics


@st.cache_resource(show_spinner=False)
def get_predictor():
    return DeliveryDelayPredictor()


# Load Data
try:
    df_orders, df_kpis, df_state, metrics = load_app_data()
    predictor = get_predictor()
except Exception as e:
    st.error(f"Error loading data pipeline artifacts: {e}. Please run `python -m src.pipeline` and `python -m src.model` first.")
    st.stop()

# Top Header
col_title, col_badge = st.columns([3, 1])
with col_title:
    st.markdown('<span class="badge-author">AMAZON OPERATIONS & DATA SCIENCE PORTFOLIO</span>', unsafe_allow_html=True)
    st.title("📦 Fulfillment & Delivery SLA Intelligence Platform")
    st.caption("Predictive Machine Learning, Operations Research & Customer Defect Mitigation Engine | Dataset: 96,470 Real Shipments")

with col_badge:
    st.markdown(
        """
        <div style="text-align: right; margin-top: 10px;">
            <a href="https://github.com/oddproblem/ecom-analytics" target="_blank" style="text-decoration: none;">
                <button style="background-color: #232F3E; color: #FFFFFF; border: 1px solid #FF9900; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-weight: 600;">
                    ⭐ GitHub Repository
                </button>
            </a>
            <div style="font-size: 0.8rem; color: #9CA3AF; margin-top: 6px;">Developed by <b>oddproblem</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=130)
st.sidebar.markdown("### ⚙️ Platform Controls")

api_key_input = st.sidebar.text_input(
    "OpenAI / Gemini API Key (Optional)",
    type="password",
    help="Enter an API key to enable AI-powered Root Cause Generation, or leave blank to use the built-in Amazon Logistics Expert Heuristics engine.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Model Health Monitor")
st.sidebar.markdown(f"**Algorithm:** `HistGradientBoosting`")
st.sidebar.markdown(f"**ROC-AUC Score:** `{metrics['roc_auc']:.4f}`")
st.sidebar.markdown(f"**PR-AUC Score:** `{metrics['pr_auc']:.4f}`")
st.sidebar.markdown(f"**Optimal Threshold:** `{metrics['optimal_threshold']:.4f}`")
st.sidebar.markdown(f"**Test Sample:** `{metrics['test_sample_size']:,} orders`")
st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Amazon DS Core Competencies** demonstrated: Class-imbalanced predictive ML, Geospatial Haversine calculation, Cost-sensitive threshold tuning, and Real-time sub-50ms inference."
)

# Navigation Tabs
tab_exec, tab_sim, tab_geo, tab_xai, tab_copilot = st.tabs(
    [
        "📊 Executive Command Center",
        "🎯 Live SLA Risk Predictor",
        "🗺️ Geographic Bottlenecks",
        "🧠 Model Transparency & XAI",
        "🤖 AI Fulfillment Copilot",
    ]
)

# ====================================================================
# TAB 1: EXECUTIVE COMMAND CENTER
# ====================================================================
with tab_exec:
    st.markdown("### 📈 Fulfillment & Customer Satisfaction Performance")
    
    kpis = df_kpis.iloc[0]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Delivered Volume", f"{int(kpis['total_orders']):,}")
    with col2:
        st.metric("Total Gross Merchandise", f"${kpis['total_gmv']/1e6:.2f}M")
    with col3:
        st.metric("On-Time SLA Rate", f"{kpis['overall_on_time_rate_pct']:.1f}%", delta="Target: 95.0%")
    with col4:
        st.metric("Avg Delivery Window", f"{kpis['avg_delivery_days']:.1f} Days", delta="-11.4 Days vs Promise")
    with col5:
        st.metric("Avg Review Score", f"{kpis['avg_review_score']:.2f} ★", delta="Defect Threshold: 3.0★")

    st.markdown("---")

    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        st.markdown("#### 🚨 The Defect Cost: Delay Impact on Review Ratings")
        st.caption("Customer Obsession Analysis: Late shipments cause a devastating drop in customer ratings.")
        
        review_data = pd.DataFrame({
            "Delivery Status": ["On-Time Delivery", "Late Delivery (SLA Breach)"],
            "Average Rating": [kpis['on_time_order_avg_review'], kpis['delayed_order_avg_review']],
            "1-Star Defect Rate (%)": [
                (df_orders[df_orders["is_delayed"] == 0]["review_score"] == 1).mean() * 100,
                (df_orders[df_orders["is_delayed"] == 1]["review_score"] == 1).mean() * 100,
            ]
        })
        
        fig_rating = go.Figure()
        fig_rating.add_trace(go.Bar(
            x=review_data["Delivery Status"],
            y=review_data["Average Rating"],
            name="Average Star Rating",
            marker_color=["#10B981", "#EF4444"],
            text=[f"{v:.2f} ★" for v in review_data["Average Rating"]],
            textposition="auto",
        ))
        fig_rating.update_layout(
            yaxis=dict(title="Star Rating (1-5)", range=[0, 5]),
            template="plotly_dark",
            height=320,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_rating, width="stretch")
        st.warning(
            f"⚠️ **Amazon Customer Insight**: Late deliveries experience a **{(review_data['1-Star Defect Rate (%)'].iloc[1] / review_data['1-Star Defect Rate (%)'].iloc[0]):.1f}x surge in 1-star defect reviews**, resulting in customer churn and costly concession credits."
        )

    with col_chart2:
        st.markdown("#### 📅 Monthly Order Volume vs On-Time Reliability")
        st.caption("Tracking operational throughput and SLA adherence over historical quarters.")
        
        df_orders["purchase_year_month"] = df_orders["order_purchase_timestamp"].dt.to_period("M").astype(str)
        monthly_trends = (
            df_orders.groupby("purchase_year_month")
            .agg(
                orders_count=("order_id", "count"),
                on_time_pct=("is_delayed", lambda x: (1.0 - x.mean()) * 100),
            )
            .reset_index()
        )
        monthly_trends = monthly_trends.iloc[3:-1]  # filter early startup and incomplete final months

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=monthly_trends["purchase_year_month"],
            y=monthly_trends["orders_count"],
            name="Delivered Orders",
            marker_color="#2563EB",
            yaxis="y",
            opacity=0.6,
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly_trends["purchase_year_month"],
            y=monthly_trends["on_time_pct"],
            name="On-Time Rate (%)",
            marker_color="#FF9900",
            yaxis="y2",
            mode="lines+markers",
            line=dict(width=3),
        ))
        fig_trend.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title="Orders Volume"),
            yaxis2=dict(title="On-Time SLA %", overlaying="y", side="right", range=[70, 100]),
        )
        st.plotly_chart(fig_trend, use_container_width=True)
        st.info("💡 **Peak Season Vulnerability**: Black Friday and holiday Q4 order surges stress regional line-haul corridors, causing on-time rates to dip.")


# ====================================================================
# TAB 2: LIVE SLA RISK PREDICTOR & SIMULATOR
# ====================================================================
with tab_sim:
    st.markdown("### 🎯 Real-Time Amazon Delivery SLA Risk Simulator")
    st.markdown(
        "Simulate any customer shipment at point of checkout or seller handoff. Our ML model evaluates cross-state transit, product bulkiness, and seller reliability to compute real-time delay probability and recommend tactical operational mitigations."
    )

    col_input1, col_input2, col_input3 = st.columns(3)

    states = sorted(df_orders["customer_state"].unique().tolist())

    with col_input1:
        st.markdown("##### 📍 Geographic Corridor")
        seller_state = st.selectbox("Seller Fulfillment Hub State", states, index=states.index("SP"))
        customer_state = st.selectbox("Customer Destination State", states, index=states.index("RJ"))
        
        is_same = int(seller_state == customer_state)
        default_dist = 80.0 if is_same else 480.0
        distance_km = st.slider("Estimated Transit Distance (km)", 10, 3500, int(default_dist), step=25)

    with col_input2:
        st.markdown("##### 📦 Package & Economics")
        price = st.number_input("Order Item Value ($)", min_value=5.0, max_value=5000.0, value=95.0, step=5.0)
        freight_value = st.number_input("Freight Charge ($)", min_value=2.0, max_value=500.0, value=18.50, step=1.0)
        weight_kg = st.slider("Package Weight (kg)", 0.1, 30.0, 1.5, step=0.1)
        volume_l = st.slider("Product Volume (Liters)", 0.5, 100.0, 8.0, step=0.5)

    with col_input3:
        st.markdown("##### ⏱️ Operations & Lead Times")
        estimated_window = st.slider("Promised SLA Window (Days)", 3, 45, 18, step=1)
        carrier_lag = st.slider("Seller Dispatch Lead Time (Days to Carrier)", 0.2, 15.0, 2.0, step=0.2)
        seller_sla_rate = st.slider("Seller Historical Late Rate (%)", 0.0, 50.0, 6.5, step=0.5) / 100.0
        is_holiday = st.selectbox("Seasonal Rush Period?", ["Standard Window", "Holiday / Black Friday Season"], index=0)

    # Prepare input payload
    sim_order = {
        "distance_km": float(distance_km),
        "freight_value": float(freight_value),
        "price": float(price),
        "product_weight_kg": float(weight_kg),
        "product_volume_liters": float(volume_l),
        "estimated_window_days": float(estimated_window),
        "approval_delay_hours": 2.0,
        "carrier_handoff_lag_days": float(carrier_lag),
        "seller_historical_delay_rate": float(seller_sla_rate),
        "customer_state": customer_state,
        "seller_state": seller_state,
        "purchase_dow": 2,
        "purchase_month": 11 if "Holiday" in is_holiday else 5,
        "is_holiday_season": 1 if "Holiday" in is_holiday else 0,
        "items_count": 1,
    }

    st.markdown("---")
    res = predictor.predict(sim_order)

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.markdown("#### 📊 Prediction Scorecard")
        
        # Risk color mapping
        risk_class = f"risk-{res['risk_level'].lower()}"
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; color: #9CA3AF;">PREDICTED DELAY PROBABILITY</div>
                <div style="font-size: 2.8rem; font-weight: 800; color: {res['risk_color']};">{res['delay_probability_pct']}%</div>
                <div style="margin-top: 6px;">
                    <span class="status-pill {risk_class}">SLA RISK: {res['risk_level']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### 🔍 Operational Risk Flags Detected:")
        for flag in res["risk_flags"]:
            st.markdown(f"• **{flag}**")

    with res_col2:
        st.markdown("#### 🛠️ Amazon Operations Tactical Mitigation")
        st.info(f"**Action Recommended:**\n\n{res['operational_action']}")

        # Financial Concession Exposure
        expected_concession_cost = (res["delay_probability"] * ESTIMATED_CONCESSION_COST_USD)
        savings_if_mitigated = expected_concession_cost * 0.70

        st.markdown(
            f"""
            <div style="background-color: #1A1F2C; border-left: 4px solid #FF9900; padding: 12px 16px; border-radius: 4px; margin-top: 15px;">
                <div style="font-size: 0.85rem; color: #9CA3AF;">FINANCIAL EXPOSURE & CONCESSION AVOIDANCE</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: #F3F4F6;">
                    Estimated Appeasement Exposure: <span style="color: #EF4444;">${expected_concession_cost:.2f}</span>
                </div>
                <div style="font-size: 0.95rem; color: #10B981; margin-top: 4px;">
                    Projected Savings via Proactive Intervention: <b>${savings_if_mitigated:.2f}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ====================================================================
# TAB 3: GEOGRAPHIC LOGISTICS & BOTTLENECKS
# ====================================================================
with tab_geo:
    st.markdown("### 🗺️ Geographic Logistics & Regional Bottlenecks")
    st.caption("Pinpoint structural delivery delays and high freight cost regions to prioritize regional fulfillment expansion.")

    col_geo1, col_geo2 = st.columns([1, 1])

    with col_geo1:
        st.markdown("#### 🚨 Top 10 Highest SLA Delay Rate States")
        top_delay_states = df_state.sort_values("delay_rate_pct", ascending=False).head(10)
        
        fig_delays = px.bar(
            top_delay_states,
            x="customer_state",
            y="delay_rate_pct",
            color="delay_rate_pct",
            color_continuous_scale="Reds",
            labels={"customer_state": "Destination State", "delay_rate_pct": "Delay Rate (%)"},
            text="delay_rate_pct",
        )
        fig_delays.update_layout(
            template="plotly_dark",
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_delays, use_container_width=True)
        st.caption("Destinations in Northeast and North (e.g., AL, MA, SE) suffer from multi-leg inter-state transit with delays exceeding 15%.")

    with col_geo2:
        st.markdown("#### 🚚 Delivery Transit Days vs Average Freight Cost")
        fig_scatter = px.scatter(
            df_state,
            x="avg_delivery_days",
            y="avg_freight",
            size="total_orders",
            color="delay_rate_pct",
            color_continuous_scale="Viridis",
            hover_name="customer_state",
            labels={
                "avg_delivery_days": "Avg Transit Days",
                "avg_freight": "Avg Freight Cost ($)",
                "delay_rate_pct": "Late Rate %",
            },
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Bubble size indicates order volume. Hub state SP enjoys lowest transit days (8.3 days) and freight ($15.10).")


# ====================================================================
# TAB 4: MODEL TRANSPARENCY & EXPLAINABILITY (XAI)
# ====================================================================
with tab_xai:
    st.markdown("### 🧠 Machine Learning Architecture & Feature Importance")
    st.caption("Rigorous evaluation, class-imbalance treatment, and feature attributions.")

    col_xai1, col_xai2 = st.columns([1, 1])

    with col_xai1:
        st.markdown("#### 🏆 Key Feature Attributions (Permutation Importance)")
        feat_df = pd.DataFrame(metrics["top_features"]).head(10)
        
        # Nicer names
        name_map = {
            "estimated_window_days": "Promised SLA Window (Days)",
            "purchase_month": "Seasonality (Month of Year)",
            "carrier_handoff_lag_days": "Seller Dispatch Delay (Days)",
            "seller_historical_delay_rate": "Seller Historical Late Rate",
            "customer_state": "Customer Destination State",
            "distance_km": "Haversine Distance (km)",
            "is_same_state": "Intra-State Shipping Flag",
            "items_count": "Items in Package",
            "freight_value": "Freight Cost ($)",
            "approval_delay_hours": "Order Approval Delay (Hours)",
        }
        feat_df["feature_name"] = feat_df["feature"].map(name_map).fillna(feat_df["feature"])

        fig_feat = px.bar(
            feat_df.sort_values("importance", ascending=True),
            x="importance",
            y="feature_name",
            orientation="h",
            labels={"importance": "Permutation Importance (AUC Degradation)", "feature_name": ""},
            color="importance",
            color_continuous_scale="Oranges",
        )
        fig_feat.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_feat, use_container_width=True)

    with col_xai2:
        st.markdown("#### 🎯 Confusion Matrix @ Optimal Threshold")
        cm = metrics["confusion_matrix"]
        
        cm_data = [
            [cm["tn"], cm["fp"]],
            [cm["fn"], cm["tp"]],
        ]
        
        fig_cm = px.imshow(
            cm_data,
            text_auto=True,
            x=["Predicted On-Time", "Predicted Delay"],
            y=["Actual On-Time", "Actual Delay"],
            color_continuous_scale="Blues",
            labels=dict(x="Model Decision", y="Ground Truth", color="Shipments"),
        )
        fig_cm.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.markdown("#### ⚙️ Interactive Threshold Tuning & Cost Curve Simulator")
    threshold_slider = st.slider(
        "Decision Threshold Tuning (Simulate Precision vs Recall Trade-Off)",
        min_value=0.10,
        max_value=0.90,
        value=float(metrics["optimal_threshold"]),
        step=0.05,
    )
    st.caption(
        f"Selected Threshold: **{threshold_slider:.2f}** | At optimal business threshold **{metrics['optimal_threshold']:.2f}**, the model captures **{metrics['recall_at_optimal']*100:.1f}% of all delayed shipments** while maintaining **{metrics['precision_at_optimal']*100:.1f}% precision**, yielding **${metrics['business_impact']['concessions_saved_usd']:,.2f}** in projected concession savings on the test set alone."
    )


# ====================================================================
# TAB 5: AI FULFILLMENT COPILOT
# ====================================================================
with tab_copilot:
    st.markdown("### 🤖 AI Supply Chain & Root-Cause Analyst")
    st.caption("Generate executive-ready root-cause memorandums and carrier negotiation briefs for critical shipments.")

    col_co1, col_co2 = st.columns([1, 2])

    with col_co1:
        st.markdown("##### 📋 Select Critical Shipment")
        high_risk_orders = df_orders[df_orders["is_delayed"] == 1].head(15)
        selected_idx = st.selectbox(
            "Select an order from historical delayed cohort:",
            range(len(high_risk_orders)),
            format_func=lambda i: f"Order #{high_risk_orders.iloc[i]['order_id'][:8]}... | {high_risk_orders.iloc[i]['seller_state']} -> {high_risk_orders.iloc[i]['customer_state']} ({high_risk_orders.iloc[i]['delay_days']:.0f} days late)",
        )
        target_row = high_risk_orders.iloc[selected_idx].to_dict()

        generate_btn = st.button("🚀 Generate Executive Root-Cause Report", type="primary")

    with col_co2:
        if generate_btn:
            with st.spinner("Analyzing fulfillment telemetry and querying AI Operations Advisor..."):
                target_pred = predictor.predict(target_row)
                report_md = generate_llm_analysis(target_row, target_pred, api_key=api_key_input)
                st.markdown(report_md)
        else:
            st.info("👈 Select a delayed order and click **Generate Executive Root-Cause Report** to see AI synthesis.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #6B7280; font-size: 0.85rem;">
        Amazon Data Science Portfolio Project | Designed & Developed by <b>oddproblem</b> | 
        <a href="https://github.com/oddproblem/ecom-analytics" target="_blank" style="color: #FF9900; text-decoration: none;">GitHub Repository</a>
    </div>
    """,
    unsafe_allow_html=True,
)
