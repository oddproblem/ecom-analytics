"""
Amazon Supply Chain & Fulfillment SLA Intelligence Console
Enterprise Operational Decision Support System

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

# Page Configuration - Clean title, standard favicon
st.set_page_config(
    page_title="Amazon Supply Chain SLA Intelligence Console",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Enterprise CSS - Strict flat design, zero glowing gradients, zero glassmorphism
st.markdown(
    """
    <style>
    /* Meta tags & Layout Constraints */
    html, body {
        overflow-x: hidden;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Global Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Metric Values */
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem;
        font-weight: 600;
        color: #FF9900;
        font-feature-settings: "tnum";
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.82rem;
        font-weight: 600;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Enterprise Metric Card (Flat, neutral border, no drop shadow, no gradient) */
    .metric-card {
        background-color: #18202C;
        border: 1px solid #283344;
        border-radius: 4px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }

    /* System Header and Breadcrumbs */
    .header-container {
        border-bottom: 1px solid #283344;
        padding-bottom: 14px;
        margin-bottom: 20px;
    }
    .system-breadcrumb {
        font-size: 0.78rem;
        font-weight: 600;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 4px;
    }
    .system-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F3F4F6;
        margin: 0;
    }
    .system-subtitle {
        font-size: 0.9rem;
        color: #9CA3AF;
        margin-top: 4px;
    }

    /* Status Badges (Flat, high contrast, standard enterprise radiuses) */
    .status-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 3px;
        font-weight: 600;
        font-size: 0.75rem;
        letter-spacing: 0.4px;
        text-transform: uppercase;
    }
    .status-critical { background-color: #7F1D1D; color: #FCA5A5; border: 1px solid #B91C1C; }
    .status-elevated { background-color: #7C2D12; color: #FDBA74; border: 1px solid #C2410C; }
    .status-moderate { background-color: #78350F; color: #FDE68A; border: 1px solid #D97706; }
    .status-low { background-color: #064E3B; color: #A7F3D0; border: 1px solid #059669; }

    /* Footer Container */
    .footer-container {
        border-top: 1px solid #283344;
        padding-top: 18px;
        margin-top: 40px;
        font-size: 0.8rem;
        color: #9CA3AF;
    }
    .footer-container a {
        color: #FF9900;
        text-decoration: none;
    }
    .footer-container a:hover {
        text-decoration: underline;
    }
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
    st.error(f"System Error: Failed to load data pipeline artifacts: {e}. Execute 'python -m src.pipeline' and 'python -m src.model'.")
    st.stop()

# Header Section
st.markdown(
    """
    <div class="header-container">
        <div class="system-breadcrumb">AWS Supply Chain / Fulfillment Operations / SLA Risk Monitoring</div>
        <div style="display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap;">
            <div>
                <h1 class="system-title">Fulfillment & Delivery SLA Intelligence Console</h1>
                <div class="system-subtitle">Predictive Machine Learning, Operations Research & Defect Mitigation Engine | Partition: 96,470 Verified Shipments</div>
            </div>
            <div style="text-align: right; margin-top: 8px;">
                <a href="https://github.com/oddproblem/ecom-analytics" target="_blank" style="text-decoration: none;">
                    <button style="background-color: #18202C; color: #F3F4F6; border: 1px solid #384656; padding: 6px 14px; border-radius: 4px; cursor: pointer; font-size: 0.85rem; font-weight: 600;">
                        Source Repository
                    </button>
                </a>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Resolve API key from Streamlit secrets or environment (never from UI)
API_KEY = None
try:
    API_KEY = (
        st.secrets.get("OPENROUTER_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
except Exception:
    API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")

_ai_available = bool(API_KEY)

# Sidebar Operational Telemetry
st.sidebar.markdown("### Operational Telemetry")
ai_status = "AI Analysis: Active (DeepSeek)" if _ai_available else "AI Analysis: Offline (Heuristics)"
st.sidebar.markdown(f"- **System Status:** `ONLINE`")
st.sidebar.markdown(f"- **{ai_status}**")
st.sidebar.markdown(f"- **Classifier:** `HistGradientBoosting`")
st.sidebar.markdown(f"- **Validation ROC-AUC:** `{metrics['roc_auc']:.4f}`")
st.sidebar.markdown(f"- **Precision-Recall AUC:** `{metrics['pr_auc']:.4f}`")
st.sidebar.markdown(f"- **Decision Threshold:** `{metrics['optimal_threshold']:.4f}`")
st.sidebar.markdown(f"- **Test Partition:** `{metrics['test_sample_size']:,} shipments`")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Supervised gradient boosting with class-imbalanced weighting and cost-sensitive threshold optimization."
)


# Navigation Tabs (Clean labels, zero emojis)
tab_exec, tab_sim, tab_geo, tab_xai, tab_copilot = st.tabs(
    [
        "Executive Overview",
        "SLA Risk Simulation",
        "Geographic Corridors",
        "Model Evaluation & XAI",
        "Root Cause Diagnostic",
    ]
)

# ====================================================================
# TAB 1: EXECUTIVE OVERVIEW
# ====================================================================
with tab_exec:
    st.markdown("### Fulfillment Performance & Customer Satisfaction")
    
    kpis = df_kpis.iloc[0]
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Delivered Volume", f"{int(kpis['total_orders']):,}")
    with col2:
        st.metric("Gross Merchandise Value", f"${kpis['total_gmv']/1e6:.2f}M")
    with col3:
        st.metric("On-Time SLA Rate", f"{kpis['overall_on_time_rate_pct']:.1f}%", delta="Target: 95.0%")
    with col4:
        st.metric("Avg Delivery Window", f"{kpis['avg_delivery_days']:.1f} Days", delta="-11.4 vs Promise")
    with col5:
        st.metric("Avg Customer Rating", f"{kpis['avg_review_score']:.2f} / 5.0", delta="Defect: < 3.0")

    st.markdown("---")

    col_chart1, col_chart2 = st.columns([1, 1])

    with col_chart1:
        st.markdown("#### Defect Rate Analysis: Late Delivery Impact on Review Scores")
        st.caption("Empirical defect measurement: Orders breaching SLA experience severe customer rating degradation.")
        
        review_data = pd.DataFrame({
            "Delivery Status": ["On-Time Fulfillment", "SLA Breach (Late Delivery)"],
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
            text=[f"{v:.2f} / 5.0" for v in review_data["Average Rating"]],
            textposition="auto",
        ))
        fig_rating.update_layout(
            yaxis=dict(title="Rating (Scale 1.0 to 5.0)", range=[0, 5]),
            template="plotly_dark",
            height=320,
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor="#18202C",
            paper_bgcolor="#0F141C",
        )
        st.plotly_chart(fig_rating, width="stretch")
        st.caption(
            f"Measurement Note: Shipments breaching promised SLA show a **{(review_data['1-Star Defect Rate (%)'].iloc[1] / review_data['1-Star Defect Rate (%)'].iloc[0]):.1f}x increase in 1-star defect reviews**, directly precipitating customer contact concession costs."
        )

    with col_chart2:
        st.markdown("#### Monthly Volume and On-Time Fulfillment Trends")
        st.caption("Operational tracking of line-haul throughput and SLA adherence across quarters.")
        
        df_orders["purchase_year_month"] = df_orders["order_purchase_timestamp"].dt.to_period("M").astype(str)
        monthly_trends = (
            df_orders.groupby("purchase_year_month")
            .agg(
                orders_count=("order_id", "count"),
                on_time_pct=("is_delayed", lambda x: (1.0 - x.mean()) * 100),
            )
            .reset_index()
        )
        monthly_trends = monthly_trends.iloc[3:-1]

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Bar(
            x=monthly_trends["purchase_year_month"],
            y=monthly_trends["orders_count"],
            name="Delivered Shipments",
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
            line=dict(width=2.5),
        ))
        fig_trend.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(title="Shipment Volume"),
            yaxis2=dict(title="On-Time %", overlaying="y", side="right", range=[70, 100]),
            plot_bgcolor="#18202C",
            paper_bgcolor="#0F141C",
        )
        st.plotly_chart(fig_trend, width="stretch")
        st.caption("Seasonal Trend: Peak Q4 order surges place stress on regional sorting hubs, leading to measurable dips in on-time reliability.")


# ====================================================================
# TAB 2: SLA RISK SIMULATION
# ====================================================================
with tab_sim:
    st.markdown("### Real-Time Delivery SLA Risk Simulator")
    st.markdown(
        "Evaluate custom shipment parameters at checkout or first-mile handoff. The trained gradient boosting model computes breach probability and indicates tactical fulfillment mitigations."
    )

    col_input1, col_input2, col_input3 = st.columns(3)

    states = sorted(df_orders["customer_state"].unique().tolist())

    with col_input1:
        st.markdown("##### Logistics Corridor")
        seller_state = st.selectbox("Origin State (Fulfillment Center)", states, index=states.index("SP"))
        customer_state = st.selectbox("Destination State (Delivery Address)", states, index=states.index("RJ"))
        
        is_same = int(seller_state == customer_state)
        default_dist = 80.0 if is_same else 480.0
        distance_km = st.slider("Haversine Distance (km)", 10, 3500, int(default_dist), step=25)

    with col_input2:
        st.markdown("##### Shipment Characteristics & Economics")
        price = st.number_input("Merchandise Price ($)", min_value=5.0, max_value=5000.0, value=95.0, step=5.0)
        freight_value = st.number_input("Freight Surcharge ($)", min_value=2.0, max_value=500.0, value=18.50, step=1.0)
        weight_kg = st.slider("Weight (kg)", 0.1, 30.0, 1.5, step=0.1)
        volume_l = st.slider("Volume (Liters)", 0.5, 100.0, 8.0, step=0.5)

    with col_input3:
        st.markdown("##### Operational Latencies & SLAs")
        estimated_window = st.slider("Promised SLA Window (Days)", 3, 45, 18, step=1)
        carrier_lag = st.slider("Seller Dispatch Lead Time (Days to Carrier)", 0.2, 15.0, 2.0, step=0.2)
        seller_sla_rate = st.slider("Merchant Historical Delay Rate (%)", 0.0, 50.0, 6.5, step=0.5) / 100.0
        is_holiday = st.selectbox("Fulfillment Window Type", ["Standard Operating Period", "Peak Holiday / Q4 Surge"], index=0)

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
        "purchase_month": 11 if "Peak" in is_holiday else 5,
        "is_holiday_season": 1 if "Peak" in is_holiday else 0,
        "items_count": 1,
    }

    st.markdown("---")
    res = predictor.predict(sim_order)

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        st.markdown("#### Inference Output")
        
        status_style = f"status-{res['risk_level'].lower()}"
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size: 0.8rem; color: #9CA3AF; text-transform: uppercase; font-weight: 600;">Predicted SLA Breach Probability</div>
                <div style="font-size: 2.5rem; font-weight: 700; color: {res['risk_color']}; font-feature-settings: 'tnum';">{res['delay_probability_pct']}%</div>
                <div style="margin-top: 6px;">
                    <span class="status-badge {status_style}">SLA Risk Classification: {res['risk_level']}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### Identified Risk Drivers:")
        for flag in res["risk_flags"]:
            st.markdown(f"- {flag}")

    with res_col2:
        st.markdown("#### Prescribed Operational Mitigation")
        st.info(f"{res['operational_action']}")

        # Financial Concession Exposure
        expected_concession_cost = (res["delay_probability"] * ESTIMATED_CONCESSION_COST_USD)
        savings_if_mitigated = expected_concession_cost * 0.70

        st.markdown(
            f"""
            <div style="background-color: #18202C; border: 1px solid #283344; padding: 14px 18px; border-radius: 4px; margin-top: 15px;">
                <div style="font-size: 0.8rem; color: #9CA3AF; text-transform: uppercase; font-weight: 600;">Financial Concession Exposure Analysis</div>
                <div style="font-size: 1.1rem; font-weight: 600; color: #F3F4F6; margin-top: 4px;">
                    Expected Customer Concession Exposure: <span style="color: #EF4444;">${expected_concession_cost:.2f}</span>
                </div>
                <div style="font-size: 0.9rem; color: #10B981; margin-top: 4px;">
                    Projected Savings via Proactive Intervention: <b>${savings_if_mitigated:.2f}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ====================================================================
# TAB 3: GEOGRAPHIC CORRIDORS
# ====================================================================
with tab_geo:
    st.markdown("### Geographic Logistics & Regional Bottlenecks")
    st.caption("Isolating structural transit duration friction and freight surcharges by delivery state.")

    col_geo1, col_geo2 = st.columns([1, 1])

    with col_geo1:
        st.markdown("#### States with Highest SLA Breach Rates")
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
            plot_bgcolor="#18202C",
            paper_bgcolor="#0F141C",
        )
        st.plotly_chart(fig_delays, width="stretch")
        st.caption("Destination states in Northern and Northeastern regions exhibit delay rates exceeding 15% due to hub transit transfers.")

    with col_geo2:
        st.markdown("#### Transit Duration vs. Freight Cost by State")
        fig_scatter = px.scatter(
            df_state,
            x="avg_delivery_days",
            y="avg_freight",
            size="total_orders",
            color="delay_rate_pct",
            color_continuous_scale="Viridis",
            hover_name="customer_state",
            labels={
                "avg_delivery_days": "Mean Transit Days",
                "avg_freight": "Mean Freight ($)",
                "delay_rate_pct": "Late %",
            },
        )
        fig_scatter.update_layout(
            template="plotly_dark",
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="#18202C",
            paper_bgcolor="#0F141C",
        )
        st.plotly_chart(fig_scatter, width="stretch")
        st.caption("Point size corresponds to aggregate order volume. São Paulo (SP) benefits from regional hub density (8.3 days, $15.10 avg freight).")


# ====================================================================
# TAB 4: MODEL EVALUATION & XAI
# ====================================================================
with tab_xai:
    st.markdown("### Machine Learning Architecture & Feature Importance")
    st.caption("Model evaluation, class-imbalance treatment, and feature attributions.")

    col_xai1, col_xai2 = st.columns([1, 1])

    with col_xai1:
        st.markdown("#### Feature Attributions (Permutation Importance on Test Partition)")
        feat_df = pd.DataFrame(metrics["top_features"]).head(10)
        
        name_map = {
            "estimated_window_days": "Promised SLA Window (Days)",
            "purchase_month": "Calendar Month (Seasonality)",
            "carrier_handoff_lag_days": "Seller Dispatch Lead Time",
            "seller_historical_delay_rate": "Merchant Historical Delay Index",
            "customer_state": "Destination State Code",
            "distance_km": "Haversine Distance (km)",
            "is_same_state": "Intra-State Corridor Flag",
            "items_count": "Package Items Count",
            "freight_value": "Freight Cost ($)",
            "approval_delay_hours": "Order Approval Lag (Hours)",
        }
        feat_df["feature_name"] = feat_df["feature"].map(name_map).fillna(feat_df["feature"])

        fig_feat = px.bar(
            feat_df.sort_values("importance", ascending=True),
            x="importance",
            y="feature_name",
            orientation="h",
            labels={"importance": "Permutation Importance (AUC Delta)", "feature_name": ""},
            color="importance",
            color_continuous_scale="Oranges",
        )
        fig_feat.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_showscale=False,
            plot_bgcolor="#18202C",
            paper_bgcolor="#0F141C",
        )
        st.plotly_chart(fig_feat, width="stretch")

    with col_xai2:
        st.markdown(f"#### Confusion Matrix (Optimal Threshold T* = {metrics['optimal_threshold']:.4f})")
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
            labels=dict(x="Model Classification", y="Ground Truth", color="Volume"),
        )
        fig_cm.update_layout(
            template="plotly_dark",
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            coloraxis_showscale=False,
            plot_bgcolor="#18202C",
            paper_bgcolor="#0F141C",
        )
        st.plotly_chart(fig_cm, width="stretch")

    st.markdown("---")
    st.markdown("#### Decision Threshold Tuning & Cost Utility Analysis")
    threshold_slider = st.slider(
        "Operating Threshold Parameter (Precision vs. Recall Control)",
        min_value=0.10,
        max_value=0.90,
        value=float(metrics["optimal_threshold"]),
        step=0.05,
    )
    st.caption(
        f"Selected Operating Point: {threshold_slider:.2f} | At the cost-optimized threshold of {metrics['optimal_threshold']:.4f}, the classifier attains {metrics['recall_at_optimal']*100:.1f}% sensitivity on delayed shipments at {metrics['precision_at_optimal']*100:.1f}% precision, preventing an estimated ${metrics['business_impact']['concessions_saved_usd']:,.2f} in concession costs on the test partition."
    )


# ====================================================================
# TAB 5: ROOT CAUSE DIAGNOSTIC  (AI-powered via OpenRouter / DeepSeek)
# ====================================================================

# In-session cache: stores {order_id: memo_text} so same order doesn't
# trigger a second API call when the user switches tabs and comes back.
if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = {}

with tab_copilot:
    st.markdown("### Root Cause Diagnostic")
    st.caption(
        "Generates a structured operational memo for any historically delayed shipment. "
        + ("Analysis powered by **DeepSeek** via OpenRouter." if _ai_available else
           "Running on deterministic expert heuristics — add `OPENROUTER_API_KEY` to Streamlit secrets to enable AI.")
    )

    st.markdown("---")

    # ---- Shipment Selector ----
    high_risk_orders = df_orders[df_orders["is_delayed"] == 1].head(20)
    
    # Aggregate some context columns for display
    col_sel, col_meta = st.columns([2, 1])
    with col_sel:
        selected_idx = st.selectbox(
            "Select a delayed shipment:",
            range(len(high_risk_orders)),
            format_func=lambda i: (
                f"Order ...{high_risk_orders.iloc[i]['order_id'][-8:]}  |  "
                f"{high_risk_orders.iloc[i]['seller_state']} → "
                f"{high_risk_orders.iloc[i]['customer_state']}  |  "
                f"{high_risk_orders.iloc[i]['delay_days']:.0f}d late  |  "
                f"${high_risk_orders.iloc[i]['price']:.0f} order"
            ),
        )

    target_row = high_risk_orders.iloc[selected_idx].to_dict()
    cache_key  = str(target_row.get("order_id", selected_idx))

    with col_meta:
        target_pred = predictor.predict(target_row)
        risk_style  = f"status-{target_pred['risk_level'].lower()}"
        st.markdown(
            f"""
            <div class="metric-card" style="margin-top: 4px;">
                <div style="font-size: 0.78rem; color: #9CA3AF; text-transform: uppercase; font-weight: 600;">Model Breach Probability</div>
                <div style="font-size: 2rem; font-weight: 700; color: {target_pred['risk_color']};">{target_pred['delay_probability_pct']}%</div>
                <span class="status-badge {risk_style}">{target_pred['risk_level']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Generate / cache ----
    generate_btn = st.button("Generate Operational Brief", type="primary")

    if generate_btn:
        if cache_key in st.session_state.analysis_cache:
            # Serve from in-session cache — zero API cost
            report_md = st.session_state.analysis_cache[cache_key]
            st.caption("Served from session cache.")
        else:
            with st.spinner("Analysing shipment telemetry..."):
                report_md = generate_llm_analysis(target_row, target_pred, api_key=API_KEY)
                st.session_state.analysis_cache[cache_key] = report_md
        st.markdown(report_md)

    elif cache_key in st.session_state.analysis_cache:
        st.caption("Previously generated for this order (session cache):")
        st.markdown(st.session_state.analysis_cache[cache_key])

    else:
        st.info("Select a shipment above and click **Generate Operational Brief**.")

    st.markdown("---")

    # ---- Analyst Q&A: ask a follow-up question about ANY delayed order ----
    st.markdown("#### Ask the Analyst")
    st.caption(
        "Ask any follow-up question about the selected shipment or the broader dataset. "
        "One focused question per request keeps API usage minimal."
    )

    user_question = st.text_input(
        "Question:",
        placeholder="e.g. What would reduce the delay rate in SP to BA corridor?",
        max_chars=200,
    )
    ask_btn = st.button("Ask", key="ask_analyst_btn")

    if ask_btn and user_question.strip():
        qa_cache_key = f"qa_{cache_key}_{hash(user_question.strip())}"
        if qa_cache_key in st.session_state.analysis_cache:
            st.markdown(st.session_state.analysis_cache[qa_cache_key])
            st.caption("Served from session cache.")
        elif not _ai_available:
            st.warning(
                "AI analysis is offline. Add your OpenRouter key to `.streamlit/secrets.toml` as "
                "`OPENROUTER_API_KEY` to enable this feature."
            )
        else:
            import re as _re
            safe_q = _re.sub(r"[^a-zA-Z0-9 .,?'\-:/]", "", user_question.strip())[:200]
            qa_prompt = (
                f"You are a supply-chain analyst.\n"
                f"Context: Delayed shipment on the {target_row.get('seller_state','SP')} → "
                f"{target_row.get('customer_state','RJ')} corridor. "
                f"Actual delay: {target_row.get('delay_days', 0):.0f} days. "
                f"Model breach probability: {target_pred['delay_probability_pct']}%.\n\n"
                f"Question: {safe_q}\n\n"
                f"Answer in 120 words or fewer. No emojis. Plain markdown only."
            )
            try:
                import json as _json, urllib.request as _ur
                payload = _json.dumps({
                    "model": "deepseek/deepseek-chat",
                    "messages": [{"role": "user", "content": qa_prompt}],
                    "max_tokens": 160,
                    "temperature": 0.2,
                }).encode()
                req = _ur.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    data=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {API_KEY}",
                        "HTTP-Referer": "https://github.com/oddproblem/ecom-analytics",
                        "X-Title": "SLA Intelligence Console",
                    },
                    method="POST",
                )
                with _ur.urlopen(req, timeout=10) as resp:
                    body = _json.loads(resp.read().decode())
                    answer = body["choices"][0]["message"]["content"].strip()
                    st.session_state.analysis_cache[qa_cache_key] = answer
                    st.markdown(answer)
            except Exception as err:
                st.error(f"API request failed: {err}")
    elif ask_btn and not user_question.strip():
        st.warning("Please type a question before clicking Ask.")


# Footer
st.markdown(
    """
    <div class="footer-container">
        <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
            <div>
                &copy; 2026 <b>oddproblem</b>. All rights reserved. |
                <a href="https://github.com/oddproblem/ecom-analytics" target="_blank">GitHub Repository</a> |
                <a href="mailto:argha.saha18@gmail.com">Contact: argha.saha18@gmail.com</a>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
