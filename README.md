# Supply Chain SLA Intelligence Console

A machine-learning system for real-time SLA breach prediction in e-commerce fulfillment networks. Built on 96,470 verified orders from a Brazilian marketplace, the system predicts delivery failures before they occur, quantifies customer satisfaction risk, and generates AI-assisted operational diagnostics.

**Live deployment:** https://ecom-analytics-argha.streamlit.app

---

## Overview

Late deliveries are not a random event. They are the predictable output of specific, measurable conditions: aggressive SLA commitments made without logistics capacity to back them up, sellers with poor dispatch track records, and routes that require multi-hop hub transfers. This system makes those conditions visible before a shipment breaches its promise date.

The pipeline ingests raw order data, engineers operational features, trains a gradient boosting classifier, and exposes predictions through a five-panel analytics console. A secondary AI layer — powered by DeepSeek via OpenRouter — converts model outputs into structured operational memos and answers analyst questions in natural language.

---

## Screenshots

### Executive Overview

![Executive Overview — KPI cards and fulfillment trends](C:/Users/argha/.gemini/antigravity-ide/brain/c7ddced7-1cc5-40ee-8a97-839585174320/tab_1_executive_overview_top_1788902837028.png)

![Executive Overview — Monthly volume and on-time trend charts](C:/Users/argha/.gemini/antigravity-ide/brain/c7ddced7-1cc5-40ee-8a97-839585174320/tab_1_executive_overview_scrolled_1788902845397.png)

### SLA Risk Simulation

![SLA Risk Simulation — Interactive parameter controls and breach probability output](C:/Users/argha/.gemini/antigravity-ide/brain/c7ddced7-1cc5-40ee-8a97-839585174320/tab_2_sla_risk_simulation_1788902875921.png)

![SLA Risk Simulation — Risk flags and operational mitigation panel](C:/Users/argha/.gemini/antigravity-ide/brain/c7ddced7-1cc5-40ee-8a97-839585174320/tab_2_sla_risk_simulation_scrolled_1788902886827.png)

### Geographic Corridors

![Geographic Corridors — Delay rate by state and freight cost scatter](C:/Users/argha/.gemini/antigravity-ide/brain/c7ddced7-1cc5-40ee-8a97-839585174320/tab_3_geographic_corridors_1788902920968.png)

### Model Evaluation and Explainability

![Model Evaluation — Feature importance and confusion matrix](C:/Users/argha/.gemini/antigravity-ide/brain/c7ddced7-1cc5-40ee-8a97-839585174320/tab_4_model_evaluation_xai_1788902976708.png)

![Model Evaluation — Threshold tuning and cost-utility analysis](C:/Users/argha/.gemini/antigravity-ide/brain/c7ddced7-1cc5-40ee-8a97-839585174320/tab_4_model_evaluation_xai_scrolled_1788902990868.png)

### Root Cause Diagnostic

![Root Cause Diagnostic — Shipment selector and AI brief generator](C:/Users/argha/.gemini/antigravity-ide/brain/c7ddced7-1cc5-40ee-8a97-839585174320/tab_5_root_cause_diagnostic_1788903049040.png)

![Root Cause Diagnostic — Generated DeepSeek operational memo](C:/Users/argha/.gemini/antigravity-ide/brain/c7ddced7-1cc5-40ee-8a97-839585174320/tab_5_operational_brief_generated_1788903093206.png)

---

## Architecture

```
Raw CSV Data (Olist Dataset — 96,470 orders)
        |
        v
+---------------------------+
|   src/pipeline.py         |   ETL: joins orders, items, customers,
|   Feature Store Builder   |   sellers, reviews, geolocation.
+---------------------------+   Outputs three Parquet files to data/processed/
        |
        |-- orders_sla_features.parquet      (per-order feature vectors)
        |-- executive_kpis.parquet           (aggregate KPIs)
        |-- state_logistics.parquet          (corridor-level summaries)
        |
        v
+---------------------------+
|   src/model.py            |   Trains HistGradientBoostingClassifier
|   Model Training          |   with class-weight balancing.
+---------------------------+   Cost-sensitive threshold tuning on PR curve.
        |
        |-- models/delivery_delay_classifier.joblib
        |-- models/model_metrics.json
        |
        v
+---------------------------+
|   src/predictor.py        |   Real-time inference wrapper.
|   Inference Engine        |   Accepts a dict of order features,
+---------------------------+   returns probability, risk tier, flags.
        |
        v
+---------------------------+
|   src/llm_advisor.py      |   Calls DeepSeek via OpenRouter API.
|   AI Operational Advisor  |   Generates 3-section operational memos.
+---------------------------+   Falls back to deterministic heuristics
        |                       if no API key is present.
        |
        v
+---------------------------+
|   app.py                  |   Streamlit application controller.
|   Analytics Console       |   Reads cached Parquet stores and the
+---------------------------+   joblib model. Renders 5 analysis panels.
```

**Data flow summary:**

1. `src/pipeline.py` reads the raw Olist CSVs from `data/raw/`, joins them, engineers 17 predictive features per order, and writes three Parquet feature stores to `data/processed/`.
2. `src/model.py` reads the feature store, applies class-imbalanced training (`HistGradientBoostingClassifier`), tunes the decision threshold on the precision-recall curve for concession-cost minimization, and persists the model artifact and performance metrics.
3. `src/predictor.py` provides a stateless `predict(order_dict)` function used by the Streamlit app for live simulation and historical order scoring.
4. `src/llm_advisor.py` wraps the OpenRouter API with strict input sanitization, a 420-token hard cap, session-level response caching, and a full deterministic fallback so the app works without an API key.
5. `app.py` is the top-level Streamlit controller. It caches data with `@st.cache_data` and the model with `@st.cache_resource`, so the pipeline loads once per session.

---

## Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | 0.8547 |
| Precision-Recall AUC | 0.4869 |
| Decision threshold (cost-optimal) | 0.7539 |
| Precision at threshold | 44.2% |
| Recall at threshold | 49.5% |
| Projected concession savings (test set) | $6,781.25 |
| Test partition | 19,294 shipments |

The decision threshold was set to maximize expected savings — not F1 score. At the operating point of 0.7539, catching a true positive saves ~$12.50 in customer concession cost; a false positive costs a small amount of intervention overhead. The threshold is tuned to the point where the expected value of acting on a flagged order is still positive.

---

## Feature Importance

The top five predictive features by permutation importance on the held-out test partition:

| Feature | Interpretation |
|---|---|
| Promised SLA Window (days) | Sellers who commit to aggressive timelines without the carrier network to support them are the primary driver of failure |
| Calendar Month | Peak season (Nov-Jan) and off-season capacity fluctuations create systematic delay spikes |
| Seller Dispatch Lead Time | Every 24-hour delay in first-mile handoff raises cumulative failure probability by ~22% |
| Merchant Historical Delay Index | Past behavior predicts future performance; high-delay sellers rarely self-correct |
| Haversine Distance | Longer routes require multi-leg hub transfers and accumulate variance at each handoff point |

---

## Console Panels

**Executive Overview** — Five KPI cards (volume, GMV, SLA rate, delivery window, satisfaction score) plus two interactive charts: a defect rate comparison between on-time and late cohorts, and a monthly dual-axis trend showing volume growth against SLA compliance.

**SLA Risk Simulation** — Configurable simulation for any hypothetical shipment. Set origin and destination state, package weight and volume, order value, freight cost, seller dispatch lag, and the promised delivery window. The model returns a breach probability, a risk tier (LOW / MODERATE / ELEVATED / CRITICAL), and a list of specific risk flags driving the prediction.

**Geographic Corridors** — A ranked bar chart of the 10 states with the highest delay rates, and a scatter plot mapping each state's average transit time against average freight cost, colored by delay rate. Northern and Northeastern states consistently show 15-24% delay rates due to structural routing through multiple sorting hubs.

**Model Evaluation and Explainability** — Feature importance bar chart (permutation-based, test partition), a full confusion matrix at the operating threshold, and an interactive threshold slider with real-time precision/recall readout and projected concession impact.

**Root Cause Diagnostic** — Select any historically delayed order from a dropdown. The system runs it through the classifier and shows a breach probability card, then generates a three-section operational memo (Root Cause / Fulfillment Actions / Customer Retention) via DeepSeek. A follow-up Q&A input lets you ask free-form questions about the selected shipment. Both the memo and Q&A answers are session-cached to avoid duplicate API calls.

---

## Dataset

Source: [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle)

96,470 verified delivered orders from September 2016 to October 2018. Covers eight raw tables: orders, order items, products, customers, sellers, reviews, payments, and geolocation. The dataset is representative of a mid-market marketplace fulfillment network with a mix of intra-state and long-haul inter-state shipments.

Overall SLA breach rate in the dataset: **8.11%** — a class imbalance that is explicitly handled during training via `class_weight` balancing in the classifier.

---

## Project Structure

```
.
├── app.py                          # Streamlit application entry point
├── requirements.txt                # Pinned Python dependencies
├── data/
│   ├── raw/                        # Original Olist CSV files (not committed)
│   └── processed/                  # Parquet feature stores (committed)
│       ├── orders_sla_features.parquet
│       ├── executive_kpis.parquet
│       └── state_logistics.parquet
├── models/
│   ├── delivery_delay_classifier.joblib   # Trained model artifact
│   └── model_metrics.json                 # Performance metrics and threshold
├── src/
│   ├── config.py                   # Centralized path and constant definitions
│   ├── pipeline.py                 # ETL and feature engineering
│   ├── model.py                    # Training, threshold tuning, artifact export
│   ├── predictor.py                # Real-time inference wrapper
│   └── llm_advisor.py              # OpenRouter AI advisor with fallback heuristics
├── tests/
│   └── test_pipeline.py            # Pipeline integrity and model performance tests
└── .streamlit/
    ├── config.toml                 # Server and security configuration
    └── secrets.toml                # API keys (gitignored, never committed)
```

---

## Local Setup

**Prerequisites:** Python 3.10+, pip

```bash
# 1. Clone the repository
git clone https://github.com/oddproblem/ecom-analytics.git
cd ecom-analytics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the Olist dataset from Kaggle and place CSV files in:
#    data/raw/

# 4. Build the feature store
python -m src.pipeline

# 5. Train the model
python -m src.model

# 6. (Optional) Add your OpenRouter API key for AI diagnostics
#    Create .streamlit/secrets.toml and add:
#    OPENROUTER_API_KEY = "sk-or-v1-..."

# 7. Launch the console
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**Running tests:**

```bash
python -m pytest tests/ -v
```

---

## Deployment

Deployed on Streamlit Community Cloud. To deploy your own instance:

1. Fork the repository on GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect the fork.
3. Set main file path to `app.py`.
4. In app Settings, add `OPENROUTER_API_KEY` under Secrets.
5. Deploy. The app will install pinned dependencies and start within ~3 minutes.

The committed Parquet feature stores and `.joblib` model artifact mean the app starts immediately without needing to re-run the pipeline on the cloud instance.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| scikit-learn | 1.8.0 (pinned) | HistGradientBoostingClassifier, metrics |
| pandas | >=2.0 | Data processing |
| numpy | >=1.24 | Numerical operations |
| pyarrow | >=14.0 | Parquet feature store read/write |
| joblib | >=1.3 | Model serialization |
| streamlit | >=1.30 | Web application framework |
| plotly | >=5.18 | Interactive charts |

scikit-learn is pinned to an exact version because the `.joblib` model artifact encodes internal sklearn structures. A version mismatch causes a `No module named '_loss'` error at load time. If you retrain the model locally, update the pin in `requirements.txt` to match your installed version.

---

## Author

[oddproblem](https://github.com/oddproblem) — argha.saha18@gmail.com
