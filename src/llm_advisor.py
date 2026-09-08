"""
Operational Root-Cause Analysis and Mitigation Advisory
Generates executive-level root cause diagnosis, carrier negotiation strategies,
and customer communication memos.
Includes input sanitization, token capping, and prompt injection defenses.

Author: oddproblem (https://github.com/oddproblem/ecom-analytics)
"""

import os
import re
import json
import urllib.request
import urllib.error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sla_advisor")


def sanitize_input_string(value: str, max_len: int = 20) -> str:
    """Sanitizes categorical input values against prompt injection and excessive length."""
    if not isinstance(value, str):
        return "N/A"
    # Allow alphanumeric and underscore only, truncate to max_len
    clean = re.sub(r"[^a-zA-Z0-9_-]", "", value).strip()
    return clean[:max_len]


def generate_llm_analysis(order_features: dict, prediction_result: dict, api_key: str = None) -> str:
    """
    Generates an operational root-cause analysis using an external LLM API if key is provided,
    otherwise executes built-in deterministic expert heuristics.
    """
    # Sanitize and validate inputs
    seller_st = sanitize_input_string(order_features.get("seller_state", "SP"), 4)
    cust_st = sanitize_input_string(order_features.get("customer_state", "RJ"), 4)
    seller_id = sanitize_input_string(order_features.get("seller_id", "MERCHANT_UNKNOWN"), 16)
    
    try:
        distance_km = float(order_features.get("distance_km", 400.0))
        freight_val = float(order_features.get("freight_value", 18.0))
        price_val = float(order_features.get("price", 85.0))
        carrier_handoff = float(order_features.get("carrier_handoff_lag_days", 2.0))
        seller_late_rate = float(order_features.get("seller_historical_delay_rate", 0.05)) * 100.0
        sla_window = float(order_features.get("estimated_window_days", 15.0))
        delay_prob = float(prediction_result.get("delay_probability_pct", 50.0))
        risk_level = sanitize_input_string(prediction_result.get("risk_level", "MODERATE"), 12)
    except (ValueError, TypeError):
        distance_km, freight_val, price_val = 400.0, 18.0, 85.0
        carrier_handoff, seller_late_rate, sla_window = 2.0, 5.0, 15.0
        delay_prob, risk_level = 50.0, "MODERATE"

    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")

    # If OpenAI key provided and format valid
    if key and isinstance(key, str) and (key.startswith("sk-") or "proj" in key):
        try:
            logger.info("Dispatching diagnostic request to OpenAI completion endpoint.")
            prompt = f"""
You are an Amazon Principal Operations Research Scientist and Fulfillment Specialist.
Analyze the following e-commerce shipment with high SLA delivery delay risk.
Constraint: Respond in concise, formal business prose. Do NOT use emojis, bullet checkmarks, or informal language.

Shipment Parameters:
- Route: {seller_st} to {cust_st} ({distance_km:.0f} km)
- Freight Value: ${freight_val:.2f} | Order Price: ${price_val:.2f}
- Carrier Handoff Lag: {carrier_handoff:.1f} days
- Seller Historical Delay Rate: {seller_late_rate:.1f}%
- Promised SLA Window: {sla_window:.0f} days
- Model Delay Probability: {delay_prob:.1f}% (Risk Level: {risk_level})

Provide a structured 3-part operational report:
1. Root Cause Bottleneck: Detailed logistics explanation for the elevated risk.
2. Fulfillment Mitigation: Specific operational actions to prevent or mitigate SLA failure.
3. Customer Impact Management: Proactive measures to protect customer satisfaction before negative rating.
"""
            req_data = json.dumps({
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a senior supply chain operations analyst. Output plain text with standard markdown headings. No emojis."
                    },
                    {"role": "user", "content": prompt.strip()}
                ],
                "temperature": 0.2,
                "max_tokens": 400
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=req_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key.strip()}"
                },
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=8) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                return res_body["choices"][0]["message"]["content"]
        except Exception as err:
            logger.warning(f"External API invocation failed, falling back to expert heuristics: {err}")

    # Deterministic Expert Heuristic Synthesis (No emojis, strictly professional)
    report = f"""### Operational Root Cause Analysis & Mitigation Brief

**1. Logistics Bottleneck Diagnosis**
- **Transit Corridor Friction:** Shipment traverses the {seller_st} to {cust_st} corridor ({distance_km:.0f} km). Cross-state long-haul routing introduces sorting hub transfers that add 48 to 72 hours of variance to transit lead time.
- **Dispatch Handoff Latency:** Seller dispatch lead time is currently {carrier_handoff:.1f} days (internal benchmark is under 1.5 days). First-mile latency represents the largest controllable driver of downstream delivery failure.
- **Merchant Performance Baseline:** Seller `{seller_id[:8]}` exhibits a historical delay rate of {seller_late_rate:.1f}%, indicating systemic warehouse packaging or carrier pickup scheduling delays.

**2. Tactical Fulfillment Interventions**
- **Dynamic Line-Haul Escalation:** Route shipment through regional air hub rather than standard surface consolidation center to recover approximately 24 hours of transit time.
- **Carrier Compliance Audit:** Issue automated performance advisory to merchant regarding first-mile dispatch SLA compliance.
- **Concession Risk Avoidance:** Intercepting this shipment preserves an estimated **$12.50** in direct customer appeasement costs and mitigates a 3.4x repeat purchase churn risk.

**3. Proactive Customer Trust Recovery**
- **Preemptive Status Notification:** Deliver automated notification: *"Your delivery is currently in transit through our regional fulfillment network. Updated tracking status is available."*
- **Appeasement Credit Protocol:** Automatically credit $5.00 promotional balance if tracking does not register line-haul arrival by Day {int(sla_window) - 1} to prevent negative customer defect rating.
"""
    return report
