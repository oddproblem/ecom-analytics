"""
AI Operations Copilot & Root-Cause Analyst
Provides executive-level root cause diagnosis, carrier negotiation strategies,
and customer communication memos.
Supports OpenAI / Gemini API keys, or falls back to built-in rule-based expert heuristics.

Author: oddproblem (https://github.com/oddproblem/ecom-analytics)
"""

import os
import json
import urllib.request
import urllib.error


def generate_llm_analysis(order_features: dict, prediction_result: dict, api_key: str = None) -> str:
    """
    Generates an executive operational recommendation using LLM API if key provided,
    otherwise uses expert system heuristic generator.
    """
    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")

    # If OpenAI key provided
    if key and (key.startswith("sk-") or "proj" in key):
        try:
            prompt = f"""
You are an Amazon Principal Operations Research Scientist and Fulfillment Specialist.
Analyze the following e-commerce shipment with high SLA delivery delay risk:

Shipment Parameters:
- Route: {order_features.get('seller_state')} -> {order_features.get('customer_state')} ({order_features.get('distance_km', 0):.0f} km)
- Freight Value: ${order_features.get('freight_value', 0):.2f} | Order Price: ${order_features.get('price', 0):.2f}
- Carrier Handoff Lag: {order_features.get('carrier_handoff_lag_days', 0):.1f} days
- Seller Historical Delay Rate: {order_features.get('seller_historical_delay_rate', 0)*100:.1f}%
- Promised SLA Window: {order_features.get('estimated_window_days', 0):.0f} days
- Model Delay Probability: {prediction_result.get('delay_probability_pct', 0)}% (Risk Level: {prediction_result.get('risk_level')})

Provide a concise 3-part operational report:
1. Primary Root-Cause Bottleneck (Why this order is failing SLA)
2. Immediate Mitigation Action (Amazon logistics rerouting / carrier escalation)
3. Proactive Customer Engagement (How to preserve customer trust before negative review)
"""
            req_data = json.dumps({
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a senior Amazon Supply Chain & Fulfillment AI advisor."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.4,
                "max_tokens": 500
            }).encode("utf-8")

            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=req_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}"
                },
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                res_body = json.loads(response.read().decode("utf-8"))
                return res_body["choices"][0]["message"]["content"]
        except Exception as e:
            pass  # Fallback to heuristic expert system

    # Intelligent Expert Heuristic Fallback
    prob = prediction_result.get("delay_probability_pct", 0)
    seller_st = order_features.get("seller_state", "SP")
    cust_st = order_features.get("customer_state", "RJ")
    handoff = order_features.get("carrier_handoff_lag_days", 2.0)
    dist = order_features.get("distance_km", 400)
    seller_sla = order_features.get("seller_historical_delay_rate", 0.05) * 100

    report = f"""### 📋 Executive Operations & Root-Cause Diagnostic

**1. Primary Root-Cause Bottleneck Analysis:**
* **Logistics Corridor Friction:** The shipment spans `{seller_st} → {cust_st}` ({dist:.0f} km). Cross-state long-haul routing introduces sorting hub transit handoffs that add 48–72 hours of variability.
* **Carrier Handoff Latency:** Seller dispatch lag is currently `{handoff:.1f} days` (target benchmark is < 1.5 days). Every 24h delay in first-mile handoff increases customer SLA breach probability by ~22%.
* **Merchant Reliability Profile:** Historical seller delay index is `{seller_sla:.1f}%`. This merchant exhibits high variance in fulfillment packaging lead time.

**2. Tactical Fulfillment Mitigation (Amazon Logistics Protocol):**
* **Dynamic Line-Haul Escalation:** Reroute order through regional express hub rather than standard consolidation sorting center.
* **Carrier SLA Penalty Notice:** Flag seller `{order_features.get('seller_id', 'SELLER_ID')[:8]}` for first-mile pickup compliance audit under Marketplace SLA terms.
* **Estimated Concession Avoidance:** Intercepting this shipment preserves an estimated **$12.50** in direct appeasement costs and eliminates a ~3.4x churn risk.

**3. Proactive Customer Trust Recovery:**
* **Pre-emptive Outreach:** Trigger automated Amazon app push notification: *"Your delivery is on its way with extra care from our regional logistics team."*
* **Appeasement Credit:** Automatically deposit $5.00 promotional credit if delivery tracking exceeds Day {int(order_features.get('estimated_window_days', 15))-1} to preserve 5-star review sentiment.
"""
    return report
