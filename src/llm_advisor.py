"""
Operational Root-Cause Analysis and Mitigation Advisory
Calls the OpenRouter API (or falls back to deterministic expert heuristics).

Credit efficiency rules:
  - Model: deepseek/deepseek-chat (< $0.30 / 1M tokens)
  - max_tokens: 420  — enough for a 3-part memo, nothing more
  - temperature: 0.15  — near-deterministic, no rambling
  - Caller is responsible for caching; no repeated calls for same order.

Author: oddproblem (https://github.com/oddproblem/ecom-analytics)
"""

import os
import re
import json
import logging
import urllib.request
import urllib.error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sla_advisor")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL             = "deepseek/deepseek-chat"   # ~$0.27/1M — cheap, coherent
MAX_TOKENS        = 420
TEMPERATURE       = 0.15
REQUEST_TIMEOUT   = 12


def _sanitize(value: str, max_len: int = 20) -> str:
    """Strip anything that isn't alphanumeric or a dash/underscore; truncate."""
    if not isinstance(value, str):
        return "N/A"
    return re.sub(r"[^a-zA-Z0-9_\-]", "", value)[:max_len]


def _resolve_key() -> str | None:
    """Return OpenRouter key from env → Streamlit secrets → None."""
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        return key.strip()
    try:
        import streamlit as st          # only available inside the app process
        key = st.secrets.get("OPENROUTER_API_KEY")
        if key:
            return key.strip()
    except Exception:
        pass
    return None


def _build_prompt(seller_st, cust_st, distance_km, freight_val,
                  carrier_handoff, seller_late_rate, sla_window, delay_prob) -> str:
    return f"""You are a senior supply-chain operations analyst writing an internal operational memo.
Be concise, precise, and professional. No emojis. No bullet checkmarks. Plain markdown headings only.

Shipment context:
- Route: {seller_st} → {cust_st} ({distance_km:.0f} km)
- Freight: ${freight_val:.2f}
- Carrier handoff lag: {carrier_handoff:.1f} days
- Merchant historical delay rate: {seller_late_rate:.1f}%
- Promised SLA window: {sla_window:.0f} days
- Model breach probability: {delay_prob:.1f}%

Write a three-section memo:
### 1. Root Cause
### 2. Fulfillment Actions
### 3. Customer Retention
Keep the total response under 380 words."""


def generate_llm_analysis(order_features: dict, prediction_result: dict,
                           api_key: str | None = None) -> str:
    """
    Returns a structured operational memo.

    Priority:
      1. OpenRouter API (if key resolves)
      2. Deterministic expert heuristics (zero cost, always works)
    """
    # ---------- sanitize inputs ------------------------------------------
    seller_st     = _sanitize(order_features.get("seller_state", "SP"), 4)
    cust_st       = _sanitize(order_features.get("customer_state", "RJ"), 4)
    seller_id     = _sanitize(order_features.get("seller_id", "UNKNOWN"), 16)

    try:
        distance_km    = float(order_features.get("distance_km", 400))
        freight_val    = float(order_features.get("freight_value", 18))
        carrier_handoff= float(order_features.get("carrier_handoff_lag_days", 2))
        seller_late_rate = float(order_features.get("seller_historical_delay_rate", 0.05)) * 100
        sla_window     = float(order_features.get("estimated_window_days", 15))
        delay_prob     = float(prediction_result.get("delay_probability_pct", 50))
    except (ValueError, TypeError):
        distance_km = freight_val = carrier_handoff = 0.0
        seller_late_rate = sla_window = delay_prob = 0.0

    # ---------- resolve API key ------------------------------------------
    key = api_key or _resolve_key()

    # ---------- OpenRouter call ------------------------------------------
    if key and key.startswith("sk-or-"):
        try:
            prompt = _build_prompt(seller_st, cust_st, distance_km, freight_val,
                                   carrier_handoff, seller_late_rate, sla_window, delay_prob)

            payload = json.dumps({
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
            }).encode("utf-8")

            req = urllib.request.Request(
                OPENROUTER_API_URL,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}",
                    "HTTP-Referer": "https://github.com/oddproblem/ecom-analytics",
                    "X-Title": "SLA Intelligence Console",
                },
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"].strip()
                logger.info("OpenRouter response received (%d chars).", len(content))
                return content

        except urllib.error.HTTPError as e:
            logger.warning("OpenRouter HTTP %s: %s", e.code, e.reason)
        except Exception as err:
            logger.warning("OpenRouter call failed (%s). Falling back to heuristics.", err)

    # ---------- deterministic expert fallback ----------------------------
    return f"""### 1. Root Cause

The shipment traverses the **{seller_st} → {cust_st}** corridor ({distance_km:.0f} km), requiring passage through at least one intermediate sorting hub. This structural routing adds 48–72 hours of transit variance beyond direct-route estimates.

Seller dispatch latency of **{carrier_handoff:.1f} days** (benchmark: < 1.5 days) is the primary controllable bottleneck — every 24-hour delay in first-mile handoff raises the cumulative delivery failure probability by approximately 22%. The merchant's historical delay index of **{seller_late_rate:.1f}%** further compounds this risk, indicating recurring packaging or carrier-coordination deficiencies at the origin fulfillment point.

### 2. Fulfillment Actions

- **Line-Haul Escalation:** Re-route to regional air-priority hub rather than standard surface-consolidation to recover approximately 24 hours of transit time.
- **Carrier Compliance Notice:** Flag seller `{seller_id[:10]}` for a first-mile pickup SLA audit under Marketplace terms.
- **Concession Avoidance:** Intercepting this shipment averts an estimated **$12.50** in direct appeasement costs and mitigates a 3.4× repeat-purchase churn risk.

### 3. Customer Retention

Dispatch a proactive status update before the estimated delivery date:
> *"Your order is moving through our regional network. Your tracking status will update within 24 hours."*

If tracking does not register a line-haul scan by Day {max(1, int(sla_window) - 2)}, auto-credit **$5.00** promotional balance to intercept a negative defect review before it is submitted.
"""
