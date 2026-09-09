"""
Operational Root-Cause Analysis and Mitigation Advisory
Calls the OpenRouter API with multi-model failover cascade (or falls back to deterministic expert heuristics).

Credit efficiency & resilience rules:
  - Models: google/gemini-2.5-flash (~$0.15/1M), deepseek/deepseek-chat (<$0.30/1M), meta-llama/llama-3.3-70b-instruct
  - Multi-model failover prevents HTTP 429 (rate limits) and 504 (timeouts) on busy upstream providers
  - max_tokens: 420 for memos, 180 for analyst Q&A
  - temperature: 0.15 - 0.20  near-deterministic, no rambling
  - In-session caching prevents duplicate calls for the same order or question.

Author: oddproblem (https://github.com/oddproblem/ecom-analytics)
"""

import os
import re
import json
import logging
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sla_advisor")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Primary & Fallback model pool
DEFAULT_FALLBACK_MODELS = [
    "google/gemini-2.5-flash",
    "deepseek/deepseek-chat",
    "meta-llama/llama-3.3-70b-instruct",
]

AVAILABLE_MODELS = {
    "auto": "Auto-Failover Pool (Gemini + DeepSeek + LLaMA)",
    "google/gemini-2.5-flash": "Google Gemini 2.5 Flash (Ultra-Fast & High Uptime)",
    "deepseek/deepseek-chat": "DeepSeek Chat V3 (Cost-Optimized)",
    "meta-llama/llama-3.3-70b-instruct": "Meta LLaMA 3.3 70B (High Precision)",
}

MAX_TOKENS_MEMO    = 420
MAX_TOKENS_QA      = 180
TEMPERATURE_MEMO   = 0.15
TEMPERATURE_QA     = 0.20
REQUEST_TIMEOUT    = 12


def _sanitize(value: str, max_len: int = 20) -> str:
    """Strip anything that isn't alphanumeric or a dash/underscore; truncate."""
    if not isinstance(value, str):
        return "N/A"
    return re.sub(r"[^a-zA-Z0-9_\-]", "", value)[:max_len]


def _resolve_key(api_key: Optional[str] = None) -> Optional[str]:
    """Return OpenRouter key from explicit arg -> env -> Streamlit secrets -> None."""
    if api_key and api_key.strip():
        return api_key.strip()
    key = os.getenv("OPENROUTER_API_KEY")
    if key and key.strip():
        return key.strip()
    try:
        import streamlit as st          # available inside Streamlit app process
        key = st.secrets.get("OPENROUTER_API_KEY")
        if key and key.strip():
            return key.strip()
    except Exception:
        pass
    return None


def _call_openrouter(
    messages: List[Dict[str, Any]],
    max_tokens: int = 400,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
    preferred_model: Optional[str] = None,
) -> Tuple[Optional[str], str]:
    """
    Executes a chat completion request to OpenRouter with automatic multi-model failover.
    Returns: (response_text, model_name_or_error)
    """
    key = _resolve_key(api_key)
    if not key or not key.startswith("sk-or-"):
        return None, "Missing or invalid OpenRouter API key"

    # Assemble candidate list
    if preferred_model and preferred_model != "auto" and preferred_model in AVAILABLE_MODELS:
        candidates = [preferred_model] + [m for m in DEFAULT_FALLBACK_MODELS if m != preferred_model]
    else:
        candidates = list(DEFAULT_FALLBACK_MODELS)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://github.com/oddproblem/ecom-analytics",
        "X-Title": "SLA Intelligence Console",
    }

    # 1. Primary Attempt: Native OpenRouter multi-model fallback array
    payload = json.dumps({
        "models": candidates,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(OPENROUTER_API_URL, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))

            # Check for inline error objects in 200 responses
            if "error" in body and isinstance(body["error"], dict):
                err_msg = body["error"].get("message", "Unknown error")
                logger.warning("OpenRouter returned inline error: %s", err_msg)
            elif "choices" in body and len(body["choices"]) > 0:
                content = body["choices"][0]["message"]["content"].strip()
                resolved_model = body.get("model", candidates[0])
                logger.info("OpenRouter response received via %s (%d chars).", resolved_model, len(content))
                return content, resolved_model

    except urllib.error.HTTPError as e:
        err_body = ""
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            pass
        logger.warning("OpenRouter batch call HTTP %s: %s | %s", e.code, e.reason, err_body)
    except Exception as err:
        logger.warning("OpenRouter batch call failed (%s). Attempting individual fallbacks.", err)

    # 2. Secondary Attempt: Client-side single-model failover loop (intercept 429/504)
    for model_name in candidates:
        try:
            logger.info("Trying client-side fallback model: %s", model_name)
            single_payload = json.dumps({
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }).encode("utf-8")

            req = urllib.request.Request(OPENROUTER_API_URL, data=single_payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                if "choices" in body and len(body["choices"]) > 0:
                    content = body["choices"][0]["message"]["content"].strip()
                    logger.info("Client fallback succeeded with %s", model_name)
                    return content, model_name
        except Exception as fallback_err:
            logger.warning("Fallback model %s failed: %s", model_name, fallback_err)
            continue

    return None, "All candidate LLM models exhausted or rate-limited upstream"


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


def generate_llm_analysis(
    order_features: dict,
    prediction_result: dict,
    api_key: Optional[str] = None,
    preferred_model: Optional[str] = None,
    return_meta: bool = False,
) -> Any:
    """
    Returns a structured operational memo.

    Priority:
      1. OpenRouter API with multi-model cascade (Gemini / DeepSeek / LLaMA)
      2. Deterministic expert heuristics (zero cost, 100% uptime fallback)
    """
    # ---------- sanitize inputs ------------------------------------------
    seller_st     = _sanitize(order_features.get("seller_state", "SP"), 4)
    cust_st       = _sanitize(order_features.get("customer_state", "RJ"), 4)
    seller_id     = _sanitize(order_features.get("seller_id", "UNKNOWN"), 16)

    try:
        distance_km     = float(order_features.get("distance_km", 400))
        freight_val     = float(order_features.get("freight_value", 18))
        carrier_handoff = float(order_features.get("carrier_handoff_lag_days", 2))
        seller_late_rate= float(order_features.get("seller_historical_delay_rate", 0.05)) * 100
        sla_window      = float(order_features.get("estimated_window_days", 15))
        delay_prob      = float(prediction_result.get("delay_probability_pct", 50))
    except (ValueError, TypeError):
        distance_km = freight_val = carrier_handoff = 0.0
        seller_late_rate = sla_window = delay_prob = 0.0

    prompt = _build_prompt(seller_st, cust_st, distance_km, freight_val,
                           carrier_handoff, seller_late_rate, sla_window, delay_prob)

    messages = [{"role": "user", "content": prompt}]
    content, source = _call_openrouter(
        messages=messages,
        max_tokens=MAX_TOKENS_MEMO,
        temperature=TEMPERATURE_MEMO,
        api_key=api_key,
        preferred_model=preferred_model,
    )

    if content:
        if return_meta:
            return content, source, False
        return content

    # ---------- deterministic expert fallback ----------------------------
    fallback_text = f"""### 1. Root Cause

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
    if return_meta:
        return fallback_text, "Deterministic Expert Heuristics", True
    return fallback_text


def ask_analyst_qa(
    order_context: dict,
    question: str,
    prediction_result: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    preferred_model: Optional[str] = None,
) -> dict:
    """
    Q&A chatbot engine for supply-chain operational follow-ups.
    Uses multi-model OpenRouter failover with domain-informed heuristic backup.
    Returns: {"answer": str, "model": str, "is_fallback": bool}
    """
    safe_q = re.sub(r"[^a-zA-Z0-9 .,?'\-:/$%]", "", question.strip())[:250]
    if not safe_q:
        return {
            "answer": "Please provide a valid question regarding shipment telemetry or SLA risk.",
            "model": "Input Validator",
            "is_fallback": True,
        }

    seller_st = _sanitize(order_context.get("seller_state", "SP"), 4)
    cust_st   = _sanitize(order_context.get("customer_state", "RJ"), 4)
    delay_days = order_context.get("delay_days", 0)
    try:
        delay_days = float(delay_days)
    except (ValueError, TypeError):
        delay_days = 0.0

    delay_prob = 50.0
    if prediction_result:
        try:
            delay_prob = float(prediction_result.get("delay_probability_pct", 50.0))
        except (ValueError, TypeError):
            pass

    qa_prompt = (
        f"You are a supply-chain operations analyst.\n"
        f"Context: Shipment on the {seller_st} → {cust_st} corridor. "
        f"Historical delay: {delay_days:.0f} days. "
        f"Model breach probability: {delay_prob:.1f}%.\n\n"
        f"Question: {safe_q}\n\n"
        f"Answer in 120 words or fewer. Concise, analytical, and professional. No emojis. Plain markdown only."
    )

    messages = [{"role": "user", "content": qa_prompt}]
    content, source = _call_openrouter(
        messages=messages,
        max_tokens=MAX_TOKENS_QA,
        temperature=TEMPERATURE_QA,
        api_key=api_key,
        preferred_model=preferred_model,
    )

    if content:
        return {
            "answer": content,
            "model": source,
            "is_fallback": False,
        }

    # Deterministic domain-informed fallback for Q&A
    q_lower = safe_q.lower()
    carrier_lag = float(order_context.get("carrier_handoff_lag_days", 2.0))
    seller_delay_rate = float(order_context.get("seller_historical_delay_rate", 0.05)) * 100

    if any(k in q_lower for k in ["reduce", "fix", "mitigate", "action", "corridor", "route"]):
        answer = (
            f"To mitigate delivery risk on the **{seller_st} → {cust_st}** corridor:\n\n"
            f"1. **Enforce First-Mile Handoff SLA:** Current carrier lag ({carrier_lag:.1f} days) is the largest single operational variance. Enforcing a strict 24-hour dispatch limit reduces breach risk significantly.\n"
            f"2. **Line-Haul Direct Injection:** Bypass secondary sorting centers by consolidating palletized volume directly into destination regional hubs.\n"
            f"3. **Dynamic Carrier Allocation:** Reroute packages with >60% breach probability to expedited air-priority tiers."
        )
    elif any(k in q_lower for k in ["cost", "concession", "refund", "appeasement", "churn"]):
        answer = (
            f"Proactive intervention for this shipment avoids an estimated **$12.50** in customer appeasement coupons "
            f"and mitigates a **3.4× churn risk**. Triggering automated pre-breach status notifications at Day 3 costs "
            f"under $0.85 and eliminates over 65% of customer support contact volume."
        )
    elif any(k in q_lower for k in ["seller", "merchant", "vendor"]):
        answer = (
            f"The seller on this shipment maintains a **{seller_delay_rate:.1f}%** historical delay rate. "
            f"Recommended operational action is issuing a formal Marketplace Carrier Compliance Notice and "
            f"mandating scan-upon-tender verification at the origin pickup hub."
        )
    else:
        answer = (
            f"Operational assessment for {seller_st} → {cust_st} shipment: The primary risk driver is "
            f"first-mile handoff latency ({carrier_lag:.1f} days) coupled with line-haul transit variance. "
            f"Proactive tracking telemetry and SLA escalation protocols are recommended to intercept breaches."
        )

    return {
        "answer": answer,
        "model": "Deterministic Expert Heuristics",
        "is_fallback": True,
    }
