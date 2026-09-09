"""
Unit and Integration Tests for LLM Operational Advisor & Q&A Chatbot
Verifies prompt generation, model failover handling, and heuristic fallbacks.

Author: oddproblem (https://github.com/oddproblem/ecom-analytics)
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm_advisor import (
    generate_llm_analysis,
    ask_analyst_qa,
    _sanitize,
    _resolve_key,
    AVAILABLE_MODELS,
    DEFAULT_FALLBACK_MODELS,
)


def test_sanitize_inputs():
    """Verify input sanitization strips unsafe characters and truncates."""
    assert _sanitize("SP!@#$%^&*()", 4) == "SP"
    assert _sanitize(12345) == "N/A"
    assert len(_sanitize("very_long_seller_identifier_1234567890", 10)) == 10


def test_generate_llm_analysis_fallback_when_no_key():
    """Verify deterministic fallback memo generation when no API key is provided."""
    sample_order = {
        "seller_state": "SP",
        "customer_state": "RJ",
        "seller_id": "SELLER_999",
        "distance_km": 420.0,
        "freight_value": 19.50,
        "carrier_handoff_lag_days": 2.5,
        "seller_historical_delay_rate": 0.08,
        "estimated_window_days": 14.0,
    }
    sample_pred = {"delay_probability_pct": 65.0, "risk_level": "ELEVATED"}

    # Force invalid key to test 100% reliable heuristic fallback
    memo = generate_llm_analysis(sample_order, sample_pred, api_key="invalid-key")
    assert "### 1. Root Cause" in memo
    assert "### 2. Fulfillment Actions" in memo
    assert "### 3. Customer Retention" in memo
    assert "SP → RJ" in memo

    # Test return_meta flag
    text, model, is_fallback = generate_llm_analysis(
        sample_order, sample_pred, api_key="invalid-key", return_meta=True
    )
    assert is_fallback is True
    assert model == "Deterministic Expert Heuristics"


def test_ask_analyst_qa_heuristics():
    """Verify Q&A chatbot provides intelligent heuristic answers when offline or invalid key."""
    sample_order = {
        "seller_state": "SP",
        "customer_state": "BA",
        "carrier_handoff_lag_days": 3.2,
        "seller_historical_delay_rate": 0.12,
        "delay_days": 4.0,
    }

    # Test route mitigation query
    res = ask_analyst_qa(
        sample_order,
        "How can we reduce delay on this corridor?",
        api_key="invalid-key"
    )
    assert "answer" in res
    assert res["is_fallback"] is True
    assert "SP → BA" in res["answer"]
    assert "First-Mile Handoff SLA" in res["answer"]

    # Test cost query
    res_cost = ask_analyst_qa(
        sample_order,
        "What is the concession cost impact?",
        api_key="invalid-key"
    )
    assert "$12.50" in res_cost["answer"]

    # Test empty query validation
    res_empty = ask_analyst_qa(sample_order, "   ", api_key="invalid-key")
    assert "Please provide a valid question" in res_empty["answer"]


def test_live_openrouter_if_key_present():
    """If a valid key exists in environment/secrets, verify live API call succeeds with failover."""
    key = _resolve_key()
    if not key or not key.startswith("sk-or-"):
        print("[SKIP] No live OpenRouter key available for live integration test.")
        return

    sample_order = {
        "seller_state": "SP",
        "customer_state": "RJ",
        "distance_km": 350.0,
        "freight_value": 15.0,
        "carrier_handoff_lag_days": 1.5,
        "seller_historical_delay_rate": 0.04,
        "estimated_window_days": 12.0,
        "delay_days": 2.0,
    }
    sample_pred = {"delay_probability_pct": 52.0}

    # Test live Q&A call
    res = ask_analyst_qa(
        sample_order,
        "What is the single highest leverage operational intervention?",
        prediction_result=sample_pred,
        api_key=key,
        preferred_model="auto",
    )
    assert "answer" in res
    assert len(res["answer"]) > 10
    print(f"[LIVE TEST] Model used: {res['model']}, Fallback: {res['is_fallback']}")


if __name__ == "__main__":
    test_sanitize_inputs()
    test_generate_llm_analysis_fallback_when_no_key()
    test_ask_analyst_qa_heuristics()
    test_live_openrouter_if_key_present()
    print("[PASS] All LLM advisor tests passed successfully!")
