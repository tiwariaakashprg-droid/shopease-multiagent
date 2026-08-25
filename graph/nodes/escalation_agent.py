import re
import json
import os
from langchain_ollama import OllamaLLM
from graph.state import AgentState
from utils.timing import track_latency

llm = OllamaLLM(model="llama3.2", temperature=0.3)

LEGAL_KEYWORDS  = ["legal action", "consumer court", "lawyer", "police", "chargeback", "fraud", "sue", "court"]
REFUND_KEYWORDS = ["refund", "money back", "return", "damaged", "broken"]
AMOUNT_THRESHOLD = 5000

CALIBRATION_PATH = "config/calibration.json"
_DEFAULT_LOW_CONFIDENCE = 0.60


def _load_calibrated_threshold() -> float:
    """Loads a data-driven confidence threshold produced by
    calibrate_confidence.py (isotonic regression over evaluation_results.csv).
    Falls back to a sane default if calibration hasn't been run yet."""
    try:
        with open(CALIBRATION_PATH) as f:
            cfg = json.load(f)
            return float(cfg.get("low_confidence_threshold", _DEFAULT_LOW_CONFIDENCE))
    except Exception:
        return _DEFAULT_LOW_CONFIDENCE


@track_latency("escalation_agent")
def escalation_agent(state: AgentState) -> AgentState:
    """
    Escalation Agent - HITL trigger karta hai aur AI draft banata hai.
    Input  : state["user_message"], state["customer_data"], state["sentiment"]
    Output : state["should_escalate"], state["escalation_reason"], state["draft_reply"]
    """
    message   = (state.get("sanitized_message") or state.get("user_message", "")).lower()
    customer  = state.get("customer_data", {}) or {}
    sentiment = state.get("sentiment", "neutral")

    should_escalate = False
    reason = ""

    # Check 0 - Guardrail flagged this message (prompt injection / jailbreak attempt)
    if state.get("blocked"):
        should_escalate = True
        reason = state.get("guardrail_reason", "Guardrail agent flagged this request")

    # Check 1 - Legal keywords
    if not should_escalate:
        for kw in LEGAL_KEYWORDS:
            if kw in message:
                should_escalate = True
                reason = f"Legal threat detected: '{kw}'"
                break

    # Check 2 - High refund amount
    if not should_escalate:
        amounts = re.findall(r'rs\.?\s*(\d+)', message)
        for amt in amounts:
            if int(amt) > AMOUNT_THRESHOLD:
                should_escalate = True
                reason = f"High refund amount: Rs {amt} (limit Rs {AMOUNT_THRESHOLD})"
                break

    # Check 3 - Repeat complaints + refund intent
    if not should_escalate:
        complaints = int(customer.get("complaints", 0))
        has_refund = any(kw in message for kw in REFUND_KEYWORDS)
        if complaints >= 3 and has_refund:
            should_escalate = True
            reason = f"Customer has {complaints} unresolved complaints with refund intent"

    # Check 4 - Angry VIP
    if not should_escalate:
        if sentiment == "angry" and customer.get("tier", "") == "VIP":
            should_escalate = True
            reason = "Angry VIP customer - priority human attention needed"

    # Check 5 - Low (calibrated) retrieval confidence on a non-trivial intent.
    # Threshold is data-driven (see calibrate_confidence.py) instead of a
    # hardcoded guess, and is re-enabled here (it was commented out in the
    # original implementation).
    if not should_escalate:
        confidence = state.get("retrieval_confidence", 1.0)
        intent = state.get("intent", "general")
        low_conf_threshold = _load_calibrated_threshold()

        if intent != "general" and confidence < low_conf_threshold:
            should_escalate = True
            reason = f"Low retrieval confidence ({confidence:.2f} < {low_conf_threshold:.2f})"

    # Generate AI draft if escalating
    draft = ""
    if should_escalate:
        name = customer.get("name", "Customer")
        tier = customer.get("tier", "regular")
        draft = llm.invoke(f"""
You are a professional customer support agent.

Write a short reply under 60 words.

Customer Name: {name}
Customer Tier: {tier}
Customer Message: {state.get('user_message')}

Rules:
- Be empathetic.
- Apologize politely if needed.
- Ask for order ID or additional information if required.
- Never mention:
  * retrieval confidence
  * confidence score
  * internal errors
  * policy classification
  * AI system
  * prompt injection or guardrails

Reply:
""")

    print(f"[Escalation Agent] escalate={should_escalate} | reason={reason}")

    return {
        **state,
        "should_escalate":   should_escalate,
        "escalation_reason": reason,
        "draft_reply":       draft,
    }
