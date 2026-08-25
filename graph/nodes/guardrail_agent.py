"""
Guardrail Agent (NEW) - runs BEFORE the Intent Agent.

Enterprise systems that touch real customer data need a safety layer that
runs before anything else in the pipeline:

  1. PII redaction  - the raw user_message may contain an email, phone
     number, or card number. We keep the ORIGINAL message for CRM lookups
     that legitimately need it, but store a redacted copy
     (sanitized_message) that is what actually gets logged and what gets
     passed into every downstream LLM prompt as conversational context.
  2. Prompt-injection screening - if the message looks like an attempt to
     override the agent's instructions ("ignore previous instructions",
     "reveal your system prompt", etc.), we flag it and force escalation to
     a human instead of letting any agent improvise a response.

This directly addresses the "Guardrails and safety layer" gap called out in
the paper review (PII leak risk + prompt-injection risk were both listed as
missing in an enterprise-grade deployment).
"""
from graph.state import AgentState
from utils.guardrails import redact_pii, detect_prompt_injection
from utils.timing import track_latency


@track_latency("guardrail_agent")
def guardrail_agent(state: AgentState) -> AgentState:
    message = state.get("user_message", "") or ""

    redacted, pii_found = redact_pii(message)
    is_injection, matched = detect_prompt_injection(message)

    blocked = False
    reason = ""

    if is_injection:
        blocked = True
        reason = f"Prompt-injection pattern detected: '{matched}'"

    print(
        f"[Guardrail Agent] pii_detected={pii_found} | "
        f"injection_detected={is_injection} | blocked={blocked}"
    )

    return {
        **state,
        "sanitized_message": redacted,
        "pii_detected": pii_found,
        "injection_detected": is_injection,
        "blocked": blocked,
        "guardrail_reason": reason,
    }
