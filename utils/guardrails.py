"""
Guardrails for an enterprise-facing agentic system.

Two independent checks are implemented, both important for an enterprise
deployment and both flagged as missing in the paper's Limitations section:

1. PII redaction  - customer-typed emails / phone numbers / card numbers are
   masked before the message is logged or sent to the LLM as few-shot
   context, so raw PII never sits in prompt logs.
2. Prompt-injection detection - a lightweight heuristic scan for the most
   common jailbreak / instruction-override patterns aimed at agentic
   customer-support bots (e.g. "ignore previous instructions", "you are now
   DAN", "reveal your system prompt", fake "admin override" messages).

Both are intentionally rule-based (fast, deterministic, auditable, no extra
model calls) rather than ML classifiers, since latency budget in the paper
is already tight (0.63 s average).
"""
import re

# ---------------------------------------------------------------------------
# PII redaction
# ---------------------------------------------------------------------------
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?\d{10}(?!\d)")
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)")
_AADHAAR_RE = re.compile(r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)")


def redact_pii(text: str) -> tuple[str, bool]:
    """Returns (redacted_text, pii_was_found)."""
    if not text:
        return text, False

    found = False
    redacted = text

    if _EMAIL_RE.search(redacted):
        redacted = _EMAIL_RE.sub("[REDACTED_EMAIL]", redacted)
        found = True

    if _CARD_RE.search(redacted):
        redacted = _CARD_RE.sub("[REDACTED_CARD]", redacted)
        found = True

    if _PHONE_RE.search(redacted):
        redacted = _PHONE_RE.sub("[REDACTED_PHONE]", redacted)
        found = True

    if _AADHAAR_RE.search(redacted):
        redacted = _AADHAAR_RE.sub("[REDACTED_ID]", redacted)
        found = True

    return redacted, found


# ---------------------------------------------------------------------------
# Prompt-injection / jailbreak detection
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    r"ignore (all|any|the) (previous|prior|above) instructions",
    r"disregard (all|any|the) (previous|prior|above)",
    r"you are now (a|an|dan|jailbroken|unrestricted)",
    r"reveal (your|the) (system|hidden) prompt",
    r"print (your|the) (system|initial) prompt",
    r"act as (an? )?(admin|developer|root|system)",
    r"pretend (you are|to be) (an? )?(admin|unfiltered|unrestricted)",
    r"bypass (your|the|any) (safety|content|filter)",
    r"do anything now",
    r"\bsudo\b",
    r"override (your|the) (instructions|rules|policy)",
    r"this is (an? )?(admin|system) override",
    r"new system prompt",
    r"</?(system|assistant|instructions)>",
]
_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)


def detect_prompt_injection(text: str) -> tuple[bool, str]:
    """Returns (is_suspicious, matched_pattern_or_empty)."""
    if not text:
        return False, ""

    match = _INJECTION_RE.search(text)
    if match:
        return True, match.group(0)
    return False, ""
