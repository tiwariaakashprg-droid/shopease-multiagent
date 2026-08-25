from typing import TypedDict, Optional, List, Dict


class AgentState(TypedDict):
    """
    Shared state — har agent yahan se padhta hai aur yahan likhta hai.
    LangGraph is state ko ek node se doosre node tak carry karta hai.
    """
    # Input
    user_message:      str
    customer_id:       str
    chat_history:      list
    memory_context: Optional[str]

    # Guardrail Agent output (NEW)
    sanitized_message:   Optional[str]
    pii_detected:         Optional[bool]
    injection_detected:   Optional[bool]
    blocked:              Optional[bool]
    guardrail_reason:     Optional[str]

    # Intent Agent output
    intent:            Optional[str]
    sentiment:         Optional[str]
    urgency:           Optional[str]

    # CRM Agent output
    customer_data:     Optional[dict]
    customer_context:  Optional[str]

    # RAG Agent output
    policy_chunks:     Optional[str]
    policy_source:     Optional[str]
    retrieval_confidence: Optional[float]
    cited_sources:        Optional[List[str]]      # NEW - which policy files were actually used
    retrieval_method:     Optional[str]             # NEW - bm25 / faiss / hybrid_rrf / hybrid_rrf_rerank

    # Escalation Agent output
    should_escalate:   Optional[bool]
    escalation_reason: Optional[str]
    draft_reply:       Optional[str]

    # Supervisor output
    final_response:    Optional[str]
    next:              Optional[str]

    # Observability (NEW) - per-agent latency in seconds
    agent_timings:      Optional[Dict[str, float]]
