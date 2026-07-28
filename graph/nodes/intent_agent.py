import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_ollama import OllamaLLM
from graph.state import AgentState
import json, re

llm = OllamaLLM(model="llama3.2", temperature=0.0)


def intent_agent(state: AgentState) -> AgentState:

    # Simple rule-based fallback + LLM
    message = state.get("user_message", "").lower()
    message = (
    message
    .replace("refnd", "refund")
    .replace("mony", "money")
    .replace("retrn", "return")
    .replace("cancle", "cancel")
    .replace("shiping", "shipping")
    .replace("pakage", "package")
    .replace("arived", "arrived")
    .replace("trak", "track")
    .replace("shipmnt", "shipment")
    .replace("screan", "screen")
    .replace("crackd", "cracked")
    .replace("damged", "damaged")
)

    import re
    message = re.sub(r"(please tell me|could you explain|i need help|urgent:|for my recent order|because i am confused|please|as soon as possible)",
                     "",message).strip()

    # Rule-based intent (fast + reliable)
    if any(w in message for w in [
        "damaged", "broken", "defective", "complaint", "faulty",
        "cracked", "shattered", "missing parts", "parts missing",
        "poor condition", "not functioning", "malfunctioning",
        "unusable", "damaged accessory", "damaged product"
    ]):
        intent = "complaint"

    elif any(w in message for w in [
    "cancel",
    "cancellation",
    "withdraw",
    "withdrawal",
    "revoke",
    "rescind",
    "abort order",
    "stop order",
    "cancel purchase",
    "don't want",
    "do not want",
    "no longer want",
    "before shipping"
]):
        intent = "cancellation"

    elif any(w in message for w in [
    "refund",
    "money back",
    "reimburse",
    "reimbursed",
    "payment back",
    "cash back",
    "recover payment",
    "reverse payment",
    "return my money",
    "get my money back",
    "get reimbursed",
    "receive compensation"
]):
        intent = "refund_request"

    elif any(w in message for w in [
    "return",
    "send back",
    "give this item back",
    "take this item back",
    "return process",
    "exchange",
    "exchange item",
    "exchange this product",
    "replacement",
    "replace"
]):
        intent = "return_item"

    elif any(w in message for w in [
    "where",
    "track",
    "tracking",
    "status",
    "delivery",
    "delivered",
    "shipment",
    "ship",
    "parcel",
    "package",
    "arrive",
    "arrival",
    "arrived",
    "reached",
    "reach",
    "international",
    "express delivery",
    "same day delivery",
    "delivery timeline",
    "delivery timelines"
]):
        intent = "order_tracking"

    elif "please tell me" in message or "could you explain" in message:
        intent = "product_query"

    elif "policy" in message:
        intent = "product_query"

    else:
        intent = "general"

    # Rule-based sentiment
    if any(w in message for w in ["angry", "furious", "unacceptable", "ridiculous", "legal", "sue", "worst"]):
        sentiment = "angry"
    elif any(w in message for w in ["frustrated", "disappointed", "terrible", "horrible"]):
        sentiment = "frustrated"
    elif any(w in message for w in ["worried", "concerned", "late", "delayed", "still"]):
        sentiment = "concerned"
    else:
        sentiment = "neutral"

    # Rule-based urgency
    if any(w in message for w in ["legal", "court", "immediately", "urgent", "asap", "now"]):
        urgency = "critical"
    elif any(w in message for w in ["late", "delayed", "waiting", "days"]):
        urgency = "high"
    elif any(w in message for w in ["when", "how long", "update"]):
        urgency = "medium"
    else:
        urgency = "low"

    print(f"[Intent Agent] intent={intent} | sentiment={sentiment} | urgency={urgency}")

    return {
        **state,
        "intent":    intent,
        "sentiment": sentiment,
        "urgency":   urgency,
    }