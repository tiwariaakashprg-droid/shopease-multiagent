"""
Ablation study (NEW) — quantifies what each agent actually contributes.

Two separate ablations, because the six agents don't all affect the same
metric:

  A) RETRIEVAL ABLATION (affects policy classification accuracy):
     BM25-only vs FAISS-only vs Hybrid-RRF vs Hybrid-RRF+Reranker
     -> reuses evaluate.py's per-mode results.

  B) COMPONENT ABLATION (affects response QUALITY, not classification,
     since CRM/Memory/Escalation don't change which policy is retrieved):
     Full pipeline vs (CRM disabled) vs (Memory disabled) vs
     (Escalation disabled) — scored with the LLM-judge rubric from
     llm_judge_eval.py on groundedness + personalization, run on a small
     sample since each condition needs its own full LLM generation pass.

Usage:
    python ablation_study.py --n 20
"""
import argparse
import copy
import pandas as pd

from graph.nodes.guardrail_agent import guardrail_agent
from graph.nodes.intent_agent import intent_agent
from graph.nodes.crm_agent import crm_agent
from graph.nodes.memory_agent import memory_agent
from graph.nodes.rag_agent import rag_agent
from graph.nodes.escalation_agent import escalation_agent
from graph.supervisor import supervisor_agent
from llm_judge_eval import judge_response


def run_pipeline(query, customer_id, chat_history, disable_crm=False,
                  disable_memory=False, disable_escalation=False):
    state = {
        "user_message": query,
        "customer_id": customer_id,
        "chat_history": chat_history,
        "agent_timings": {},
    }

    state = guardrail_agent(state)
    state = intent_agent(state)

    if disable_crm:
        state = {**state, "customer_data": {}, "customer_context": "Not available."}
    else:
        state = crm_agent(state)

    if disable_memory:
        state = {**state, "memory_context": "No previous conversation."}
    else:
        state = memory_agent(state)

    state = rag_agent(state)

    if disable_escalation:
        state = {**state, "should_escalate": False, "escalation_reason": "", "draft_reply": ""}
    else:
        state = escalation_agent(state)

    state = supervisor_agent(state)
    return state


def main(n: int, seed: int):
    df = pd.read_csv("data/test_queries_clean.csv").sample(n=n, random_state=seed).reset_index(drop=True)

    configs = {
        "full_pipeline": dict(),
        "no_crm": dict(disable_crm=True),
        "no_memory": dict(disable_memory=True),
        "no_escalation": dict(disable_escalation=True),
    }

    rows = []
    for _, row in df.iterrows():
        query, expected = row["query"], row["expected_policy"]

        for config_name, kwargs in configs.items():
            state = run_pipeline(query, "C1190", [], **kwargs)  # C1190 = VIP w/ complaint history, exercises CRM
            response = state.get("final_response") or state.get("draft_reply") or ""

            judge = judge_response(query, response, expected_policy=expected,
                                    customer_context=state.get("customer_context", ""))

            rows.append({
                "query": query,
                "config": config_name,
                "expected_policy": expected,
                "predicted_policy": state.get("policy_source"),
                "escalated": state.get("should_escalate"),
                "groundedness": judge.get("groundedness"),
                "personalization": judge.get("personalization"),
                "relevance": judge.get("relevance"),
                "overall": judge.get("overall"),
            })
            print(f"[{config_name}] {query[:45]:45s} -> overall={judge.get('overall')}")

    results = pd.DataFrame(rows)
    results.to_csv("research_results/ablation_study_results.csv", index=False)

    summary = results.groupby("config")[["groundedness", "personalization", "relevance", "overall"]].mean().round(2)
    summary.to_csv("research_results/ablation_study_summary.csv")

    print("\n" + "=" * 60)
    print("ABLATION SUMMARY (mean LLM-judge scores, 1-5 scale)")
    print("=" * 60)
    print(summary)
    print("\nSaved -> research_results/ablation_study_results.csv")
    print("Saved -> research_results/ablation_study_summary.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.n, args.seed)
