"""
Baseline comparison (NEW): naive single-agent LLM chatbot vs ShopEase's
full multi-agent + Hybrid RAG pipeline.

This is the comparison reviewers ask for first: "how much does the
multi-agent + retrieval architecture actually buy you over just calling an
LLM directly?" The single-agent baseline (graph/nodes/single_agent.py) has
NO retrieval, NO CRM context, and NO conversation memory — it just answers
from the base model's own knowledge.

Since the single-agent baseline produces free-text (no policy_source
field), we score it by checking whether policy-category keywords appear
in its response — a conservative, reproducible proxy for "did it land on
the right policy topic at all". The full pipeline is scored the normal way
(policy_source == expected_policy).

Usage:
    python baseline_compare.py --n 100      # sample size (full 2632 is slow — single-agent has no shortcuts)
"""
import argparse
import time
import random
import pandas as pd

from graph.nodes.single_agent import single_agent_response
from graph.workflow import run_graph

POLICY_KEYWORDS = {
    "Refund Policy": ["refund", "money back", "reimburse"],
    "Return Policy": ["return", "exchange", "send back"],
    "Shipping Policy": ["shipping", "delivery", "tracking", "shipment", "courier"],
    "Cancellation Policy": ["cancel", "cancellation"],
    "Damaged Product Policy": ["damage", "damaged", "broken", "defective", "faulty"],
    "Unknown": [],
}


def keyword_score(response: str, expected: str) -> int:
    if expected == "Unknown":
        # can't reliably score "Unknown" via keyword presence — skip (counted separately)
        return -1
    keywords = POLICY_KEYWORDS.get(expected, [])
    response_l = response.lower()
    return int(any(kw in response_l for kw in keywords))


def main(n: int, seed: int):
    df = pd.read_csv("data/test_queries_clean.csv")
    df = df[df["expected_policy"] != "Unknown"]  # fair comparison only on scoreable categories
    sample = df.sample(n=min(n, len(df)), random_state=seed).reset_index(drop=True)

    rows = []
    for _, row in sample.iterrows():
        query, expected = row["query"], row["expected_policy"]

        # ---- Baseline: single-agent, no retrieval/CRM/memory ----
        t0 = time.time()
        baseline_response = single_agent_response(query)
        baseline_latency = time.time() - t0
        baseline_correct = keyword_score(baseline_response, expected)

        # ---- Full ShopEase pipeline ----
        t0 = time.time()
        state = run_graph(user_message=query, customer_id="C4521", chat_history=[])
        full_latency = time.time() - t0
        full_predicted = state.get("policy_source", "Unknown")
        full_correct = int(full_predicted == expected)

        rows.append({
            "query": query,
            "expected": expected,
            "baseline_correct": baseline_correct,
            "baseline_latency": baseline_latency,
            "shopease_predicted": full_predicted,
            "shopease_correct": full_correct,
            "shopease_latency": full_latency,
            "shopease_escalated": state.get("should_escalate", False),
        })
        print(f"  [{len(rows)}/{len(sample)}] baseline={baseline_correct} shopease={full_correct} | {query[:50]}")

    results = pd.DataFrame(rows)
    results.to_csv("research_results/baseline_comparison.csv", index=False)

    baseline_acc = results["baseline_correct"].mean()
    shopease_acc = results["shopease_correct"].mean()

    print("\n" + "=" * 55)
    print(f"Sample size              : {len(results)}")
    print(f"Baseline (single-agent)  : {baseline_acc:.2%} accuracy | "
          f"{results['baseline_latency'].mean():.3f}s avg latency")
    print(f"ShopEase (multi-agent)   : {shopease_acc:.2%} accuracy | "
          f"{results['shopease_latency'].mean():.3f}s avg latency")
    print(f"Absolute improvement     : {(shopease_acc - baseline_acc) * 100:.2f} points")
    print("=" * 55)
    print("Saved -> research_results/baseline_comparison.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.n, args.seed)

