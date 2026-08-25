"""
LLM-as-judge response quality evaluation (NEW).

The paper's original 83.19% number measures policy CLASSIFICATION
accuracy only — it says nothing about whether the actual generated
response was correct, grounded in the retrieved policy/CRM data, or
appropriately personalized. This script closes that gap using a
separate judge LLM (same local model by default — swap to a stronger
model if available) that rates each response on a 1-5 rubric across:

  - relevance       : does the response address the customer's question?
  - groundedness     : does it stick to the provided CRM/policy context
                        instead of hallucinating (order IDs, dates, amounts)?
  - personalization  : does it use the customer's name/tier/order details?
  - overall          : holistic quality score

Usage (standalone):
    python llm_judge_eval.py --n 30
"""
import argparse
import json
import re
import pandas as pd
from langchain_ollama import OllamaLLM

from graph.workflow import run_graph

_judge_llm = OllamaLLM(model="llama3.2", temperature=0.0)

JUDGE_PROMPT = """You are an impartial evaluator for an enterprise customer support system.
Rate the AGENT RESPONSE below on four criteria, each from 1 (poor) to 5 (excellent).

CUSTOMER QUESTION:
{query}

EXPECTED POLICY CATEGORY:
{expected_policy}

CUSTOMER CONTEXT PROVIDED TO THE AGENT:
{customer_context}

AGENT RESPONSE:
{response}

Rate strictly using this JSON schema and output ONLY valid JSON, nothing else:
{{
  "relevance": <1-5, does it address the actual question?>,
  "groundedness": <1-5, does it avoid inventing facts not present in the customer context / policy?>,
  "personalization": <1-5, does it use the customer's actual details when available?>,
  "overall": <1-5, holistic quality>
}}
"""


def judge_response(query: str, response: str, expected_policy: str = "", customer_context: str = "") -> dict:
    prompt = JUDGE_PROMPT.format(
        query=query,
        expected_policy=expected_policy or "N/A",
        customer_context=customer_context or "Not available",
        response=response or "(empty response)",
    )
    raw = _judge_llm.invoke(prompt).strip()

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"relevance": None, "groundedness": None, "personalization": None, "overall": None}

    try:
        parsed = json.loads(match.group(0))
        return {
            "relevance": parsed.get("relevance"),
            "groundedness": parsed.get("groundedness"),
            "personalization": parsed.get("personalization"),
            "overall": parsed.get("overall"),
        }
    except json.JSONDecodeError:
        return {"relevance": None, "groundedness": None, "personalization": None, "overall": None}


def main(n: int, seed: int):
    df = pd.read_csv("data/test_queries_clean.csv").sample(n=n, random_state=seed).reset_index(drop=True)

    rows = []
    for _, row in df.iterrows():
        query, expected = row["query"], row["expected_policy"]

        state = run_graph(user_message=query, customer_id="C1190", chat_history=[])
        response = state.get("final_response") or state.get("draft_reply") or ""

        scores = judge_response(query, response, expected, state.get("customer_context", ""))

        rows.append({
            "query": query,
            "expected_policy": expected,
            "predicted_policy": state.get("policy_source"),
            "escalated": state.get("should_escalate"),
            "response": response,
            **scores,
        })
        print(f"[{len(rows)}/{n}] overall={scores.get('overall')} | {query[:50]}")

    results = pd.DataFrame(rows)
    results.to_csv("research_results/llm_judge_results.csv", index=False)

    numeric_cols = ["relevance", "groundedness", "personalization", "overall"]
    means = results[numeric_cols].mean(numeric_only=True).round(2)

    print("\n" + "=" * 50)
    print("LLM-AS-JUDGE — MEAN SCORES (1-5 scale)")
    print("=" * 50)
    print(means)
    print("\nSaved -> research_results/llm_judge_results.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.n, args.seed)
