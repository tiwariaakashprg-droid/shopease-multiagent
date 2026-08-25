"""
Retrieval-method comparison (BM25 vs FAISS vs Hybrid RAG).

Now that policy classification is retrieval-driven (see
graph/nodes/retrieval_utils.py) instead of intent-keyword-driven, this
script produces an HONEST accuracy comparison across the three retrieval
strategies, run separately so results can be paired for the McNemar
significance test in significance_test.py.

Usage:
    python evaluate.py --mode bm25
    python evaluate.py --mode faiss
    python evaluate.py --mode hybrid
    python evaluate.py --mode all      # runs all three and saves each
"""
import argparse
import pandas as pd
import time
from graph.nodes.rag_agent import rag_agent
from graph.nodes.rag_faiss import rag_faiss
from graph.nodes.rag_bm25 import rag_bm25
from graph.nodes.intent_agent import intent_agent
from graph.nodes.guardrail_agent import guardrail_agent


def run_eval(mode: str, dataset_path: str = "data/test_queries_clean.csv"):
    df = pd.read_csv(dataset_path)
    correct = 0
    total = len(df)
    results = []

    for _, row in df.iterrows():
        query = row["query"]
        expected = row["expected_policy"]

        start = time.time()

        state = {
            "user_message": query,
            "customer_id": "C4521",
            "chat_history": [],
            "agent_timings": {},
        }

        state = guardrail_agent(state)
        state = intent_agent(state)

        if mode == "hybrid":
            state = rag_agent(state)
        elif mode == "faiss":
            state = rag_faiss(state)
        elif mode == "bm25":
            state = rag_bm25(state)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        latency = time.time() - start

        predicted = state.get("policy_source", "Unknown")
        confidence = state.get("retrieval_confidence", 0.0)

        is_correct = int(predicted == expected)
        correct += is_correct

        results.append({
            "query": query,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "confidence": confidence,
            "retrieval_method": state.get("retrieval_method", mode),
            "latency": latency,
        })

    accuracy = correct / total
    results_df = pd.DataFrame(results)
    avg_latency = results_df["latency"].mean()

    out_path = f"research_results/evaluation_results_{mode}.csv"
    results_df.to_csv(out_path, index=False)

    print("=" * 50)
    print(f"Mode          : {mode}")
    print(f"Total Queries : {total}")
    print(f"Correct       : {correct}")
    print(f"Accuracy      : {accuracy:.2%}")
    print(f"Avg Latency   : {avg_latency:.4f} sec")
    print(f"Saved         : {out_path}")
    print("=" * 50)

    wrong = results_df[results_df["expected"] != results_df["predicted"]]
    wrong.to_csv(f"research_results/wrong_predictions_{mode}.csv", index=False)

    return {"mode": mode, "accuracy": accuracy, "avg_latency": avg_latency,
            "correct": correct, "total": total}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["bm25", "faiss", "hybrid", "all"], default="hybrid")
    parser.add_argument("--dataset", default="data/test_queries_clean.csv")
    args = parser.parse_args()

    if args.mode == "all":
        summary = []
        for m in ["bm25", "faiss", "hybrid"]:
            summary.append(run_eval(m, args.dataset))
        pd.DataFrame(summary).to_csv("research_results/retrieval_comparison_summary.csv", index=False)
        print("\nSaved combined summary -> research_results/retrieval_comparison_summary.csv")
    else:
        run_eval(args.mode, args.dataset)
