import pandas as pd
import time
from graph.workflow import run_graph
from graph.nodes.rag_agent import rag_agent
from graph.nodes.rag_faiss import rag_faiss
from graph.nodes.rag_bm25 import rag_bm25
from graph.nodes.intent_agent import intent_agent

#EVAL_MODE = "bm25"
#EVAL_MODE = "faiss"
EVAL_MODE = "hybrid"
#EVAL_MODE = "full"

df = pd.read_csv("data/test_queries_clean.csv")

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
        "chat_history": []
    }

    state = intent_agent(state)

    if EVAL_MODE == "hybrid":
        state = rag_agent(state)

    elif EVAL_MODE == "faiss":
        state = rag_faiss(state)

    elif EVAL_MODE == "bm25":
        state = rag_bm25(state)

    else:
        state = run_graph(
            user_message=query,
            customer_id="C4521",
            chat_history=[]
        )

    end = time.time()
    latency = end - start

    predicted = state.get(
        "policy_source",
        "Unknown"
    )

    confidence = state.get(
        "retrieval_confidence",
        0.0
    )

    escalate = state.get(
        "should_escalate",
        False
    )

    if predicted == expected:
        correct += 1

    results.append({
        "query": query,
        "expected": expected,
        "predicted": predicted,
        "confidence": confidence,
        "escalated": escalate,
        "latency": latency,
    })

accuracy = correct / total

pd.DataFrame(results).to_csv(
    "evaluation_results.csv",
    index=False
)

avg_latency = (
    pd.DataFrame(results)["latency"].mean()
)

print("=" * 50)
print(f"Mode          : {EVAL_MODE}")
print(f"Total Queries : {total}")
print(f"Correct       : {correct}")
print(f"Accuracy      : {accuracy:.2%}")
print(f"Avg Time      : {avg_latency:.5f} sec")
print("=" * 50)

wrong = pd.DataFrame(results)
wrong = wrong[
    wrong["expected"] != wrong["predicted"]
]

wrong.to_csv(
    "wrong_predictions.csv",
    index=False
)

print("Saved wrong_predictions.csv")