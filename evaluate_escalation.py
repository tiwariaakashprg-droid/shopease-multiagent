"""
Escalation Agent evaluation (NEW) — precision / recall / F1 / confusion
matrix on a hand-labeled test set (data/escalation_test_set.csv).

The original paper only showed ONE qualitative HITL example (Fig. 7). This
gives the Escalation Agent an actual quantitative evaluation, which is
needed to claim the HITL mechanism "works" rather than just "exists".

Usage:
    python evaluate_escalation.py
"""
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

from graph.nodes.guardrail_agent import guardrail_agent
from graph.nodes.intent_agent import intent_agent
from graph.nodes.crm_agent import crm_agent
from graph.nodes.rag_agent import rag_agent
from graph.nodes.escalation_agent import escalation_agent


def main():
    df = pd.read_csv("data/escalation_test_set.csv")

    y_true = []
    y_pred = []
    rows = []

    for _, row in df.iterrows():
        query = row["query"]
        customer_id = row["customer_id"]
        expected = bool(row["expected_escalate"])

        state = {
            "user_message": query,
            "customer_id": customer_id,
            "chat_history": [],
            "agent_timings": {},
        }
        state = guardrail_agent(state)
        state = intent_agent(state)
        state = crm_agent(state)
        state = rag_agent(state)
        state = escalation_agent(state)

        predicted = bool(state.get("should_escalate"))

        y_true.append(expected)
        y_pred.append(predicted)

        rows.append({
            "query": query,
            "customer_id": customer_id,
            "trigger_type": row.get("trigger_type", ""),
            "expected_escalate": expected,
            "predicted_escalate": predicted,
            "escalation_reason": state.get("escalation_reason", ""),
            "correct": expected == predicted,
        })
        status = "OK" if expected == predicted else "MISS"
        print(f"[{status}] expected={expected} predicted={predicted} | {query[:55]}")

    results = pd.DataFrame(rows)
    results.to_csv("research_results/escalation_evaluation_results.csv", index=False)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[True, False])

    print("\n" + "=" * 50)
    print("ESCALATION AGENT — EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1-score  : {f1:.3f}")
    print("\nConfusion Matrix (rows=actual, cols=predicted) [True, False]:")
    print(cm)
    print("\nPer-trigger-type accuracy:")
    print(results.groupby("trigger_type")["correct"].mean().round(2))
    print("\nSaved -> research_results/escalation_evaluation_results.csv")

    with open("research_results/escalation_evaluation_summary.txt", "w") as f:
        f.write(f"Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}\n\n")
        f.write("Confusion Matrix [True, False] x [True, False]:\n")
        f.write(str(cm) + "\n\n")
        f.write(classification_report(y_true, y_pred, target_names=["No Escalation", "Escalate"], zero_division=0))


if __name__ == "__main__":
    main()
