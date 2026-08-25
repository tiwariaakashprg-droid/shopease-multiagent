import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from graph.nodes.single_agent import single_agent_response
from graph.workflow import run_graph

queries = [
    "Where is my order?",
    "I want a refund",
    "My laptop is damaged",
    "I will file a consumer court case and need a refund of Rs 10000"
]

for q in queries:

    print("\n" + "=" * 80)
    print("QUERY:", q)
    print("=" * 80)

    print("\n[SINGLE AGENT]")
    print(single_agent_response(q))
    print("\n[MULTI AGENT]")

    result = run_graph(
    q,
    customer_id="C4521",
    chat_history=[]
)

    print(
        result.get(
            "final_response",
            "No response"
        )
    )