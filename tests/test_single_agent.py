import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph.nodes.single_agent import single_agent_response

queries = [
    "Where is my order?",
    "I want a refund",
    "My laptop is damaged",
    "I will file a consumer court case and need a refund of Rs 10000"
]

for q in queries:
    print("\n" + "=" * 50)
    print("QUERY:", q)
    print("=" * 50)

    response = single_agent_response(q)

    print(response)