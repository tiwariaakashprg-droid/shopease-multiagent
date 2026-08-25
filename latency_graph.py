"""
Latency comparison for the evaluated retrieval methods.

Methods:
1. BM25-only
2. FAISS-only
3. RRF + Cross-Encoder
"""

import pandas as pd
import matplotlib.pyplot as plt

FILES = {
    "BM25-only": "research_results/evaluation_results_bm25.csv",
    "FAISS-only": "research_results/evaluation_results_faiss.csv",
    "RRF + Cross-Encoder": "research_results/evaluation_results_hybrid.csv",
}

methods = []
latencies = []

for method, path in FILES.items():
    df = pd.read_csv(path)

    avg_latency = df["latency"].mean()

    methods.append(method)
    latencies.append(avg_latency)

    print(f"{method}: {avg_latency:.4f} seconds")


plt.figure(figsize=(9, 6))

bars = plt.bar(methods, latencies, width=0.55)

plt.xlabel("Retrieval Method", fontsize=12)
plt.ylabel("Average Latency (seconds)", fontsize=12)
plt.title("Average Response Latency Comparison", fontsize=14)

for bar, value in zip(bars, latencies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + max(latencies) * 0.02,
        f"{value:.4f}s",
        ha="center",
        fontsize=11
    )

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "research_results/latency_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nSaved -> research_results/latency_comparison.png")