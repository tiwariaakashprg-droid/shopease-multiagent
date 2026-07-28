import matplotlib.pyplot as plt

models = [
    "BM25",
    "FAISS",
    "Hybrid RAG"
]

latency = [
    0.00030,
    0.11586,
    0.63437
]

plt.figure(figsize=(8,6))

bars = plt.bar(
    models,
    latency,
    width=0.55
)

plt.xlabel("Retrieval Method", fontsize=12)
plt.ylabel("Average Latency (seconds)", fontsize=12)
plt.title("Average Response Latency Comparison", fontsize=14)

for bar, value in zip(bars, latency):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        value + 0.01,
        f"{value:.4f}s",
        ha="center",
        fontsize=11
    )

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "final_project/results/latency_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()