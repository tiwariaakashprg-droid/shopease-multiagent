import matplotlib.pyplot as plt

models = [
    "BM25",
    "FAISS",
    "Hybrid RAG"
]

accuracy = [
    76.64,
    76.64,
    83.19
]

plt.figure(figsize=(8,6))

bars = plt.bar(
    models,
    accuracy,
    width=0.55
)

plt.ylim(0,100)

plt.xlabel("Retrieval Method", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("Accuracy Comparison of Retrieval Methods", fontsize=14)

for bar, value in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        value + 1,
        f"{value:.2f}%",
        ha="center",
        fontsize=11
    )

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "final_project/results/accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()