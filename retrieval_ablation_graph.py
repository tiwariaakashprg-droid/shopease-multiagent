import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Retrieval Ablation Study
# Dataset: 590 valid queries
# ---------------------------------------------------------

methods = [
    "BM25-only",
    "FAISS-only",
    "RRF-only",
    "RRF + Cross-Encoder"
]

accuracies = [
    50.34,
    92.71,
    83.39,
    90.68
]

plt.figure(figsize=(10, 6))

bars = plt.bar(methods, accuracies)

plt.xlabel("Retrieval Configuration", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)

plt.title(
    "Retrieval Ablation Study",
    fontsize=14
)

plt.ylim(0, 100)

# Percentage labels
for bar, accuracy in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center",
        fontsize=11
    )

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()

plt.savefig(
    "research_results/retrieval_ablation_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "Saved -> "
    "research_results/retrieval_ablation_comparison.png"
)