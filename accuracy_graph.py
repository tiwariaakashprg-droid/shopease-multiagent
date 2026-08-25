import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Final verified retrieval-method accuracy results
# Dataset: 590 valid queries
# ---------------------------------------------------------

methods = [
    "BM25-only",
    "FAISS-only",
    "RRF-only (Fair)",
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

plt.ylabel("Accuracy (%)", fontsize=12)
plt.xlabel("Retrieval Method", fontsize=12)
plt.title("Accuracy Comparison of Retrieval Methods", fontsize=14)

plt.ylim(0, 100)

# Add percentage labels above bars
for bar, accuracy in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{accuracy:.2f}%",
        ha="center",
        fontsize=11
    )

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "research_results/accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Saved -> research_results/accuracy_comparison.png")