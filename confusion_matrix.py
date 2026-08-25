"""
Confusion matrix for the final Hybrid RAG system.
Final system = RRF + Cross-Encoder.
Dataset = 590 valid queries.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

INPUT_FILE = "research_results/evaluation_results_hybrid.csv"
OUTPUT_FILE = "research_results/confusion_matrix.png"

df = pd.read_csv(INPUT_FILE)

labels = [
    "Refund Policy",
    "Return Policy",
    "Shipping Policy",
    "Cancellation Policy",
    "Damaged Product Policy",
    "Unknown"
]

cm = confusion_matrix(
    df["expected"],
    df["predicted"],
    labels=labels
)

fig, ax = plt.subplots(figsize=(10, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45
)

plt.title(
    "Confusion Matrix — ShopEase Hybrid RAG\n"
    "RRF + Cross-Encoder"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"Saved -> {OUTPUT_FILE}")