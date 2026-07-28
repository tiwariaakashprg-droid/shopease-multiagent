import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load results
df = pd.read_csv("evaluation_results.csv")

# Labels
labels = [
    "Refund Policy",
    "Return Policy",
    "Shipping Policy",
    "Cancellation Policy",
    "Damaged Product Policy",
    "Unknown"
]

# Confusion Matrix
cm = confusion_matrix(
    df["expected"],
    df["predicted"],
    labels=labels
)

# Plot
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(
    cmap="Blues",
    ax=ax,
    xticks_rotation=45
)

plt.title("Confusion Matrix of the Proposed ShopEase Framework")
plt.tight_layout()

# Save image
plt.savefig("final_project/results/confusion_matrix.png")

# Show
plt.show()
print("Confusion Matrix Saved Successfully")