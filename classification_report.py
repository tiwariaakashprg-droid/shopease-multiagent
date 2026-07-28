import pandas as pd
from sklearn.metrics import classification_report

# Load evaluation results
df = pd.read_csv("evaluation_results.csv")

labels = [
    "Refund Policy",
    "Return Policy",
    "Shipping Policy",
    "Cancellation Policy",
    "Damaged Product Policy",
    "Unknown"
]

# Generate classification report
report = classification_report(
    df["expected"],
    df["predicted"],
    labels=labels,
    digits=3,
    zero_division=0
)

print("=" * 60)
print("Classification Report")
print("=" * 60)
print(report)

# Save report
with open(
    "final_project/results/classification_report.txt",
    "w",
    encoding="utf-8"
) as f:
    f.write("Classification Report\n")
    f.write("=" * 60 + "\n")
    f.write(report)

print("\nClassification report saved successfully.")