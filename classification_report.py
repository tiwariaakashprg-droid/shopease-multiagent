"""
Per-class precision/recall/F1 report for RRF + Cross-Encoder.

Uses the actual evaluation output:
research_results/evaluation_results_hybrid.csv
"""

import pandas as pd
from sklearn.metrics import classification_report

INPUT_PATH = "research_results/evaluation_results_hybrid.csv"

df = pd.read_csv(INPUT_PATH)

labels = [
    "Refund Policy",
    "Return Policy",
    "Shipping Policy",
    "Cancellation Policy",
    "Damaged Product Policy",
    "Unknown"
]

report = classification_report(
    df["expected"],
    df["predicted"],
    labels=labels,
    digits=3,
    zero_division=0
)

print("=" * 60)
print("Classification Report — RRF + Cross-Encoder")
print("=" * 60)
print(report)

with open(
    "research_results/classification_report.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("Classification Report — RRF + Cross-Encoder\n")
    f.write("=" * 60 + "\n")
    f.write(report)

print("\nSaved -> research_results/classification_report.txt")