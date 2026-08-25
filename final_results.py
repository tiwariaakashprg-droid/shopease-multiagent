"""
Aggregates evaluate.py's per-mode results into one summary file.
Run `python evaluate.py --mode all` first.
"""
import pandas as pd

df = pd.read_csv("research_results/retrieval_comparison_summary.csv")

with open("research_results/final_results.txt", "w", encoding="utf-8") as f:
    f.write("=" * 60 + "\n")
    f.write("SHOPEASE EXPERIMENTAL RESULTS (retrieval-driven classification)\n")
    f.write("=" * 60 + "\n\n")

    f.write("Accuracy Comparison\n")
    f.write("-" * 30 + "\n")
    for _, row in df.iterrows():
        f.write(f"{row['mode'].upper():12s}: {row['accuracy'] * 100:.2f}%\n")

    f.write("\nAverage Response Latency\n")
    f.write("-" * 30 + "\n")
    for _, row in df.iterrows():
        f.write(f"{row['mode'].upper():12s}: {row['avg_latency']:.5f} sec\n")

print("=" * 60)
print("Final results generated successfully -> research_results/final_results.txt")
print("=" * 60)
