import pandas as pd

# Load evaluation results
df = pd.read_csv("evaluation_results.csv")

# Calculate Hybrid RAG results
hybrid_accuracy = (
    (df["expected"] == df["predicted"]).mean() * 100
)

hybrid_latency = df["latency"].mean()

# Baseline results (from your experiments)
bm25_accuracy = 76.64
faiss_accuracy = 76.64

bm25_latency = 0.00030
faiss_latency = 0.11586

# Write summary
with open(
    "final_project/results/final_results.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 60 + "\n")
    f.write("SHOPEASE EXPERIMENTAL RESULTS\n")
    f.write("=" * 60 + "\n\n")

    f.write("Accuracy Comparison\n")
    f.write("-" * 30 + "\n")
    f.write(f"BM25        : {bm25_accuracy:.2f}%\n")
    f.write(f"FAISS       : {faiss_accuracy:.2f}%\n")
    f.write(f"Hybrid RAG  : {hybrid_accuracy:.2f}%\n\n")

    f.write("Average Response Latency\n")
    f.write("-" * 30 + "\n")
    f.write(f"BM25        : {bm25_latency:.5f} sec\n")
    f.write(f"FAISS       : {faiss_latency:.5f} sec\n")
    f.write(f"Hybrid RAG  : {hybrid_latency:.5f} sec\n")

print("=" * 60)
print("Final results generated successfully.")
print("=" * 60)