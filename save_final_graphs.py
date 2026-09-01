import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FINAL 2632-QUERY GRAPH GENERATION
# ============================================================

BASE_DIR = "research_results/final_2632_evaluation"


FILES = {
    "BM25": "evaluation_results_bm25.csv",
    "FAISS": "evaluation_results_faiss.csv",
    "Fair RRF": "evaluation_results_fair_rrf.csv",
    "Weighted RRF": "evaluation_results_weighted_rrf.csv",
    "Hybrid + CE": "evaluation_results_hybrid.csv",
    "Top-10 Hybrid + CE": "evaluation_results_top10_hybrid.csv",
}


# ============================================================
# Output directory
# ============================================================

GRAPH_DIR = os.path.join(
    BASE_DIR,
    "graphs"
)

os.makedirs(
    GRAPH_DIR,
    exist_ok=True
)


# ============================================================
# Load all final evaluation files
# ============================================================

data = {}

print("=" * 80)
print("FINAL 2632-QUERY GRAPH GENERATION")
print("=" * 80)

for name, filename in FILES.items():

    path = os.path.join(
        BASE_DIR,
        filename
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Missing evaluation file:\n{path}"
        )

    df = pd.read_csv(path)

    if len(df) != 2632:

        raise ValueError(
            f"{name} contains {len(df)} queries. "
            "Expected 2632."
        )

    data[name] = df

    print(
        f"{name:20s} | Queries: {len(df)} "
        f"| Accuracy: {df['correct'].mean():.2%}"
    )


# ============================================================
# 1. ACCURACY COMPARISON GRAPH
# ============================================================

accuracy_names = list(data.keys())

accuracy_values = [
    data[name]["correct"].mean() * 100
    for name in accuracy_names
]


plt.figure(
    figsize=(11, 7)
)

bars = plt.bar(
    accuracy_names,
    accuracy_values
)

plt.title(
    "Accuracy Comparison Across Retrieval Methods"
)

plt.xlabel(
    "Retrieval Method"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.ylim(
    0,
    100
)

plt.xticks(
    rotation=25,
    ha="right"
)

# Add exact values above bars
for bar, value in zip(
    bars,
    accuracy_values
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )


plt.tight_layout()

accuracy_path = os.path.join(
    GRAPH_DIR,
    "accuracy_comparison_2632.png"
)

plt.savefig(
    accuracy_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nSaved -> {accuracy_path}"
)


# ============================================================
# 2. LATENCY COMPARISON GRAPH
# ============================================================

latency_names = []
latency_values = []


for name, df in data.items():

    if "latency" not in df.columns:

        print(
            f"Skipping latency for {name}: "
            "latency not recorded."
        )

        continue

    latency = df["latency"].dropna()

    if len(latency) == 0:

        print(
            f"Skipping latency for {name}: "
            "no latency values."
        )

        continue

    latency_names.append(
        name
    )

    latency_values.append(
        latency.mean() * 1000
    )


plt.figure(
    figsize=(11, 7)
)

bars = plt.bar(
    latency_names,
    latency_values
)

plt.title(
    "Mean Latency Comparison Across Retrieval Methods"
)

plt.xlabel(
    "Retrieval Method"
)

plt.ylabel(
    "Mean Latency (ms)"
)

plt.xticks(
    rotation=25,
    ha="right"
)

for bar, value in zip(
    bars,
    latency_values
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.2f} ms",
        ha="center",
        va="bottom"
    )


plt.tight_layout()

latency_path = os.path.join(
    GRAPH_DIR,
    "latency_comparison_2632.png"
)

plt.savefig(
    latency_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved -> {latency_path}"
)


# ============================================================
# 3. RETRIEVAL / ABLATION COMPARISON GRAPH
# ============================================================

ablation_names = [
    "BM25",
    "FAISS",
    "Fair RRF",
    "Weighted RRF",
    "Hybrid + CE",
    "Top-10 Hybrid + CE"
]


ablation_values = [
    data[name]["correct"].mean() * 100
    for name in ablation_names
]


plt.figure(
    figsize=(11, 7)
)

bars = plt.bar(
    ablation_names,
    ablation_values
)

plt.title(
    "Retrieval Ablation Comparison — 2632 Queries"
)

plt.xlabel(
    "Retrieval Configuration"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.ylim(
    0,
    100
)

plt.xticks(
    rotation=25,
    ha="right"
)

for bar, value in zip(
    bars,
    ablation_values
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )


plt.tight_layout()

ablation_path = os.path.join(
    GRAPH_DIR,
    "retrieval_ablation_comparison_2632.png"
)

plt.savefig(
    ablation_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    f"Saved -> {ablation_path}"
)


# ============================================================
# 4. SAVE GRAPH DATA
# ============================================================

accuracy_df = pd.DataFrame({
    "method": accuracy_names,
    "accuracy_percent": accuracy_values
})

accuracy_df.to_csv(
    os.path.join(
        GRAPH_DIR,
        "accuracy_comparison_2632.csv"
    ),
    index=False
)


latency_df = pd.DataFrame({
    "method": latency_names,
    "mean_latency_ms": latency_values
})

latency_df.to_csv(
    os.path.join(
        GRAPH_DIR,
        "latency_comparison_2632.csv"
    ),
    index=False
)


ablation_df = pd.DataFrame({
    "method": ablation_names,
    "accuracy_percent": ablation_values
})

ablation_df.to_csv(
    os.path.join(
        GRAPH_DIR,
        "retrieval_ablation_comparison_2632.csv"
    ),
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("ALL FINAL 2632-QUERY GRAPHS SAVED")
print("=" * 80)

print(
    f"Graphs directory -> {GRAPH_DIR}"
)

print(
    "\nGenerated:"
)

print(
    "1. accuracy_comparison_2632.png"
)

print(
    "2. latency_comparison_2632.png"
)

print(
    "3. retrieval_ablation_comparison_2632.png"
)

print(
    "\nData files:"
)

print(
    "4. accuracy_comparison_2632.csv"
)

print(
    "5. latency_comparison_2632.csv"
)

print(
    "6. retrieval_ablation_comparison_2632.csv"
)

print("=" * 80)