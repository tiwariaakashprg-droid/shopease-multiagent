import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# FINAL 2632 EVALUATION - COMPONENT ABLATION GRAPH
# Uses the final 100-query component ablation results.
# ============================================================

input_path = (
    "research_results/final_2632_evaluation/"
    "ablation_study/ablation_study_summary_2632.csv"
)

output_path = (
    "research_results/final_2632_evaluation/"
    "ablation_study/ablation_comparison_100.png"
)

# Load final ablation summary
df = pd.read_csv(input_path)

# Configuration names for publication-quality display
df["config"] = df["config"].replace({
    "full_pipeline": "Full Pipeline",
    "no_crm": "No CRM",
    "no_memory": "No Memory",
    "no_escalation": "No Escalation"
})

# Keep the intended configuration order
config_order = [
    "Full Pipeline",
    "No CRM",
    "No Memory",
    "No Escalation"
]

df["config"] = pd.Categorical(
    df["config"],
    categories=config_order,
    ordered=True
)

df = df.sort_values("config")

# Metrics
metrics = [
    "groundedness",
    "personalization",
    "relevance",
    "overall"
]

# Create grouped bar chart
ax = df.set_index("config")[metrics].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.xlabel("System Configuration", fontsize=12)
plt.ylabel("LLM-Judge Score (1–5)", fontsize=12)

plt.title(
    "Component Ablation Study of ShopEase",
    fontsize=14
)

plt.xticks(rotation=0)
plt.ylim(0, 5.5)

plt.legend(
    ["Groundedness", "Personalization", "Relevance", "Overall"],
    title="Metric"
)

plt.tight_layout()

# Save high-resolution figure
plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("=" * 70)
print("FINAL COMPONENT ABLATION GRAPH")
print("=" * 70)
print(f"Input  -> {input_path}")
print(f"Output -> {output_path}")
print("Queries evaluated in ablation : 100")
print("Configurations                 : 4")
print("Total LLM-judge evaluations     : 400")
print("=" * 70)