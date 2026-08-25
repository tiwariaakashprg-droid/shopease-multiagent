import pandas as pd
import matplotlib.pyplot as plt

# Load ablation summary
path = "research_results/ablation_study_summary.csv"
df = pd.read_csv(path)

# Configuration names for better display
df["config"] = df["config"].replace({
    "full_pipeline": "Full Pipeline",
    "no_crm": "No CRM",
    "no_memory": "No Memory",
    "no_escalation": "No Escalation"
})

# Metrics to visualize
metrics = ["groundedness", "personalization", "relevance", "overall"]

# Create grouped bar chart
ax = df.set_index("config")[metrics].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.xlabel("System Configuration")
plt.ylabel("LLM-Judge Score (1–5)")
plt.title("Ablation Study of ShopEase Components")
plt.xticks(rotation=0)
plt.ylim(0, 5.5)
plt.legend(
    ["Groundedness", "Personalization", "Relevance", "Overall"],
    title="Metric"
)

plt.tight_layout()

# Save figure
output_path = "research_results/ablation_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved -> {output_path}")