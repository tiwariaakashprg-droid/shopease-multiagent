import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


# ============================================================
# FINAL 2632-QUERY EVALUATION
# ============================================================

BASE_DIR = "research_results/final_2632_evaluation"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "confusion_matrices"
)


LABELS = [
    "Refund Policy",
    "Return Policy",
    "Shipping Policy",
    "Cancellation Policy",
    "Damaged Product Policy",
    "Unknown"
]


FILES = {
    "bm25": "evaluation_results_bm25.csv",
    "faiss": "evaluation_results_faiss.csv",
    "fair_rrf": "evaluation_results_fair_rrf.csv",
    "weighted_rrf": "evaluation_results_weighted_rrf.csv",
    "hybrid": "evaluation_results_hybrid.csv",
    "top10_hybrid": "evaluation_results_top10_hybrid.csv",
}


def generate_confusion_matrix(name, filename):

    input_path = os.path.join(
        BASE_DIR,
        filename
    )

    if not os.path.exists(input_path):

        raise FileNotFoundError(
            f"Input file not found:\n{input_path}"
        )

    df = pd.read_csv(
        input_path
    )

    # Safety check
    if len(df) != 2632:
        raise ValueError(
            f"{name} contains {len(df)} queries. "
            "Expected 2632."
        )

    # --------------------------------------------------------
    # Calculate confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        df["expected"],
        df["predicted"],
        labels=LABELS
    )

    # --------------------------------------------------------
    # Save numerical confusion matrix
    # --------------------------------------------------------

    cm_df = pd.DataFrame(
        cm,
        index=LABELS,
        columns=LABELS
    )

    csv_path = os.path.join(
        OUTPUT_DIR,
        f"confusion_matrix_{name}.csv"
    )

    cm_df.to_csv(
        csv_path
    )

    # --------------------------------------------------------
    # Generate PNG
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    image = ax.imshow(
        cm,
        interpolation="nearest"
    )

    ax.set_title(
        f"Confusion Matrix — {name}"
    )

    ax.set_xlabel(
        "Predicted Policy"
    )

    ax.set_ylabel(
        "Actual Policy"
    )

    ax.set_xticks(
        range(len(LABELS))
    )

    ax.set_yticks(
        range(len(LABELS))
    )

    ax.set_xticklabels(
        LABELS,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        LABELS
    )

    # --------------------------------------------------------
    # Cell values
    # --------------------------------------------------------

    threshold = cm.max() / 2.0

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] > threshold
                    else "black"
                )
            )

    fig.colorbar(
        image,
        ax=ax
    )

    plt.tight_layout()

    png_path = os.path.join(
        OUTPUT_DIR,
        f"confusion_matrix_{name}.png"
    )

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(
        f"\n{name.upper()}"
    )

    print(
        f"Queries  : {len(df)}"
    )

    print(
        f"Correct  : {df['correct'].sum()}"
    )

    print(
        f"Accuracy : {df['correct'].mean():.2%}"
    )

    print(
        f"CSV saved -> {csv_path}"
    )

    print(
        f"PNG saved -> {png_path}"
    )


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print(
        "FINAL 2632-QUERY CONFUSION MATRICES"
    )
    print("=" * 70)

    for name, filename in FILES.items():

        generate_confusion_matrix(
            name,
            filename
        )

    print("\n" + "=" * 70)
    print(
        "ALL 6 CONFUSION MATRICES SAVED"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()