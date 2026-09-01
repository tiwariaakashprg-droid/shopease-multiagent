import os
import pandas as pd
from sklearn.metrics import classification_report


BASE_DIR = "research_results/final_2632_evaluation"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "classification_reports"
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


def generate_report(name, filename):

    input_path = os.path.join(
        BASE_DIR,
        filename
    )

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file not found:\n{input_path}"
        )

    df = pd.read_csv(input_path)

    # Safety check: final experiments must contain 2632 queries
    if len(df) != 2632:
        raise ValueError(
            f"{name} contains {len(df)} queries. "
            "Expected 2632."
        )

    report = classification_report(
        df["expected"],
        df["predicted"],
        labels=LABELS,
        digits=3,
        zero_division=0
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"classification_report_{name}.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"Classification Report — {name}\n"
        )

        f.write("=" * 70 + "\n")

        f.write(
            f"Input File    : {filename}\n"
        )

        f.write(
            f"Total Queries : {len(df)}\n"
        )

        f.write(
            f"Correct       : {df['correct'].sum()}\n"
        )

        f.write(
            f"Accuracy      : {df['correct'].mean():.2%}\n\n"
        )

        f.write(report)

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
        f"Saved -> {output_path}"
    )


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print(
        "FINAL 2632-QUERY CLASSIFICATION REPORTS"
    )
    print("=" * 70)

    for name, filename in FILES.items():

        generate_report(
            name,
            filename
        )

    print("\n" + "=" * 70)
    print(
        "ALL 6 CLASSIFICATION REPORTS SAVED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()