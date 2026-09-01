import os
import pandas as pd


BASE_DIR = "research_results/final_2632_evaluation"

ERROR_DIR = os.path.join(
    BASE_DIR,
    "error_analysis"
)


FILES = {
    "bm25": "evaluation_results_bm25.csv",
    "faiss": "evaluation_results_faiss.csv",
    "fair_rrf": "evaluation_results_fair_rrf.csv",
    "weighted_rrf": "evaluation_results_weighted_rrf.csv",
    "hybrid": "evaluation_results_hybrid.csv",
    "top10_hybrid": "evaluation_results_top10_hybrid.csv",
}


def analyze_errors(name, filename):

    path = os.path.join(
        BASE_DIR,
        filename
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    df = pd.read_csv(path)

    errors = df[
        df["correct"] == 0
    ].copy()

    total_errors = len(errors)

    # --------------------------------------------------------
    # Error pair counts
    # --------------------------------------------------------

    pairs = (
        errors
        .groupby(
            ["expected", "predicted"]
        )
        .size()
        .reset_index(
            name="error_count"
        )
        .sort_values(
            "error_count",
            ascending=False
        )
    )

    # --------------------------------------------------------
    # Percentage of all errors
    # --------------------------------------------------------

    if total_errors > 0:
        pairs["error_percentage"] = (
            pairs["error_count"]
            / total_errors
            * 100
        )
    else:
        pairs["error_percentage"] = 0.0

    pairs["error_percentage"] = (
        pairs["error_percentage"]
        .round(2)
    )

    # --------------------------------------------------------
    # Save error-pair CSV
    # --------------------------------------------------------

    output_path = os.path.join(
        ERROR_DIR,
        f"{name}_error_pairs.csv"
    )

    pairs.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------------
    # Save all wrong queries separately
    # --------------------------------------------------------

    wrong_queries_path = os.path.join(
        ERROR_DIR,
        f"{name}_wrong_queries.csv"
    )

    errors.to_csv(
        wrong_queries_path,
        index=False
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        f"{name.upper()} ERROR ANALYSIS"
    )

    print("=" * 70)

    print(
        f"Total Queries : {len(df)}"
    )

    print(
        f"Correct       : {df['correct'].sum()}"
    )

    print(
        f"Accuracy      : {df['correct'].mean():.2%}"
    )

    print(
        f"Total Errors  : {total_errors}"
    )

    print("\nTop Error Pairs:")

    print(
        pairs.head(10).to_string(
            index=False
        )
    )

    print(
        f"\nSaved -> {output_path}"
    )

    print(
        f"Saved -> {wrong_queries_path}"
    )


def main():

    os.makedirs(
        ERROR_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print("FINAL 2632-QUERY ERROR ANALYSIS")
    print("=" * 70)

    for name, filename in FILES.items():

        analyze_errors(
            name,
            filename
        )

    print("\n" + "=" * 70)
    print("ALL ERROR ANALYSIS FILES SAVED")
    print("=" * 70)


if __name__ == "__main__":
    main()