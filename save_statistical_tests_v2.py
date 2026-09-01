import os
import itertools
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests


# ============================================================
# FINAL 2632-QUERY STATISTICAL SIGNIFICANCE ANALYSIS V2
#
# Methods:
#   BM25
#   FAISS
#   Fair RRF
#   Weighted RRF
#   Hybrid + Cross-Encoder
#   Top-10 Hybrid + Cross-Encoder
#
# Test:
#   Paired McNemar's test
#
# Multiple comparisons:
#   Holm correction
#
# Evaluation set:
#   2,632 identical held-out queries
# ============================================================


BASE_DIR = "research_results/final_2632_evaluation"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "statistical_tests"
)


FILES = {
    "BM25": "evaluation_results_bm25.csv",
    "FAISS": "evaluation_results_faiss.csv",
    "Fair RRF": "evaluation_results_fair_rrf.csv",
    "Weighted RRF": "evaluation_results_weighted_rrf.csv",
    "Hybrid + Cross-Encoder": "evaluation_results_hybrid.csv",
    "Top-10 Hybrid + Cross-Encoder": "evaluation_results_top10_hybrid.csv",
}


def load_results():
    """
    Load all final evaluation files and verify
    that they contain the same queries in the same order.
    """

    results = {}

    for method, filename in FILES.items():

        path = os.path.join(
            BASE_DIR,
            filename
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing evaluation file: {path}"
            )

        df = pd.read_csv(path)

        required = {
            "query",
            "expected",
            "predicted",
            "correct"
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"{method} is missing columns: {missing}"
            )

        results[method] = df.copy()

    return results


def verify_same_queries(results):
    """
    Ensure all methods were evaluated on exactly
    the same 2,632 queries in the same order.
    """

    methods = list(results.keys())

    reference = results[methods[0]]

    for method in methods[1:]:

        current = results[method]

        if len(current) != len(reference):
            raise ValueError(
                f"Query count mismatch: "
                f"{methods[0]}={len(reference)}, "
                f"{method}={len(current)}"
            )

        if not reference["query"].equals(
            current["query"]
        ):
            raise ValueError(
                f"Query ordering/content mismatch "
                f"between {methods[0]} and {method}"
            )

        if not reference["expected"].equals(
            current["expected"]
        ):
            raise ValueError(
                f"Expected-label mismatch "
                f"between {methods[0]} and {method}"
            )

    print(
        f"Verified: all {len(methods)} methods "
        f"use the same {len(reference)} queries."
    )


def run_mcnemar(results, method_a, method_b):

    a = results[method_a]["correct"].astype(bool)
    b = results[method_b]["correct"].astype(bool)

    both_correct = int((a & b).sum())

    a_correct_b_wrong = int(
        (a & ~b).sum()
    )

    a_wrong_b_correct = int(
        (~a & b).sum()
    )

    both_wrong = int(
        (~a & ~b).sum()
    )

    # McNemar contingency table
    table = [
        [both_correct, a_correct_b_wrong],
        [a_wrong_b_correct, both_wrong]
    ]

    # Exact=False + correction=True gives
    # continuity-corrected McNemar test.
    test = mcnemar(
        table,
        exact=False,
        correction=True
    )

    return {
        "method_a": method_a,
        "method_b": method_b,
        "queries": len(a),

        "accuracy_a": a.mean(),
        "accuracy_b": b.mean(),

        "both_correct": both_correct,
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "both_wrong": both_wrong,

        "chi_square": test.statistic,
        "p_value": test.pvalue,
    }


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 80)
    print("FINAL 2632-QUERY STATISTICAL SIGNIFICANCE ANALYSIS V2")
    print("=" * 80)

    results = load_results()

    verify_same_queries(results)

    methods = list(results.keys())

    print("\nMethods:")
    for method in methods:
        accuracy = results[method]["correct"].mean()
        print(
            f"  {method:32s} "
            f"{accuracy:.4%}"
        )

    print("\nRunning pairwise McNemar tests...")

    rows = []

    # All unique pairwise comparisons
    for method_a, method_b in itertools.combinations(
        methods,
        2
    ):

        result = run_mcnemar(
            results,
            method_a,
            method_b
        )

        rows.append(result)

    output = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Holm multiple-comparison correction
    # --------------------------------------------------------

    reject, corrected_p, _, _ = multipletests(
        output["p_value"],
        alpha=0.05,
        method="holm"
    )

    output["holm_adjusted_p_value"] = corrected_p
    output["significant_after_holm"] = reject

    # Raw significance too
    output["significant_raw"] = (
        output["p_value"] < 0.05
    )

    # Accuracy difference: B - A
    output["accuracy_difference"] = (
        output["accuracy_b"]
        - output["accuracy_a"]
    )

    # Percentage-point difference
    output["accuracy_difference_pp"] = (
        output["accuracy_difference"]
        * 100
    )

    # Reorder columns
    output = output[
        [
            "method_a",
            "method_b",
            "queries",
            "accuracy_a",
            "accuracy_b",
            "accuracy_difference",
            "accuracy_difference_pp",
            "both_correct",
            "a_correct_b_wrong",
            "a_wrong_b_correct",
            "both_wrong",
            "chi_square",
            "p_value",
            "significant_raw",
            "holm_adjusted_p_value",
            "significant_after_holm",
        ]
    ]

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        OUTPUT_DIR,
        "statistical_significance_tests_v2.csv"
    )

    output.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # Save human-readable TXT
    # --------------------------------------------------------

    txt_path = os.path.join(
        OUTPUT_DIR,
        "statistical_significance_tests_v2.txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "Statistical Significance Tests V2\n"
        )

        f.write(
            "=" * 80 + "\n\n"
        )

        f.write(
            "Test: McNemar's test\n"
        )

        f.write(
            "Significance level: alpha = 0.05\n"
        )

        f.write(
            "Continuity correction: applied\n"
        )

        f.write(
            "Multiple-comparison correction: "
            "Holm correction\n"
        )

        f.write(
            "Evaluation set: 2,632 held-out queries\n\n"
        )

        f.write(
            "=" * 80 + "\n"
        )

        for _, row in output.iterrows():

            f.write("\n")
            f.write(
                f"{row['method_a']} vs "
                f"{row['method_b']}\n"
            )

            f.write(
                "-" * 80 + "\n"
            )

            f.write(
                f"Queries: {int(row['queries'])}\n"
            )

            f.write(
                f"{row['method_a']} accuracy: "
                f"{row['accuracy_a']:.4%}\n"
            )

            f.write(
                f"{row['method_b']} accuracy: "
                f"{row['accuracy_b']:.4%}\n"
            )

            f.write(
                f"Accuracy difference "
                f"(B - A): "
                f"{row['accuracy_difference_pp']:.4f} "
                f"percentage points\n"
            )

            f.write("\n")

            f.write(
                "Contingency Table:\n"
            )

            f.write(
                "                B Correct   B Wrong\n"
            )

            f.write(
                f"A Correct       "
                f"{int(row['both_correct']):8d}   "
                f"{int(row['a_correct_b_wrong']):7d}\n"
            )

            f.write(
                f"A Wrong         "
                f"{int(row['a_wrong_b_correct']):8d}   "
                f"{int(row['both_wrong']):7d}\n"
            )

            f.write("\n")

            f.write(
                f"Chi-square: "
                f"{row['chi_square']:.6f}\n"
            )

            f.write(
                f"Raw p-value: "
                f"{row['p_value']:.10f}\n"
            )

            f.write(
                f"Raw significant at alpha=0.05: "
                f"{'YES' if row['significant_raw'] else 'NO'}\n"
            )

            f.write(
                f"Holm-adjusted p-value: "
                f"{row['holm_adjusted_p_value']:.10f}\n"
            )

            f.write(
                f"Significant after Holm correction: "
                f"{'YES' if row['significant_after_holm'] else 'NO'}\n"
            )

            f.write(
                "\n" + "-" * 80 + "\n"
            )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    for _, row in output.iterrows():

        print(
            f"\n{row['method_a']} vs "
            f"{row['method_b']}"
        )

        print(
            f"Accuracy: "
            f"{row['accuracy_a']:.2%} vs "
            f"{row['accuracy_b']:.2%}"
        )

        print(
            f"Difference: "
            f"{row['accuracy_difference_pp']:.3f} pp"
        )

        print(
            f"Raw p-value: "
            f"{row['p_value']:.8f}"
        )

        print(
            f"Holm-adjusted p-value: "
            f"{row['holm_adjusted_p_value']:.8f}"
        )

        print(
            "Significant after Holm:",
            "YES"
            if row["significant_after_holm"]
            else "NO"
        )

    print("\n" + "=" * 80)
    print("FILES SAVED")
    print("=" * 80)

    print(
        f"CSV -> {csv_path}"
    )

    print(
        f"TXT -> {txt_path}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()