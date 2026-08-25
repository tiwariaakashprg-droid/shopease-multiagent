"""
McNemar's statistical significance test for retrieval methods.

Compares paired predictions on the SAME 595 queries:

    BM25-only
    FAISS-only
    Fair RRF-only

The Fair RRF result is the retrieval-only implementation without
cross-encoder reranking.
"""

import pandas as pd
from scipy.stats import chi2
import itertools


FILES = {
    "bm25": "research_results/evaluation_results_bm25.csv",
    "faiss": "research_results/evaluation_results_faiss.csv",
    "rrf": "research_results/hybrid_rrf_fair_results.csv",
}


def mcnemar_test(correct_a, correct_b):
    """
    McNemar's test for paired binary predictions.

    n01 = A wrong, B correct
    n10 = A correct, B wrong
    """

    n01 = sum(
        (a == 0 and b == 1)
        for a, b in zip(correct_a, correct_b)
    )

    n10 = sum(
        (a == 1 and b == 0)
        for a, b in zip(correct_a, correct_b)
    )

    n = n01 + n10

    if n == 0:
        return 0.0, 1.0, n01, n10

    # Continuity-corrected McNemar statistic
    statistic = ((abs(n01 - n10) - 1) ** 2) / n

    p_value = 1 - chi2.cdf(statistic, df=1)

    return statistic, p_value, n01, n10


def main():

    dfs = {}

    for name, path in FILES.items():

        try:
            dfs[name] = pd.read_csv(path)

        except FileNotFoundError:
            print(f"Missing file: {path}")
            print("Please check that the evaluation result exists.")
            return

    # --------------------------------------------------
    # Check that all datasets contain the same number
    # of queries.
    # --------------------------------------------------

    lengths = {
        name: len(df)
        for name, df in dfs.items()
    }

    print("=" * 65)
    print("McNemar's Test — Retrieval Method Comparison")
    print("=" * 65)

    print("\nDataset sizes:")
    for name, length in lengths.items():
        print(f"  {name.upper():10s}: {length}")

    if len(set(lengths.values())) != 1:
        print("\nWARNING: Result files have different lengths.")

    # --------------------------------------------------
    # Pairwise comparisons
    # --------------------------------------------------

    rows = []

    for a, b in itertools.combinations(dfs.keys(), 2):

        stat, p, n01, n10 = mcnemar_test(
            dfs[a]["correct"],
            dfs[b]["correct"]
        )

        significant = p < 0.05

        print(f"\n{a.upper()} vs {b.upper()}")

        print(
            f"  {a} wrong / {b} right : {n01}"
        )

        print(
            f"  {a} right / {b} wrong : {n10}"
        )

        print(
            f"  chi2 statistic        : {stat:.4f}"
        )

        print(
            f"  p-value               : {p:.6f}"
        )

        print(
            f"  statistically sig.    : "
            f"{'YES (p < 0.05)' if significant else 'NO (p >= 0.05)'}"
        )

        rows.append({
            "comparison": f"{a}_vs_{b}",
            "a_wrong_b_right": n01,
            "a_right_b_wrong": n10,
            "chi2_statistic": round(stat, 4),
            "p_value": round(p, 6),
            "significant_at_0.05": significant,
        })

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    output = pd.DataFrame(rows)

    output.to_csv(
        "research_results/significance_test_results.csv",
        index=False
    )

    print(
        "\nSaved -> "
        "research_results/significance_test_results.csv"
    )


if __name__ == "__main__":
    main()