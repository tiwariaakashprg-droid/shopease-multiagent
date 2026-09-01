import os
import pandas as pd


BASE_DIR = "research_results/final_2632_evaluation"

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "latency_analysis"
)


FILES = {
    "bm25": "evaluation_results_bm25.csv",
    "faiss": "evaluation_results_faiss.csv",
    "fair_rrf": "evaluation_results_fair_rrf.csv",
    "weighted_rrf": "evaluation_results_weighted_rrf.csv",
    "hybrid": "evaluation_results_hybrid.csv",
    "top10_hybrid": "evaluation_results_top10_hybrid.csv",
}


def analyze_latency(name, filename):

    input_path = os.path.join(
        BASE_DIR,
        filename
    )

    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Input file not found:\n{input_path}"
        )

    df = pd.read_csv(input_path)

    if len(df) != 2632:
        raise ValueError(
            f"{name} contains {len(df)} queries. "
            "Expected 2632."
        )

    # Fair RRF does not contain latency in the
    # saved evaluation CSV.
    if "latency" not in df.columns:

        return {
            "experiment": name,
            "total_queries": len(df),
            "latency_samples": 0,
            "mean_latency_sec": None,
            "median_latency_sec": None,
            "min_latency_sec": None,
            "max_latency_sec": None,
            "std_latency_sec": None,
            "latency_status": "Not recorded"
        }

    latency = df["latency"].dropna()

    if len(latency) == 0:

        return {
            "experiment": name,
            "total_queries": len(df),
            "latency_samples": 0,
            "mean_latency_sec": None,
            "median_latency_sec": None,
            "min_latency_sec": None,
            "max_latency_sec": None,
            "std_latency_sec": None,
            "latency_status": "Not available"
        }

    return {
        "experiment": name,
        "total_queries": len(df),
        "latency_samples": len(latency),
        "mean_latency_sec": latency.mean(),
        "median_latency_sec": latency.median(),
        "min_latency_sec": latency.min(),
        "max_latency_sec": latency.max(),
        "std_latency_sec": latency.std(),
        "latency_status": "Available"
    }


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 80)
    print("FINAL 2632-QUERY LATENCY ANALYSIS")
    print("=" * 80)

    results = []

    for name, filename in FILES.items():

        result = analyze_latency(
            name,
            filename
        )

        results.append(result)

        print(
            f"\n{name.upper()}"
        )

        print(
            f"Queries         : {result['total_queries']}"
        )

        print(
            f"Latency samples : {result['latency_samples']}"
        )

        print(
            f"Status          : {result['latency_status']}"
        )

        if result["latency_status"] == "Available":

            print(
                f"Mean            : "
                f"{result['mean_latency_sec']:.6f} sec"
            )

            print(
                f"Median          : "
                f"{result['median_latency_sec']:.6f} sec"
            )

            print(
                f"Min             : "
                f"{result['min_latency_sec']:.6f} sec"
            )

            print(
                f"Max             : "
                f"{result['max_latency_sec']:.6f} sec"
            )

            print(
                f"Std             : "
                f"{result['std_latency_sec']:.6f} sec"
            )

        else:

            print(
                "Latency statistics: NOT RECORDED"
            )

    summary = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        OUTPUT_DIR,
        "latency_summary.csv"
    )

    summary.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # Save human-readable TXT
    # --------------------------------------------------------

    txt_path = os.path.join(
        OUTPUT_DIR,
        "latency_summary.txt"
    )

    with open(
        txt_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "FINAL 2632-QUERY LATENCY ANALYSIS\n"
        )

        f.write("=" * 80 + "\n\n")

        for result in results:

            f.write(
                f"Experiment: {result['experiment']}\n"
            )

            f.write(
                f"Total Queries   : "
                f"{result['total_queries']}\n"
            )

            f.write(
                f"Latency Samples : "
                f"{result['latency_samples']}\n"
            )

            f.write(
                f"Status          : "
                f"{result['latency_status']}\n"
            )

            if result["latency_status"] == "Available":

                f.write(
                    f"Mean Latency    : "
                    f"{result['mean_latency_sec']:.6f} sec\n"
                )

                f.write(
                    f"Median Latency  : "
                    f"{result['median_latency_sec']:.6f} sec\n"
                )

                f.write(
                    f"Minimum Latency : "
                    f"{result['min_latency_sec']:.6f} sec\n"
                )

                f.write(
                    f"Maximum Latency : "
                    f"{result['max_latency_sec']:.6f} sec\n"
                )

                f.write(
                    f"Std Deviation   : "
                    f"{result['std_latency_sec']:.6f} sec\n"
                )

            else:

                f.write(
                    "Latency statistics: NOT RECORDED\n"
                )

            f.write(
                "\n" + "-" * 80 + "\n\n"
            )

    print("\n" + "=" * 80)
    print("LATENCY SUMMARY SAVED")
    print("=" * 80)

    print(
        f"CSV -> {csv_path}"
    )

    print(
        f"TXT -> {txt_path}"
    )


if __name__ == "__main__":
    main()