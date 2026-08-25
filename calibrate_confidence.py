"""
Confidence calibration (NEW) for the Escalation Agent's low-confidence
threshold.

The original threshold (retrieval_confidence < 0.60) was a hardcoded guess.
This script fits an isotonic regression between retrieval_confidence and
actual correctness (from evaluate.py's hybrid results) to find the
confidence value below which predictions are unreliable enough to warrant
human review, and writes it to config/calibration.json where
escalation_agent.py reads it at runtime.

Usage:
    python evaluate.py --mode hybrid          # produces the input data
    python calibrate_confidence.py
"""
import json
import os
import pandas as pd
from sklearn.isotonic import IsotonicRegression

INPUT_PATH = "research_results/evaluation_results_hybrid.csv"
OUTPUT_PATH = "config/calibration.json"


def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Missing {INPUT_PATH} — run `python evaluate.py --mode hybrid` first.")
        return

    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=["confidence", "correct"])

    if len(df) < 10:
        print("Not enough data points to calibrate reliably. Using default threshold.")
        return

    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(df["confidence"], df["correct"])

    # Find the confidence value where predicted P(correct) first crosses 0.75 —
    # i.e. below this point, the retrieval-based prediction is right less than
    # 75% of the time and should be escalated for human review instead.
    grid = [i / 100 for i in range(0, 101)]
    predicted_p_correct = iso.predict(grid)

    threshold = 0.60  # fallback
    for conf, p_correct in zip(grid, predicted_p_correct):
        if p_correct >= 0.75:
            threshold = conf
            break

    os.makedirs("config", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({
            "low_confidence_threshold": round(threshold, 3),
            "calibration_method": "isotonic_regression",
            "n_samples": len(df),
            "target_p_correct": 0.75,
        }, f, indent=2)

    print("=" * 50)
    print("CONFIDENCE CALIBRATION COMPLETE")
    print("=" * 50)
    print(f"Samples used         : {len(df)}")
    print(f"Calibrated threshold : {threshold:.3f}")
    print(f"(predictions with confidence below this are escalated)")
    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
