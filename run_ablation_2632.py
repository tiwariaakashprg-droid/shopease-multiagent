"""
2632-Query Component Ablation Study
===================================

Configurations:
    1. full_pipeline
    2. no_crm
    3. no_memory
    4. no_escalation

Metrics:
    groundedness
    personalization
    relevance
    overall

The experiment supports:
    --n 10
    --n 100
    --n 2632

Checkpoint/resume:
    Results are saved after EVERY completed run.

The original 2632-query ablation files are NOT modified.
"""

import argparse
import os
import pandas as pd

from graph.nodes.guardrail_agent import guardrail_agent
from graph.nodes.intent_agent import intent_agent
from graph.nodes.crm_agent import crm_agent
from graph.nodes.memory_agent import memory_agent
from graph.nodes.rag_agent import rag_agent
from graph.nodes.escalation_agent import escalation_agent
from graph.supervisor import supervisor_agent

from llm_judge_eval import judge_response


# ============================================================
# PATHS
# ============================================================

INPUT_PATH = (
    "research_results"
    "\\final_2632_evaluation"
    "\\ablation_study"
    "\\ablation_queries_2632.csv"
)

OUTPUT_DIR = (
    "research_results"
    "\\final_2632_evaluation"
    "\\ablation_study"
)

RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "ablation_study_results_2632.csv"
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "ablation_study_summary_2632.csv"
)

SUMMARY_TXT_PATH = os.path.join(
    OUTPUT_DIR,
    "ablation_study_summary_2632.txt"
)


# ============================================================
# ABLATION CONFIGURATIONS
# ============================================================

CONFIGS = {
    "full_pipeline": {},

    "no_crm": {
        "disable_crm": True
    },

    "no_memory": {
        "disable_memory": True
    },

    "no_escalation": {
        "disable_escalation": True
    },
}


# ============================================================
# RUN PIPELINE
# ============================================================

def run_pipeline(
    query,
    customer_id,
    chat_history,
    disable_crm=False,
    disable_memory=False,
    disable_escalation=False
):

    state = {
        "user_message": query,
        "customer_id": customer_id,
        "chat_history": chat_history,
        "agent_timings": {},
    }

    # --------------------------------------------------------
    # Guardrail
    # --------------------------------------------------------

    state = guardrail_agent(state)

    # --------------------------------------------------------
    # Intent
    # --------------------------------------------------------

    state = intent_agent(state)

    # --------------------------------------------------------
    # CRM
    # --------------------------------------------------------

    if disable_crm:

        state = {
            **state,
            "customer_data": {},
            "customer_context": "Not available."
        }

    else:

        state = crm_agent(state)

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    if disable_memory:

        state = {
            **state,
            "memory_context": "No previous conversation."
        }

    else:

        state = memory_agent(state)

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    state = rag_agent(state)

    # --------------------------------------------------------
    # Escalation
    # --------------------------------------------------------

    if disable_escalation:

        state = {
            **state,
            "should_escalate": False,
            "escalation_reason": "",
            "draft_reply": ""
        }

    else:

        state = escalation_agent(state)

    # --------------------------------------------------------
    # Supervisor
    # --------------------------------------------------------

    state = supervisor_agent(state)

    return state


# ============================================================
# LOAD EXISTING CHECKPOINT
# ============================================================

def load_checkpoint():

    if not os.path.exists(RESULTS_PATH):

        return pd.DataFrame()

    try:

        checkpoint = pd.read_csv(
            RESULTS_PATH
        )

    except Exception as error:

        print("\nWARNING: Could not read checkpoint.")
        print(error)

        return pd.DataFrame()

    if checkpoint.empty:

        return pd.DataFrame()

    print(
        f"\nExisting checkpoint found:"
        f" {len(checkpoint)} completed runs"
    )

    return checkpoint


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(rows):

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        RESULTS_PATH,
        index=False
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(results):

    if results.empty:

        return

    summary = (
        results
        .groupby("config")[
            [
                "groundedness",
                "personalization",
                "relevance",
                "overall"
            ]
        ]
        .mean()
        .round(2)
    )

    summary.to_csv(
        SUMMARY_PATH
    )

    with open(
        SUMMARY_TXT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "FINAL 2632-QUERY COMPONENT ABLATION STUDY\n"
        )

        file.write(
            "=" * 80 + "\n\n"
        )

        file.write(
            "LLM-Judge Mean Scores (1-5 scale)\n\n"
        )

        file.write(
            summary.to_string()
        )

        file.write(
            "\n\n"
        )

        file.write(
            f"Completed runs: {len(results)}\n"
        )

    return summary


# ============================================================
# MAIN
# ============================================================

def main(n):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 80)
    print("FINAL 2632-QUERY COMPONENT ABLATION STUDY")
    print("=" * 80)

    # --------------------------------------------------------
    # Validate n
    # --------------------------------------------------------

    if n < 1:

        raise ValueError(
            "--n must be at least 1."
        )

    if n > 2632:

        raise ValueError(
            "--n cannot exceed 2632."
        )

    # --------------------------------------------------------
    # Load 2632-query input
    # --------------------------------------------------------

    if not os.path.exists(INPUT_PATH):

        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_PATH}"
        )

    full_df = pd.read_csv(
        INPUT_PATH
    )

    if len(full_df) != 2632:

        raise ValueError(
            f"Expected 2632 input queries, "
            f"found {len(full_df)}."
        )

    # --------------------------------------------------------
    # Select requested number of queries
    # --------------------------------------------------------

    df = full_df.head(n).copy()

    df = df.reset_index(
        drop=True
    )

    print(
        f"\nFull dataset      : 2632 queries"
    )

    print(
        f"Selected queries  : {len(df)}"
    )

    print(
        f"Configurations    : {len(CONFIGS)}"
    )

    total_runs = (
        len(df) * len(CONFIGS)
    )

    print(
        f"Expected runs     : {total_runs}"
    )

    print(
        "\nConfiguration list:"
    )

    for config_name in CONFIGS:

        print(
            f"  - {config_name}"
        )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = load_checkpoint()

    if checkpoint.empty:

        rows = []

    else:

        rows = checkpoint.to_dict(
            orient="records"
        )

    # --------------------------------------------------------
    # Determine completed runs
    # --------------------------------------------------------

    completed = set()

    if not checkpoint.empty:

        required_columns = {
            "query_id",
            "config"
        }

        if required_columns.issubset(
            checkpoint.columns
        ):

            for _, checkpoint_row in checkpoint.iterrows():

                query_id = int(
                    checkpoint_row["query_id"]
                )

                config_name = (
                    checkpoint_row["config"]
                )

                completed.add(
                    (query_id, config_name)
                )

    print(
        f"\nCheckpoint runs    : {len(completed)}"
    )

    # --------------------------------------------------------
    # Run experiment
    # --------------------------------------------------------

    for query_id, row in df.iterrows():

        query = row[
            "query"
        ]

        expected = row[
            "expected_policy"
        ]

        for config_name, kwargs in CONFIGS.items():

            key = (
                query_id,
                config_name
            )

            # ------------------------------------------------
            # Resume
            # ------------------------------------------------

            if key in completed:

                print(
                    f"[SKIP] Query "
                    f"{query_id + 1}/{len(df)} | "
                    f"{config_name}"
                )

                continue

            print(
                "\n" + "-" * 80
            )

            print(
                f"Query         : "
                f"{query_id + 1}/{len(df)}"
            )

            print(
                f"Configuration : "
                f"{config_name}"
            )

            print(
                f"Expected      : "
                f"{expected}"
            )

            print(
                f"Query         : "
                f"{query[:120]}"
            )

            # ------------------------------------------------
            # Pipeline
            # ------------------------------------------------

            state = run_pipeline(
                query=query,
                customer_id="C1190",
                chat_history=[],
                **kwargs
            )

            # ------------------------------------------------
            # Response
            # ------------------------------------------------

            response = (
                state.get("final_response")
                or state.get("draft_reply")
                or ""
            )

            # ------------------------------------------------
            # LLM Judge
            # ------------------------------------------------

            judge = judge_response(
                query,
                response,
                expected_policy=expected,
                customer_context=state.get(
                    "customer_context",
                    ""
                )
            )

            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            result_row = {

                "query_id":
                    query_id,

                "query":
                    query,

                "config":
                    config_name,

                "expected_policy":
                    expected,

                "predicted_policy":
                    state.get(
                        "policy_source"
                    ),

                "escalated":
                    state.get(
                        "should_escalate"
                    ),

                "groundedness":
                    judge.get(
                        "groundedness"
                    ),

                "personalization":
                    judge.get(
                        "personalization"
                    ),

                "relevance":
                    judge.get(
                        "relevance"
                    ),

                "overall":
                    judge.get(
                        "overall"
                    ),
            }

            rows.append(
                result_row
            )

            completed.add(
                key
            )

            # ------------------------------------------------
            # CHECKPOINT
            # ------------------------------------------------

            save_results(
                rows
            )

            current_count = len(
                completed
            )

            print(
                "\nCHECKPOINT SAVED"
            )

            print(
                f"Completed runs : "
                f"{current_count}/{total_runs}"
            )

            print(
                f"Groundedness   : "
                f"{judge.get('groundedness')}"
            )

            print(
                f"Personalization: "
                f"{judge.get('personalization')}"
            )

            print(
                f"Relevance      : "
                f"{judge.get('relevance')}"
            )

            print(
                f"Overall        : "
                f"{judge.get('overall')}"
            )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    results = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Only summarize requested query range
    # --------------------------------------------------------

    selected_query_ids = set(
        df.index.tolist()
    )

    if not results.empty:

        results_for_summary = results[
            results["query_id"].isin(
                selected_query_ids
            )
        ].copy()

    else:

        results_for_summary = results

    # --------------------------------------------------------
    # Save final summary
    # --------------------------------------------------------

    summary = save_summary(
        results_for_summary
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "ABLATION RUN FINISHED"
    )

    print(
        "=" * 80
    )

    print(
        f"Selected queries : "
        f"{len(df)}"
    )

    print(
        f"Expected runs    : "
        f"{total_runs}"
    )

    print(
        f"Completed runs   : "
        f"{len(completed)}"
    )

    if summary is not None:

        print(
            "\nLLM-Judge Summary:"
        )

        print(
            summary
        )

    print(
        f"\nResults -> "
        f"{RESULTS_PATH}"
    )

    print(
        f"Summary -> "
        f"{SUMMARY_PATH}"
    )

    print(
        f"TXT -> "
        f"{SUMMARY_TXT_PATH}"
    )


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Run the 2632-query component ablation study."
        )
    )

    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help=(
            "Number of queries to evaluate. "
            "Use 10 for pilot, 100 for medium experiment, "
            "or 2632 for the full dataset."
        )
    )

    args = parser.parse_args()

    main(
        args.n
    )
