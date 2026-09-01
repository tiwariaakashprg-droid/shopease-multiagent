"""
Weighted RRF-only evaluation.

Pipeline:
    Guardrail Agent
          ↓
    Intent Agent
          ↓
    FAISS Top-5 + BM25 Top-5
          ↓
    Weighted RRF
          ↓
    Top-5 documents
          ↓
    Retrieval-driven classification

Weights:
    FAISS = 0.8
    BM25  = 0.2

NO Cross-Encoder.

Dataset:
    data/test_dataset.csv

Output:
    research_results/new_evaluation/evaluation_results_rrf.csv

The CSV includes per-query latency so that mean/median/min/max
latency can be calculated reliably after evaluation.
"""

import os
import time
import pandas as pd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi

from graph.nodes.guardrail_agent import guardrail_agent
from graph.nodes.intent_agent import intent_agent

from graph.nodes.retrieval_utils import (
    load_policy_chunks,
    classify_from_documents,
)


# ============================================================
# CONFIGURATION
# ============================================================

VECTOR_STORE_PATH = "data/faiss_index"

DATASET_PATH = "data/test_dataset.csv"

OUTPUT_PATH = (
    "research_results/new_evaluation/"
    "evaluation_results_rrf.csv"
)

FAISS_WEIGHT = 0.8
BM25_WEIGHT = 0.2

RRF_K = 60

TOP_K = 5


# ============================================================
# LOAD INDEXES
# ============================================================

def load_indexes():

    print("\nLoading policy documents...")

    documents = load_policy_chunks()

    print(f"Loaded {len(documents)} policy chunks.")

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    tokenized_docs = [
        doc.page_content.lower().split()
        for doc in documents
    ]

    bm25 = BM25Okapi(tokenized_docs)

    # --------------------------------------------------------
    # FAISS
    # --------------------------------------------------------

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    if os.path.exists(VECTOR_STORE_PATH):

        vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True,
        )

        print("[FAISS] Existing vector store loaded.")

    else:

        print("[FAISS] Building vector store...")

        vectorstore = FAISS.from_documents(
            documents,
            embeddings
        )

        vectorstore.save_local(
            VECTOR_STORE_PATH
        )

        print("[FAISS] Vector store saved.")

    return vectorstore, bm25, documents


# ============================================================
# WEIGHTED RRF
# ============================================================

def weighted_rrf(
    faiss_docs,
    bm25_docs,
    k=60,
    faiss_weight=0.8,
    bm25_weight=0.2
):
    """
    Weighted Reciprocal Rank Fusion.

    Documents are LangChain Document objects, which are not
    directly hashable. Therefore page_content is used as the
    stable fusion key.
    """

    scores = {}

    docs_by_key = {}

    # --------------------------------------------------------
    # FAISS ranking
    # --------------------------------------------------------

    for rank, doc in enumerate(
        faiss_docs,
        start=1
    ):

        key = doc.page_content

        docs_by_key[key] = doc

        score = (
            faiss_weight /
            (k + rank)
        )

        scores[key] = (
            scores.get(key, 0.0)
            + score
        )

    # --------------------------------------------------------
    # BM25 ranking
    # --------------------------------------------------------

    for rank, doc in enumerate(
        bm25_docs,
        start=1
    ):

        key = doc.page_content

        docs_by_key[key] = doc

        score = (
            bm25_weight /
            (k + rank)
        )

        scores[key] = (
            scores.get(key, 0.0)
            + score
        )

    # --------------------------------------------------------
    # Sort by fused score
    # --------------------------------------------------------

    ranked_keys = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    return [
        docs_by_key[key]
        for key in ranked_keys
    ]


# ============================================================
# EVALUATION
# ============================================================

def run_weighted_rrf(
    vectorstore,
    bm25,
    documents
):

    df = pd.read_csv(
        DATASET_PATH
    )

    total = len(df)

    correct = 0

    results = []

    print("\n" + "=" * 60)
    print("WEIGHTED RRF EVALUATION")
    print("=" * 60)

    print(
        f"FAISS Weight : {FAISS_WEIGHT}"
    )

    print(
        f"BM25 Weight  : {BM25_WEIGHT}"
    )

    print(
        f"Total Queries: {total}"
    )

    # ========================================================
    # QUERY LOOP
    # ========================================================

    for index, row in df.iterrows():

        # ----------------------------------------------------
        # Start latency timer
        # ----------------------------------------------------

        start_time = time.perf_counter()

        query = row["query"]

        expected = row["expected_policy"]

        # ----------------------------------------------------
        # SAME PREPROCESSING AS evaluate.py
        # ----------------------------------------------------

        state = {

            "user_message": query,

            "customer_id": "C4521",

            "chat_history": [],

            "agent_timings": {},
        }

        state = guardrail_agent(
            state
        )

        state = intent_agent(
            state
        )

        intent = state.get(
            "intent",
            "general"
        )

        message = (
            state.get(
                "sanitized_message"
            )
            or state.get(
                "user_message",
                ""
            )
        )

        # EXACT retrieval query construction
        retrieval_query = (
            f"{intent} {message}"
        )

        # ----------------------------------------------------
        # FAISS TOP-5
        # ----------------------------------------------------

        faiss_results = (
            vectorstore
            .similarity_search_with_score(
                retrieval_query,
                k=TOP_K
            )
        )

        faiss_ranked = [
            doc
            for doc, _ in faiss_results
        ]

        # ----------------------------------------------------
        # BM25 TOP-5
        # ----------------------------------------------------

        tokenized_query = (
            retrieval_query
            .lower()
            .split()
        )

        bm25_scores = bm25.get_scores(
            tokenized_query
        )

        top_indices = sorted(
            range(
                len(bm25_scores)
            ),
            key=lambda i:
                bm25_scores[i],
            reverse=True
        )[:TOP_K]

        bm25_ranked = [
            documents[i]
            for i in top_indices
        ]

        # ----------------------------------------------------
        # WEIGHTED RRF
        # ----------------------------------------------------

        fused = weighted_rrf(
            faiss_ranked,
            bm25_ranked,
            k=RRF_K,
            faiss_weight=FAISS_WEIGHT,
            bm25_weight=BM25_WEIGHT
        )

        # ----------------------------------------------------
        # TOP-5 FUSED DOCUMENTS
        # ----------------------------------------------------

        top_docs = fused[:TOP_K]

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        predicted, cited_sources = (
            classify_from_documents(
                top_docs
            )
        )

        is_correct = int(
            predicted == expected
        )

        correct += is_correct

        # ----------------------------------------------------
        # END LATENCY TIMER
        # ----------------------------------------------------

        latency = (
            time.perf_counter()
            - start_time
        )

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results.append({

            "query": query,

            "expected": expected,

            "predicted": predicted,

            "correct": is_correct,

            "confidence": None,

            "retrieval_method":
                "weighted_rrf",

            "latency":
                latency,

            "intent":
                intent,

            "sources":
                "|".join(
                    cited_sources
                ),
        })

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if (index + 1) % 50 == 0:

            current_accuracy = (
                correct /
                (index + 1)
            )

            print(
                f"Processed "
                f"{index + 1}/{total} "
                f"| Accuracy: "
                f"{current_accuracy:.2%}"
            )

    # ========================================================
    # SAVE CSV
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    os.makedirs(
        "research_results/new_evaluation",
        exist_ok=True
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ========================================================
    # FINAL METRICS
    # ========================================================

    accuracy = (
        correct / total
    )

    latency_values = (
        results_df["latency"]
    )

    mean_latency = (
        latency_values.mean()
    )

    median_latency = (
        latency_values.median()
    )

    min_latency = (
        latency_values.min()
    )

    max_latency = (
        latency_values.max()
    )

    std_latency = (
        latency_values.std()
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 60)
    print("WEIGHTED RRF RESULTS")
    print("=" * 60)

    print(
        f"Total Queries : {total}"
    )

    print(
        f"Correct       : {correct}"
    )

    print(
        f"Accuracy      : {accuracy:.2%}"
    )

    print(
        f"Mean Latency  : "
        f"{mean_latency:.6f} sec"
    )

    print(
        f"Median Latency: "
        f"{median_latency:.6f} sec"
    )

    print(
        f"Min Latency   : "
        f"{min_latency:.6f} sec"
    )

    print(
        f"Max Latency   : "
        f"{max_latency:.6f} sec"
    )

    print(
        f"Std Latency   : "
        f"{std_latency:.6f} sec"
    )

    print(
        f"\nSaved -> {OUTPUT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    vectorstore, bm25, documents = (
        load_indexes()
    )

    run_weighted_rrf(
        vectorstore,
        bm25,
        documents
    )