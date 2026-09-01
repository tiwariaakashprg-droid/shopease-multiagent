import argparse

# Weighted-RRF defaults: FAISS is stronger on the current benchmark.
FAISS_WEIGHT = 1.0
BM25_WEIGHT = 1.0
"""
Fair RRF-only ablation.

Uses the EXACT same preprocessing as evaluate.py:
    guardrail_agent -> intent_agent -> query construction

Then performs:
    FAISS top-5 + BM25 top-5
              ↓
          RRF fusion
              ↓
          top-5 documents
              ↓
        majority vote

NO cross-encoder reranker.

Output:
    research_results/new_evaluation/evaluation_results_fair_rrf.csv
"""

import os
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
    reciprocal_rank_fusion,
)


VECTOR_STORE_PATH = "data/faiss_index"


def load_indexes():

    print("\nLoading policy documents...")

    documents = load_policy_chunks()

    # ---------------- BM25 ----------------
    tokenized_docs = [
        doc.page_content.lower().split()
        for doc in documents
    ]

    bm25 = BM25Okapi(tokenized_docs)

    # ---------------- FAISS ----------------
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

    print(f"Loaded {len(documents)} policy chunks.")

    return vectorstore, bm25, documents


def run_fair_rrf(vectorstore, bm25, documents):

    dataset_path = "data/test_dataset.csv"

    df = pd.read_csv(dataset_path)

    total = len(df)
    correct = 0

    results = []

    print("\n" + "=" * 60)
    print("FAIR RRF-ONLY EVALUATION")
    print("=" * 60)

    for index, row in df.iterrows():

        query = row["query"]
        expected = row["expected_policy"]

        # ==================================================
        # EXACT SAME PREPROCESSING AS evaluate.py
        # ==================================================

        state = {
            "user_message": query,
            "customer_id": "C4521",
            "chat_history": [],
            "agent_timings": {},
        }

        state = guardrail_agent(state)

        state = intent_agent(state)

        intent = state.get("intent", "general")

        message = (
            state.get("sanitized_message")
            or state.get("user_message", "")
        )

        # EXACT query construction used by rag_agent.py,
        # rag_faiss.py and rag_bm25.py
        retrieval_query = f"{intent} {message}"

        # ==================================================
        # FAISS TOP-5
        # ==================================================

        faiss_results = vectorstore.similarity_search_with_score(
            retrieval_query,
            k=5
        )

        faiss_ranked = [
            doc for doc, _ in faiss_results
        ]

        # ==================================================
        # BM25 TOP-5
        # ==================================================

        tokenized_query = retrieval_query.lower().split()

        scores = bm25.get_scores(
            tokenized_query
        )

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:5]

        bm25_ranked = [
            documents[i]
            for i in top_indices
        ]

        # ==================================================
        # RRF FUSION
        # ==================================================

        # Weighted Reciprocal Rank Fusion
        fused = weighted_rrf(
            faiss_ranked,
            bm25_ranked,
            k=60,
            faiss_weight=1.0,
            bm25_weight=1.0
        )

        # IMPORTANT:
        # NO CROSS-ENCODER HERE.
        #
        # This is:
        # FAISS + BM25 -> RRF -> top-5
        #
        # NOT:
        # FAISS + BM25 -> RRF -> Cross-Encoder

        top_docs = fused[:5]

        # ==================================================
        # RETRIEVAL-DRIVEN CLASSIFICATION
        # ==================================================

        predicted, cited_sources = classify_from_documents(
            top_docs
        )

        is_correct = int(
            predicted == expected
        )

        correct += is_correct

        results.append({
            "query": query,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "retrieval_method": "rrf_only",
            "intent": intent,
            "sources": "|".join(cited_sources),
        })

        # Progress every 50 queries
        if (index + 1) % 50 == 0:

            current_accuracy = (
                correct / (index + 1)
            )

            print(
                f"Processed {index + 1}/{total} "
                f"| Accuracy: {current_accuracy:.2%}"
            )

    # ======================================================
    # SAVE RESULTS
    # ======================================================

    results_df = pd.DataFrame(results)

    os.makedirs("research_results/new_evaluation", exist_ok=True)

    output_path = (
        "research_results/new_evaluation/"
        "evaluation_results_fair_rrf.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    accuracy = correct / total

    print("\n" + "=" * 60)
    print("FAIR RRF-ONLY RESULTS")
    print("=" * 60)

    print(f"Total Queries : {total}")
    print(f"Correct       : {correct}")
    print(f"Accuracy      : {accuracy:.2%}")

    print(f"\nSaved -> {output_path}")

    print("\n" + "=" * 60)
    print("CURRENT COMPARISON")
    print("=" * 60)

    print("BM25-only              : run separately on the same held-out test set")
    print("FAISS-only             : run separately on the same held-out test set")
    print(f"Fair RRF (FAISS=1.0, BM25=1.0)         : {accuracy:.2%}")
    print("RRF + Cross-Encoder     : run separately on the same held-out test set")



def weighted_rrf(faiss_docs, bm25_docs, k=60, faiss_weight=1.0, bm25_weight=1.0):
    """Fuse LangChain Document rankings using weighted reciprocal-rank fusion.

    The same document may be returned by both retrievers. We use its
    page_content as the stable fusion key and keep the original Document
    object as the value so downstream code still receives Documents.
    """
    scores = {}
    docs_by_key = {}

    for rank, doc in enumerate(faiss_docs, start=1):
        key = doc.page_content
        docs_by_key[key] = doc
        scores[key] = scores.get(key, 0.0) + faiss_weight / (k + rank)

    for rank, doc in enumerate(bm25_docs, start=1):
        key = doc.page_content
        docs_by_key[key] = doc
        scores[key] = scores.get(key, 0.0) + bm25_weight / (k + rank)

    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    return [docs_by_key[key] for key in ranked_keys]

if __name__ == "__main__":

    vectorstore, bm25, documents = load_indexes()

    run_fair_rrf(
        vectorstore,
        bm25,
        documents
    )

