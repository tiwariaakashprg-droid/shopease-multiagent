"""
Shared retrieval utilities used by rag_bm25.py, rag_faiss.py and rag_agent.py.

WHY THIS FILE EXISTS (important methodological fix):
------------------------------------------------------
In the original implementation, `policy_source` for ALL THREE retrieval
modes (BM25-only, FAISS-only, Hybrid) was predicted almost entirely from a
hardcoded `intent -> policy` lookup table, with the actually-retrieved
chunks barely influencing the label. That means the "83.19% vs 76.64%"
comparison in the paper was really comparing an LLM disambiguation step
(triggered only in the hybrid path when confidence <= 0.60), not comparing
retrieval quality.

This module makes classification GENUINELY retrieval-driven: each policy
document's filename is mapped to its ground-truth category, and the
predicted label is a majority vote over the categories of the top-ranked
retrieved chunks. This makes BM25-only / FAISS-only / Hybrid a fair,
honest, reproducible comparison of retrieval methods - the way the paper
claims to be evaluating them.
"""
import os
from collections import Counter
from typing import List, Tuple

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

POLICIES_DIR = "data/policies"

# filename -> ground-truth policy category (must match test_queries_clean.csv labels)
POLICY_FILE_MAP = {
    "refund_policy.txt": "Refund Policy",
    "return_policy.txt": "Return Policy",
    "shipping_policy.txt": "Shipping Policy",
    "tracking_policy.txt": "Shipping Policy",
    "international_shipping_policy.txt": "Shipping Policy",
    "cancellation_policy.txt": "Cancellation Policy",
    "damaged_policy.txt": "Damaged Product Policy",
    # not part of the 6 evaluation categories -> treated as Unknown
    "coupon_policy.txt": "Unknown",
    "emi_policy.txt": "Unknown",
    "membership_policy.txt": "Unknown",
}


def load_policy_chunks(chunk_size: int = 150, chunk_overlap: int = 30):
    """Loads + splits every policy file, tagging each chunk's metadata with
    its source filename AND its mapped ground-truth category, so downstream
    retrieval-driven classification is possible."""
    docs = []
    for file in sorted(os.listdir(POLICIES_DIR)):
        if not file.endswith(".txt"):
            continue
        loader = TextLoader(os.path.join(POLICIES_DIR, file))
        loaded = loader.load()
        for d in loaded:
            d.metadata["policy_file"] = file
            d.metadata["policy_category"] = POLICY_FILE_MAP.get(file, "Unknown")
        docs.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)  # metadata propagates to each chunk
    return chunks


def classify_from_documents(documents: List, top_k: int = 5) -> Tuple[str, List[str]]:
    """Majority-vote policy category over the top_k retrieved chunks.
    Returns (predicted_category, cited_source_filenames)."""
    if not documents:
        return "Unknown", []

    top = documents[:top_k]
    categories = [d.metadata.get("policy_category", "Unknown") for d in top]
    sources = list(dict.fromkeys(d.metadata.get("policy_file", "unknown") for d in top))

    counts = Counter(categories)
    predicted, _ = counts.most_common(1)[0]
    return predicted, sources


def reciprocal_rank_fusion(ranked_lists: List[List], k: int = 60) -> List:
    """
    Standard Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, 2009).
    ranked_lists: list of ranked document lists (best first), e.g.
                  [faiss_ranked_docs, bm25_ranked_docs]
    Returns a single fused ranking (best first), deduplicated by content.
    """
    scores = {}
    doc_lookup = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_lookup[key] = doc

    fused_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_lookup[key] for key in fused_keys]
