"""
Cross-encoder reranking step for the Hybrid RAG pipeline (NEW).

FAISS and BM25 each return a ranked list of chunks. Reciprocal Rank Fusion
(RRF) already gives a solid retrieval-only fusion of the two ranked lists
(Cormack et al., 2009) and is what `rag_agent.py` uses as its primary
combination strategy - it requires no extra model and needs no internet
access.

On top of RRF we ADD an optional cross-encoder reranking pass
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) that jointly scores
(query, chunk) pairs - this is strictly more accurate than independently
scored FAISS/BM25 lists because the model attends to the query and the
chunk together. Because model weights need to be downloaded the first time
the model is used, this step is best-effort: if `sentence-transformers`
isn't installed or the model can't be downloaded (e.g. offline env), we
silently fall back to the RRF ranking and keep going - the system must never
crash because of an unavailable reranker.
"""
from typing import List, Tuple

_cross_encoder = None
_reranker_available = None  # tri-state: None=not checked, True/False after first attempt


def _get_cross_encoder():
    global _cross_encoder, _reranker_available

    if _reranker_available is False:
        return None
    if _cross_encoder is not None:
        return _cross_encoder

    try:
        from sentence_transformers import CrossEncoder
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        _reranker_available = True
        print("[Reranker] Cross-encoder loaded.")
    except Exception as e:
        _reranker_available = False
        print(f"[Reranker] Unavailable ({e}). Falling back to RRF-only ranking.")
        return None

    return _cross_encoder


def rerank(query: str, documents: List) -> Tuple[List, str]:
    """
    documents: list of langchain Document objects, already RRF-fused/ordered.
    Returns (reranked_documents, method_name).
    """
    if not documents:
        return documents, "rrf_only"

    model = _get_cross_encoder()
    if model is None:
        return documents, "rrf_only"

    try:
        pairs = [(query, doc.page_content) for doc in documents]
        scores = model.predict(pairs)
        ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked], "rrf_plus_cross_encoder"
    except Exception as e:
        print(f"[Reranker] Scoring failed ({e}). Falling back to RRF-only ranking.")
        return documents, "rrf_only"
