from rank_bm25 import BM25Okapi
from graph.state import AgentState
from graph.nodes.retrieval_utils import load_policy_chunks, classify_from_documents
from utils.timing import track_latency

_bm25 = None
_documents = None


def _get_bm25():
    global _bm25, _documents

    if _bm25 is not None:
        return _bm25

    _documents = load_policy_chunks()
    tokenized_docs = [doc.page_content.lower().split() for doc in _documents]
    _bm25 = BM25Okapi(tokenized_docs)

    print(f"[BM25] Built! {len(_documents)} chunks indexed.")
    return _bm25


@track_latency("rag_agent")
def rag_bm25(state: AgentState) -> AgentState:
    """BM25-only retrieval. Classification is now retrieval-driven (majority
    vote over top-k retrieved chunks' true source category), NOT derived
    from the rule-based intent label - see retrieval_utils.py docstring."""

    intent = state.get("intent", "general")
    message = state.get("sanitized_message") or state.get("user_message", "")
    query = f"{intent} {message}"

    try:
        bm25 = _get_bm25()
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        ranked_docs = [_documents[i] for i in top_indices]

        chunks = "\n\n".join(dict.fromkeys(d.page_content for d in ranked_docs))
        policy_source, cited_sources = classify_from_documents(ranked_docs)

        top_score = max(scores) if len(scores) else 0.0
        retrieval_confidence = max(0.0, min(1.0, top_score / 10))

        print(
            f"[BM25] Retrieved {len(ranked_docs)} chunks. "
            f"policy={policy_source} confidence={retrieval_confidence:.2f}"
        )

    except Exception as e:
        chunks = f"Retrieval Error: {str(e)}"
        policy_source = "Unknown"
        cited_sources = []
        retrieval_confidence = 0.0
        print(f"[BM25] Error: {e}")

    return {
        **state,
        "policy_chunks": chunks,
        "policy_source": policy_source,
        "retrieval_confidence": retrieval_confidence,
        "cited_sources": cited_sources,
        "retrieval_method": "bm25",
    }
