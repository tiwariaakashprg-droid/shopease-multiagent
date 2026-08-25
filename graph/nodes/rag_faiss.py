import os
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from graph.state import AgentState
from graph.nodes.retrieval_utils import load_policy_chunks, classify_from_documents
from utils.timing import track_latency

VECTOR_STORE_PATH = "data/faiss_index"
_vectorstore = None


def _get_vectorstore():
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if os.path.exists(VECTOR_STORE_PATH):
        _vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
        print("[FAISS] Loaded existing vector store.")
    else:
        print("[FAISS] Building vector store...")
        chunks = load_policy_chunks()
        _vectorstore = FAISS.from_documents(chunks, embeddings)
        _vectorstore.save_local(VECTOR_STORE_PATH)
        print(f"[FAISS] Built! {len(chunks)} chunks indexed.")

    return _vectorstore


@track_latency("rag_agent")
def rag_faiss(state: AgentState) -> AgentState:
    """FAISS-only (dense) retrieval. Classification is retrieval-driven,
    same as rag_bm25.py, for a fair apples-to-apples comparison."""

    intent = state.get("intent", "general")
    message = state.get("sanitized_message") or state.get("user_message", "")
    query = f"{intent} {message}"

    try:
        vs = _get_vectorstore()
        results = vs.similarity_search_with_score(query, k=5)

        ranked_docs = [doc for doc, _ in results]
        faiss_scores = [score for _, score in results]

        chunks = "\n\n".join(dict.fromkeys(d.page_content for d in ranked_docs))
        policy_source, cited_sources = classify_from_documents(ranked_docs)

        if faiss_scores:
            avg_score = sum(faiss_scores) / len(faiss_scores)
            retrieval_confidence = max(0.0, min(1.0, 1 / (1 + avg_score)))
        else:
            retrieval_confidence = 0.0

        print(
            f"[FAISS] Retrieved {len(ranked_docs)} chunks. "
            f"policy={policy_source} confidence={retrieval_confidence:.2f}"
        )

    except Exception as e:
        chunks = f"Retrieval Error: {str(e)}"
        policy_source = "Unknown"
        cited_sources = []
        retrieval_confidence = 0.0
        print(f"[FAISS] Error: {e}")

    return {
        **state,
        "policy_chunks": chunks,
        "policy_source": policy_source,
        "retrieval_confidence": retrieval_confidence,
        "cited_sources": cited_sources,
        "retrieval_method": "faiss",
    }
