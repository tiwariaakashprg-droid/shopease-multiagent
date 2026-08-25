import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from rank_bm25 import BM25Okapi
from graph.state import AgentState
from graph.nodes.retrieval_utils import load_policy_chunks, classify_from_documents, reciprocal_rank_fusion
from graph.nodes.reranker import rerank
from utils.timing import track_latency

VECTOR_STORE_PATH = "data/faiss_index"

_vectorstore = None
_bm25 = None
_documents = None

llm = OllamaLLM(model="llama3.2", temperature=0.0)


def _get_indexes():
    global _vectorstore, _bm25, _documents

    if _vectorstore is not None and _bm25 is not None:
        return _vectorstore, _bm25

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    _documents = load_policy_chunks()

    tokenized_docs = [doc.page_content.lower().split() for doc in _documents]
    _bm25 = BM25Okapi(tokenized_docs)

    if os.path.exists(VECTOR_STORE_PATH):
        _vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True
        )
        print("[RAG Agent] Loaded existing vector store.")
    else:
        print("[RAG Agent] Building vector store...")
        _vectorstore = FAISS.from_documents(_documents, embeddings)
        _vectorstore.save_local(VECTOR_STORE_PATH)
        print(f"[RAG Agent] Built! {len(_documents)} chunks indexed.")

    return _vectorstore, _bm25


@track_latency("rag_agent")
def rag_agent(state: AgentState) -> AgentState:
    """
    Hybrid RAG pipeline (upgraded):

      1. FAISS dense search (top-5) + BM25 lexical search (top-5)
      2. Reciprocal Rank Fusion merges the two ranked lists into one
         (retrieval-only fusion, no ML dependency, always available)
      3. Optional cross-encoder reranking on top of the fused list
         (best-effort; silently skipped if unavailable)
      4. Policy label = majority vote over the top reranked chunks' TRUE
         source category (retrieval-driven, see retrieval_utils.py)
      5. Only if that vote is ambiguous/Unknown AND retrieval confidence is
         low do we fall back to an LLM disambiguation call - this keeps the
         expensive LLM step rare instead of on the critical path for every
         query.
    """
    print("******** RAG AGENT CALLED ********")

    intent = state.get("intent", "general")
    message = state.get("sanitized_message") or state.get("user_message", "")
    query = f"{intent} {message}"

    try:
        vs, bm25 = _get_indexes()

        # ---------------- FAISS ----------------
        faiss_results = vs.similarity_search_with_score(query, k=5)
        faiss_ranked = [doc for doc, _ in faiss_results]
        faiss_scores = [score for _, score in faiss_results]

        # ---------------- BM25 ----------------
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
        bm25_ranked = [_documents[i] for i in top_indices]

        # ---------------- Fuse (RRF) ----------------
        fused = reciprocal_rank_fusion([faiss_ranked, bm25_ranked])

        # ---------------- Optional cross-encoder rerank ----------------
        reranked, retrieval_method = rerank(query, fused)
        retrieval_method = f"hybrid_{retrieval_method}"

        top_docs = reranked[:5]
        chunks = "\n\n".join(dict.fromkeys(d.page_content for d in top_docs))

        # ---------------- Confidence (from FAISS distance, as before) ----------------
        if faiss_scores:
            avg_score = sum(faiss_scores) / len(faiss_scores)
            retrieval_confidence = max(0.0, min(1.0, 1 / (1 + avg_score)))
        else:
            retrieval_confidence = 0.0

        # ---------------- Retrieval-driven classification ----------------
        policy_source, cited_sources = classify_from_documents(top_docs)

        # ---------------- LLM fallback (rare path) ----------------
        if policy_source == "Unknown" and retrieval_confidence < 0.60:
            prompt = f"""
You are a customer support policy classifier.

Predicted Intent:
{intent}

Choose EXACTLY one label:

Refund Policy
Return Policy
Shipping Policy
Cancellation Policy
Damaged Product Policy
Unknown

Rules:

- Refunds, reimbursement, money back, compensation
  -> Refund Policy

- Returns, exchanges, replacements, send back
  -> Return Policy

- Delivery, shipment, tracking, parcel, package status
  -> Shipping Policy

- Cancel, revoke, rescind, stop order
  -> Cancellation Policy

- Damaged, broken, defective, faulty products
  -> Damaged Product Policy

- If the question is unrelated to these categories,
  return Unknown.

- Never guess a policy.

User Question:
{message}

Retrieved Policies:
{chunks}

Output ONLY one label.
Do not explain.
"""
            response = llm.invoke(prompt).strip().replace(".", "").replace("\n", "")

            valid_policies = ["Refund Policy", "Return Policy", "Shipping Policy",
                               "Cancellation Policy", "Damaged Product Policy", "Unknown"]

            for policy in valid_policies:
                if policy.lower() in response.lower():
                    policy_source = policy
                    break

            retrieval_method += "+llm_fallback"

        print(
            f"[RAG Agent] Retrieved {len(faiss_ranked)} FAISS + {len(bm25_ranked)} BM25 -> "
            f"fused {len(fused)} chunks via {retrieval_method}. "
            f"policy={policy_source} confidence={retrieval_confidence:.2f} "
            f"sources={cited_sources}"
        )

    except Exception as e:
        chunks = f"Policy retrieval error: {str(e)}"
        policy_source = "Unknown"
        cited_sources = []
        retrieval_confidence = 0.0
        retrieval_method = "error"
        print(f"[RAG Agent] Error: {e}")

    return {
        **state,
        "policy_chunks": chunks,
        "policy_source": policy_source,
        "retrieval_confidence": retrieval_confidence,
        "cited_sources": cited_sources,
        "retrieval_method": retrieval_method,
    }
