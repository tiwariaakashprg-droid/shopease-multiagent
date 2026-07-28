import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from graph.state import AgentState


_bm25 = None
_documents = None


def _get_bm25():
    global _bm25, _documents

    if _bm25 is not None:
        return _bm25

    loader = TextLoader("data/policies.txt")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    _documents = chunks

    tokenized_docs = [
        doc.page_content.lower().split()
        for doc in chunks
    ]

    _bm25 = BM25Okapi(tokenized_docs)

    print(
        f"[BM25] Built! {len(chunks)} chunks indexed."
    )

    return _bm25


def rag_bm25(state: AgentState) -> AgentState:
    """
    BM25-only retrieval
    """

    intent = state.get("intent", "general")
    message = state.get("user_message", "")
    query = f"{intent} {message}"

    try:
        bm25 = _get_bm25()

        tokenized_query = query.lower().split()

        scores = bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:3]

        bm25_results = [
            _documents[i]
            for i in top_indices
        ]

        all_results = []

        for doc in bm25_results:
            all_results.append(doc.page_content)

        chunks = "\n\n".join(all_results)

        # Policy prediction
        if intent == "refund_request":
            policy_source = "Refund Policy"

        elif intent == "return_item":
            policy_source = "Return Policy"

        elif intent == "order_tracking":
            policy_source = "Shipping Policy"

        elif intent == "cancellation":
            policy_source = "Cancellation Policy"

        elif intent == "complaint":
            policy_source = "Damaged Product Policy"

        else:
            policy_source = "Unknown"

        # BM25 confidence
        if top_indices:
            top_score = max(scores)
            retrieval_confidence = max(
                0.0,
                min(
                    1.0,
                    top_score / 10
                )
            )
        else:
            retrieval_confidence = 0.0

        print(
            f"[BM25] Retrieved "
            f"{len(bm25_results)} chunks. "
            f"Confidence={retrieval_confidence:.2f}"
        )

    except Exception as e:
        chunks = f"Retrieval Error: {str(e)}"
        policy_source = "Unknown"
        retrieval_confidence = 0.0

        print(f"[BM25] Error: {e}")

    return {
        **state,
        "policy_chunks": chunks,
        "policy_source": policy_source,
        "retrieval_confidence": retrieval_confidence,
    }