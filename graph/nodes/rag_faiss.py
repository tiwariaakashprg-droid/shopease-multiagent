import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from graph.state import AgentState

VECTOR_STORE_PATH = "data/faiss_index"
_vectorstore = None


def _get_vectorstore():
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    if os.path.exists(VECTOR_STORE_PATH):
        _vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print("[FAISS] Loaded existing vector store.")

    else:
        print("[FAISS] Building vector store...")

        loader = TextLoader("data/policies.txt")
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(docs)

        _vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        _vectorstore.save_local(VECTOR_STORE_PATH)

        print(
            f"[FAISS] Built! {len(chunks)} chunks indexed."
        )

    return _vectorstore


def rag_faiss(state: AgentState) -> AgentState:

    intent = state.get("intent", "general")
    message = state.get("user_message", "")
    query = f"{intent} {message}"

    try:
        vs = _get_vectorstore()

        # FAISS ONLY
        faiss_results = vs.similarity_search_with_score(
            query,
            k=3
        )

        faiss_scores = []

        all_results = []

        for doc, score in faiss_results:
            all_results.append(doc.page_content)
            faiss_scores.append(score)

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

        # Confidence
        if faiss_scores:
            avg_score = sum(faiss_scores) / len(
                faiss_scores
            )

            retrieval_confidence = max(
                0.0,
                min(
                    1.0,
                    1 / (1 + avg_score)
                )
            )

        else:
            retrieval_confidence = 0.0

        print(
            f"[FAISS] Retrieved "
            f"{len(faiss_results)} chunks. "
            f"Confidence={retrieval_confidence:.2f}"
        )

    except Exception as e:
        chunks = f"Retrieval Error: {str(e)}"
        policy_source = "Unknown"
        retrieval_confidence = 0.0

        print(f"[FAISS] Error: {e}")

    return {
        **state,
        "policy_chunks": chunks,
        "policy_source": policy_source,
        "retrieval_confidence": retrieval_confidence,
    }