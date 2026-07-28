import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from rank_bm25 import BM25Okapi
from graph.state import AgentState

VECTOR_STORE_PATH = "data/faiss_index"

_vectorstore = None
_bm25 = None
_documents = None

llm = OllamaLLM(
    model="llama3.2",
    temperature=0.0
)


def _get_vectorstore():
    global _vectorstore, _bm25, _documents

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

        print("[RAG Agent] Loaded existing vector store.")

        
        docs = []

        for file in os.listdir("data/policies"):
            if file.endswith(".txt"):
                loader = TextLoader(
                    os.path.join(
                        "data/policies",
                        file
                    )
                )
                loaded_docs = loader.load()
                docs.extend(loaded_docs)

        splitter = RecursiveCharacterTextSplitter(chunk_size=150,chunk_overlap=30)

        chunks = splitter.split_documents(docs)

        _documents = chunks

        tokenized_docs = [
            doc.page_content.lower().split()
            for doc in chunks
        ]

        _bm25 = BM25Okapi(tokenized_docs)

    else:
        print("[RAG Agent] Building vector store...")

        docs = []

        for file in os.listdir("data/policies"):
            if file.endswith(".txt"):
                loader = TextLoader(
                    os.path.join(
                        "data/policies",
                        file
                    )
                )
                loaded_docs = loader.load()
                docs.extend(loaded_docs)

        splitter = RecursiveCharacterTextSplitter(chunk_size=150,chunk_overlap=30)

        chunks = splitter.split_documents(docs)
        
        _documents = chunks

        tokenized_docs = [
            doc.page_content.lower().split()
            for doc in chunks
        ]

        _bm25 = BM25Okapi(tokenized_docs)

        _vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        _vectorstore.save_local(
            VECTOR_STORE_PATH
        )

        print(
            f"[RAG Agent] Built! "
            f"{len(chunks)} chunks indexed."
        )

    return _vectorstore


def rag_agent(state: AgentState) -> AgentState:
    print("******** RAG AGENT CALLED ********")
    """
    Hybrid RAG:
    FAISS + BM25 retrieval
    LLM-based policy classification
    """

    intent = state.get("intent", "general")
    message = state.get(
        "user_message",
        ""
    )

    query = f"{intent} {message}"

    try:
        vs = _get_vectorstore()

        # ---------------- FAISS ----------------
        faiss_results = (vs.similarity_search_with_score(query,k=3))

        faiss_scores = []

        for doc, score in faiss_results:
            faiss_scores.append(score)

        # ---------------- BM25 ----------------
        bm25_results = []

        if (_bm25 is not None and _documents is not None):
            tokenized_query = (query.lower().split())

            scores = _bm25.get_scores(tokenized_query)

            top_indices = sorted(range(len(scores)),key=lambda i: scores[i], reverse=True)[:3]

            bm25_results = [_documents[i] for i in top_indices]

        # ---------------- Merge ----------------
        all_results = []

        for doc, score in faiss_results:
            all_results.append(doc.page_content)

        for doc in bm25_results:
            all_results.append(doc.page_content)

        all_results = list(dict.fromkeys(all_results))

        chunks = "\n\n".join(all_results[:5])

        # ---------------- Confidence ----------------
        if faiss_scores:
            avg_score = (sum(faiss_scores) / len(faiss_scores))

            retrieval_confidence = max(0.0,min(1.0,1 / (1 + avg_score)))

        else:
            retrieval_confidence = 0.0

        # ---------------- Policy Prediction ----------------
        if retrieval_confidence > 0.60:

            if intent == "refund_request":
                policy_source = "Refund Policy"

            elif intent == "return_item":
                policy_source = "Return Policy"

            elif intent == "cancellation":
                policy_source = "Cancellation Policy"

            elif intent == "complaint":
                policy_source = "Damaged Product Policy"

            elif intent == "order_tracking":
                policy_source = "Shipping Policy"

            else:
                policy_source = "Unknown"

        else:

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

            response = (llm.invoke(prompt).strip().replace(".", "").replace("\n", ""))

            valid_policies = ["Refund Policy","Return Policy","Shipping Policy",
                              "Cancellation Policy","Damaged Product Policy","Unknown"]

            policy_source = "Unknown"

            for policy in valid_policies:
                if policy.lower() in response.lower():
                    policy_source = policy
                    break

        print(
            f"[RAG Agent] Retrieved "
            f"{len(faiss_results)} FAISS + "
            f"{len(bm25_results)} BM25 chunks. "
            f"Confidence="
            f"{retrieval_confidence:.2f}"
        )

    except Exception as e:

        chunks = (
            f"Policy retrieval error: "
            f"{str(e)}"
        )

        policy_source = "Unknown"
        retrieval_confidence = 0.0

        print(
            f"[RAG Agent] Error: {e}"
        )

    return {
        **state,
        "policy_chunks": chunks,
        "policy_source": policy_source,
        "retrieval_confidence":
            retrieval_confidence,
    }