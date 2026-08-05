# 🛍️ ShopEase — Multi-Agent AI Customer Support System

> A Generative AI-based Multi-Agent Customer Support System using **LangGraph, Hybrid Retrieval-Augmented Generation (FAISS + BM25), LLaMA 3.2, CRM integration, conversational memory, and Human-in-the-Loop escalation.**

---

## 📌 Overview

**ShopEase** is an intelligent enterprise customer-support framework designed to automate e-commerce customer queries using a coordinated multi-agent architecture.

Instead of relying on a single LLM to perform every task, ShopEase decomposes customer-support reasoning into specialized agents responsible for intent understanding, CRM retrieval, conversational memory, enterprise knowledge retrieval, escalation decisions, response generation, and response validation.

The system combines **dense semantic retrieval using FAISS** with **sparse lexical retrieval using BM25** to improve policy retrieval accuracy.

ShopEase also incorporates **Human-in-the-Loop (HITL)** escalation for sensitive or high-risk customer queries.

---

## ✨ Key Features

* 🤖 Multi-Agent architecture orchestrated using **LangGraph**
* 🧠 Local LLM inference using **LLaMA 3.2 via Ollama**
* 🔍 **Hybrid RAG — FAISS + BM25**
* 📚 Semantic embeddings using **nomic-embed-text**
* 👤 CRM-aware personalized customer responses
* 💭 Conversational Memory Agent
* 🚨 Human-in-the-Loop escalation
* 🛡️ Supervisor-based response generation
* 🔎 Reflection Agent for response validation
* 📊 Complete evaluation pipeline
* 📈 Accuracy, latency, confusion matrix, and classification analysis
* 💻 Interactive Streamlit customer-support interface
* 🔐 Fully local AI inference with no mandatory external LLM API

---

# 🏗️ System Architecture

```text
                     Customer Query
                           │
                           ▼
                  ┌─────────────────┐
                  │  Intent Agent   │
                  │                 │
                  │ Intent          │
                  │ Sentiment       │
                  │ Urgency         │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │    CRM Agent    │
                  │                 │
                  │ Customer Data   │
                  │ Order Details   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │  Memory Agent   │
                  │                 │
                  │ Conversation    │
                  │ Context         │
                  └────────┬────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │  Hybrid RAG Agent   │
                │                     │
                │   FAISS  +  BM25    │
                │      │       │      │
                │      └───┬───┘      │
                │          ▼          │
                │   Policy Context    │
                └──────────┬──────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │ Escalation Agent │
                 │                  │
                 │ HITL Decision    │
                 └────────┬─────────┘
                          │
                          ▼
                    ┌────────────┐
                    │ Supervisor │
                    │   Agent    │
                    └─────┬──────┘
                          │
                ┌─────────┴─────────┐
                │                   │
          Normal Response       HITL Required
                │                   │
                ▼                   ▼
       ┌────────────────┐     ┌─────────────┐
       │ Reflection     │     │ Human Agent │
       │ Agent          │     │ Escalation  │
       └───────┬────────┘     └─────────────┘
               │
               ▼
          Final Response
```

---

# 🤖 Multi-Agent Architecture

ShopEase uses specialized agents instead of relying on a single monolithic LLM.

| Agent                | Responsibility                                                |
| -------------------- | ------------------------------------------------------------- |
| **Intent Agent**     | Identifies customer intent, sentiment, and urgency            |
| **CRM Agent**        | Retrieves customer and order information                      |
| **Memory Agent**     | Maintains relevant conversational context                     |
| **Hybrid RAG Agent** | Retrieves enterprise policies using FAISS + BM25              |
| **Escalation Agent** | Determines whether Human-in-the-Loop intervention is required |
| **Supervisor Agent** | Generates the final customer-support response                 |
| **Reflection Agent** | Provides a validation stage for generated responses           |

The agents are orchestrated through a **LangGraph StateGraph** and share information using a common `AgentState`.

---

# 🔍 Hybrid Retrieval-Augmented Generation

One of the central components of ShopEase is its Hybrid RAG pipeline.

Traditional RAG systems often rely exclusively on semantic vector search. ShopEase combines two complementary retrieval strategies:

### 1. Dense Semantic Retrieval — FAISS

FAISS retrieves policy chunks based on semantic similarity.

Embeddings are generated using:

```text
nomic-embed-text
```

This allows the system to retrieve relevant policies even when the customer's wording differs from the stored policy text.

### 2. Sparse Lexical Retrieval — BM25

BM25 performs keyword-based retrieval.

It is particularly useful when customer queries contain exact terms such as:

```text
refund
return
damaged product
shipping
cancellation
```

### 3. Hybrid Retrieval

```text
Customer Query
      │
      ├───────────────┐
      ▼               ▼
    FAISS            BM25
 Semantic           Lexical
 Retrieval          Retrieval
      │               │
      └───────┬───────┘
              ▼
       Result Fusion
              │
              ▼
     Relevant Policies
              │
              ▼
          LLaMA 3.2
```

Combining semantic and lexical retrieval improves robustness compared with using either retrieval method independently.

---

# 🚨 Human-in-the-Loop Escalation

Not every customer-support query should be handled autonomously.

ShopEase includes an **Escalation Agent** that detects cases requiring human intervention.

Examples include:

* Legal or regulatory threats
* High-value refund requests
* Repeated unresolved complaints
* Sensitive customer situations
* High-risk or uncertain requests
* Cases requiring human judgment

When escalation is triggered, the workflow can route the request away from normal autonomous response generation.

---

# 🧠 Technology Stack

| Layer                     | Technology                  |
| ------------------------- | --------------------------- |
| Programming Language      | Python                      |
| Multi-Agent Orchestration | LangGraph                   |
| LLM Integration           | LangChain                   |
| Large Language Model      | LLaMA 3.2                   |
| Local LLM Runtime         | Ollama                      |
| Dense Retrieval           | FAISS                       |
| Sparse Retrieval          | BM25                        |
| Embeddings                | nomic-embed-text            |
| User Interface            | Streamlit                   |
| Data Processing           | Pandas                      |
| CRM Simulation            | CSV                         |
| Knowledge Base            | Enterprise policy documents |

---

# 📁 Project Structure

```text
shopease-multiagent/
│
├── app.py
│
├── requirements.txt
│
├── evaluate.py
├── compare_agents.py
├── accuracy_graph.py
├── latency_graph.py
├── confusion_matrix.py
├── classification_report.py
├── final_results.py
│
├── graph/
│   │
│   ├── __init__.py
│   ├── state.py
│   ├── workflow.py
│   ├── supervisor.py
│   ├── reflection_agent.py
│   │
│   └── nodes/
│       ├── __init__.py
│       ├── intent_agent.py
│       ├── crm_agent.py
│       ├── memory_agent.py
│       ├── rag_agent.py
│       ├── rag_faiss.py
│       ├── rag_bm25.py
│       ├── escalation_agent.py
│       └── single_agent.py
│
├── data/
│   ├── customers.csv
│   ├── policies.txt
│   ├── test_queries.csv
│   ├── test_queries_clean.csv
│   └── test_queries_old.csv
│
└── tests/
    └── test_single_agent.py
```

---

# 🔄 Agent Workflow

The LangGraph workflow follows the sequence:

```text
Intent Agent
     ↓
CRM Agent
     ↓
Memory Agent
     ↓
Hybrid RAG Agent
     ↓
Escalation Agent
     ↓
Supervisor Agent
     ↓
Reflection Agent
     ↓
Final Response
```

The Supervisor also supports conditional routing when escalation is required.

---

# 📊 Experimental Evaluation

The system was evaluated on a manually curated dataset containing:

**595 customer-support queries**

across multiple policy categories.

### Policy Categories

* Refund Policy
* Return Policy
* Shipping Policy
* Cancellation Policy
* Damaged Product Policy
* Unknown / Out-of-domain queries

---

## 📈 Results

| Metric                   |           Result |
| ------------------------ | ---------------: |
| **Hybrid RAG Accuracy**  |       **83.19%** |
| Evaluated Queries        |          **595** |
| Correct Predictions      |          **495** |
| Average Response Latency | **0.54 seconds** |

The experimental results demonstrate the benefit of combining semantic and lexical retrieval for enterprise customer-support queries.

The evaluation pipeline also generates:

* Confusion Matrix
* Classification Report
* Accuracy Comparison
* Latency Analysis
* Incorrect Prediction Analysis

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/tiwariaakashprg-droid/shopease-multiagent.git
cd shopease-multiagent
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🦙 Install Ollama

ShopEase uses local LLM inference through Ollama.

After installing Ollama, download LLaMA 3.2:

```bash
ollama pull llama3.2
```

Download the embedding model:

```bash
ollama pull nomic-embed-text
```

Verify the installed models:

```bash
ollama list
```

You should see:

```text
llama3.2
nomic-embed-text
```

---

# ▶️ Run ShopEase

Make sure Ollama is running.

Then execute:

```bash
streamlit run app.py
```

Streamlit will start the application locally.

Open:

```text
http://localhost:8501
```

in your browser.

---

# 🖥️ Application Interface

The Streamlit application provides:

* Customer selection
* Customer profile and order information
* Interactive support chat
* Real-time agent decisions
* Intent detection
* Sentiment analysis
* Urgency detection
* RAG retrieval information
* HITL escalation status
* Session statistics

This provides visibility into how the multi-agent workflow processes each customer query.

---

# 🧪 Evaluation

To run the evaluation pipeline:

```bash
python evaluate.py
```

Additional analysis scripts include:

```bash
python confusion_matrix.py
python classification_report.py
python accuracy_graph.py
python latency_graph.py
python compare_agents.py
```

These scripts are used to analyze retrieval performance and generate experimental results.

---

# 🎯 Example Queries

### Order Tracking

```text
Where is my order?
```

### Refund

```text
I want a refund for my order.
```

### Damaged Product

```text
My product arrived damaged. What should I do?
```

### Multiple Policies

```text
My laptop is damaged. Can I return it and get a refund?
```

### Escalation

```text
I have complained several times and I want to take legal action.
```

The last example can trigger the Human-in-the-Loop escalation mechanism.

---

# 🔬 Research Motivation

Modern enterprise customer-support systems must handle:

* Large volumes of customer queries
* Multiple enterprise policies
* Ambiguous user language
* Personalized customer information
* Safety-sensitive requests
* Hallucination risks
* Escalation requirements

ShopEase explores how **Multi-Agent Systems + Hybrid RAG + Human-in-the-Loop reasoning** can be combined into a modular enterprise customer-support architecture.

---

# 🚀 Future Improvements

Future extensions include:

* FastAPI backend
* React / Next.js frontend
* PostgreSQL-based CRM
* Redis-based conversational memory
* Qdrant/Pinecone vector database
* Docker containerization
* Cloud deployment
* Authentication and authorization
* LangSmith/OpenTelemetry observability
* Parallel agent execution
* Improved reflection and self-correction
* Advanced reranking
* Voice-based customer support
* Multilingual support
* Real enterprise CRM integration
* Production monitoring and analytics

---

# 📄 Research

This repository contains the implementation and experimental evaluation of the ShopEase research project:

> **ShopEase: A Generative AI-Based Multi-Agent Framework for Enterprise Customer Support Using Hybrid Retrieval-Augmented Generation**

The project investigates the combination of:

**Generative AI + Multi-Agent Systems + Hybrid RAG + CRM + Conversational Memory + Human-in-the-Loop AI**

for enterprise customer-support automation.

---

# 👨‍💻 Author

**Akash Kumar Tiwari**

M.Tech — Computer Science and Data Processing
Indian Institute of Technology Kharagpur

Research Interests:

`Artificial Intelligence` • `Machine Learning` • `Generative AI` • `Agentic AI` • `Retrieval-Augmented Generation`

GitHub: `@tiwariaakashprg-droid`

---

# ⭐ Support

If you find this project useful or interesting, consider giving the repository a ⭐.

---

<div align="center">

### 🛍️ ShopEase

**Multi-Agent AI × Hybrid RAG × Enterprise Customer Support**

Built with Python • LangGraph • LangChain • LLaMA 3.2 • FAISS • BM25 • Ollama • Streamlit

</div>
