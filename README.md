🛍️ ShopEase — Multi-Agent AI Customer Support System

A Generative AI-based Multi-Agent Customer Support System using LangGraph, Hybrid Retrieval-Augmented Generation (FAISS + BM25), LLaMA 3.2, CRM integration, conversational memory, Human-in-the-Loop escalation, and response reflection.

📌 Overview

ShopEase is an enterprise customer-support framework designed to automate e-commerce customer queries through a coordinated multi-agent architecture.

Instead of relying on a single LLM for the complete task, ShopEase decomposes customer-support reasoning into specialized agents for:

Guardrail and safety screening

Intent, sentiment, and urgency analysis

CRM/customer-context retrieval

Conversational memory

Enterprise policy retrieval

Human-in-the-Loop escalation

Response generation

Response reflection/validation

The system combines dense semantic retrieval using FAISS with sparse lexical retrieval using BM25. The main hybrid pipeline additionally applies Reciprocal Rank Fusion (RRF) followed by an optional cross-encoder reranker.

🆕 Research & Engineering Improvements

1. Retrieval-driven classification

The evaluation pipeline now predicts the policy category from the retrieved policy documents, rather than relying on a hardcoded intent-to-policy mapping.

This makes the comparison between BM25, FAISS, RRF, and the main hybrid pipeline a retrieval-based experiment.

The common retrieval utilities are implemented in:

graph/nodes/retrieval_utils.py

2. RRF-based Hybrid Retrieval

The main retrieval pipeline is:

Customer Query
      │
      ├───────────────┐
      ▼               ▼
    FAISS            BM25
  Dense Search     Lexical Search
      │               │
      └───────┬───────┘
              ▼
       RRF Rank Fusion
              │
              ▼
     Cross-Encoder Reranker
              │
              ▼
      Top Policy Chunks
              │
              ▼
   Retrieval-driven Classification

Retrieval configurations

Configuration

Description

BM25-only

Sparse lexical retrieval

FAISS-only

Dense semantic retrieval

RRF-only (Fair)

FAISS + BM25 → RRF, without cross-encoder

RRF + Cross-Encoder

FAISS + BM25 → RRF → cross-encoder reranking; main hybrid configuration

The RRF-only fair configuration is kept separate from the main hybrid configuration so that the retrieval ablation does not accidentally get mixed with the cross-encoder results.

🤖 Multi-Agent Architecture

ShopEase uses a LangGraph StateGraph.

                         Customer Query
                              │
                              ▼
                    ┌──────────────────┐
                    │ Guardrail Agent  │
                    │ PII + Injection  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Intent Agent    │
                    │ Intent/Sentiment │
                    │     /Urgency     │
                    └────────┬─────────┘
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
          ┌──────────────┐       ┌──────────────┐
          │  CRM Agent   │       │ Memory Agent │
          └──────┬───────┘       └──────┬───────┘
                 │                      │
                 └──────────┬───────────┘
                            ▼
                    ┌──────────────────┐
                    │    Hybrid RAG    │
                    │ FAISS + BM25/RRF │
                    │ + Cross-Encoder  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Escalation Agent │
                    │      HITL         │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Supervisor Agent │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Reflection Agent │
                    └────────┬─────────┘
                             │
                             ▼
                       Final Response

Important implementation detail

The CRM Agent and Memory Agent run in parallel after the Intent Agent because they operate on independent state fields. LangGraph then performs a fan-in before the RAG Agent.

This reduces unnecessary sequential execution and enables per-agent timing analysis.

🧩 Agents

Agent

Responsibility

Guardrail Agent

PII detection/redaction and prompt-injection detection

Intent Agent

Intent, sentiment, and urgency classification

CRM Agent

Customer profile and order-context retrieval

Memory Agent

Conversation-history/context retrieval

Hybrid RAG Agent

Enterprise policy retrieval and retrieval-driven classification

Escalation Agent

Determines whether HITL intervention is required

Supervisor Agent

Generates the final customer-support response

Reflection Agent

Validates/refines the generated response

All agents communicate through a shared AgentState.

🔍 Hybrid Retrieval-Augmented Generation

1. FAISS — Dense Retrieval

FAISS performs semantic similarity search using embeddings generated by:

nomic-embed-text

This helps retrieve relevant policies even when the customer's wording differs from the wording in the policy documents.

2. BM25 — Sparse Retrieval

BM25 performs lexical retrieval and is useful for exact policy terms such as:

refund
return
damaged
shipping
cancellation

3. Reciprocal Rank Fusion

The ranked results from FAISS and BM25 are combined using Reciprocal Rank Fusion (RRF).

FAISS Top-k
     +
BM25 Top-k
     │
     ▼
RRF Fusion
     │
     ▼
Fused Ranking

4. Cross-Encoder Reranking

The main hybrid configuration optionally reranks the fused candidates using:

cross-encoder/ms-marco-MiniLM-L-6-v2

If the cross-encoder is unavailable, the implementation can fall back to the RRF ranking.

🛡️ Guardrails & Human-in-the-Loop

The Guardrail Agent runs before the downstream agents.

It handles:

PII detection/redaction

Prompt-injection/jailbreak detection

Safety-sensitive requests

The Escalation Agent can route cases requiring human judgment to a Human-in-the-Loop workflow.

Examples include:

Legal or regulatory threats

Repeated unresolved complaints

High-risk requests

Sensitive customer situations

Requests requiring human judgment

🖥️ Streamlit Application

ShopEase includes an interactive Streamlit interface with:

Customer selection

Customer profile and order information

Interactive support chat

Intent/sentiment/urgency trace

CRM information

RAG retrieval status

HITL escalation status

Session statistics

Research Evaluation Dashboard

Accuracy and latency comparison

Retrieval ablation

Component ablation

Confusion matrix

Statistical analysis

Run the application using:

streamlit run app.py

Then open:

http://localhost:8501

📊 Research Evaluation Dashboard

The Streamlit application also exposes a research-oriented dashboard using the generated files in:

research_results/

The dashboard summarizes the current evaluation on 590 valid test queries and includes:

BM25-only
FAISS-only
RRF-only (Fair)
RRF + Cross-Encoder

along with:

Accuracy comparison

Latency comparison

Retrieval ablation

Component ablation

Confusion matrix

Statistical significance analysis

📈 Experimental Evaluation

The original test CSV contained 595 rows.

Five rows were accidental header/metadata rows with:

expected_policy = expected_policy

These were removed before the final evaluation.

Therefore, the final valid evaluation dataset contains:

590 customer-support queries

Final class distribution

Policy Category

Queries

Damaged Product Policy

118

Refund Policy

117

Shipping Policy

117

Cancellation Policy

117

Return Policy

116

Unknown

5

Total

590

🏆 Main Retrieval Results

The current evaluation results are:

Retrieval Configuration

Accuracy

Avg. Latency

BM25-only

50.34%

0.0007 s

FAISS-only

92.71%

0.0579 s

RRF-only (Fair)

83.39%

—

RRF + Cross-Encoder

90.68%

0.1668 s

Main hybrid configuration

The main ShopEase retrieval configuration is:

FAISS + BM25 → RRF → Cross-Encoder

It achieved:

90.68% accuracy

535 / 590 correct predictions

0.1668 seconds average retrieval/evaluation latency

The RRF-only result is reported separately as an ablation:

83.39% accuracy (492 / 590)

This separation prevents the RRF-only and cross-encoder results from being incorrectly presented as the same experiment.

📉 Per-Class Performance — Main Hybrid Configuration

The current RRF + Cross-Encoder classification report is:

Policy

Precision

Recall

F1

Refund Policy

0.980

0.829

0.898

Return Policy

0.782

0.897

0.835

Shipping Policy

0.941

0.957

0.949

Cancellation Policy

0.907

1.000

0.951

Damaged Product Policy

0.981

0.890

0.933

Unknown

0.000

0.000

0.000

Overall:

Accuracy       : 0.907
Macro F1       : 0.761
Weighted F1    : 0.906

The Unknown category contains only five examples, and the current system does not correctly classify those five examples. This should be acknowledged as a limitation rather than hidden.

📊 Retrieval Ablation

The retrieval ablation isolates the contribution of the ranking/reranking stages:

BM25-only
FAISS-only
RRF-only (Fair)
RRF + Cross-Encoder

The key fair RRF result is:

RRF-only (Fair)
Accuracy: 83.39%
Correct : 492 / 590

The main hybrid configuration reaches:

RRF + Cross-Encoder
Accuracy: 90.68%
Correct : 535 / 590

This allows the experiment to separately evaluate:

Individual retrieval methods

RRF fusion

Cross-encoder reranking

🧪 Statistical Significance

Pairwise retrieval comparisons are evaluated using McNemar's test on the same test queries.

The current analysis reports statistically significant differences at:

p < 0.05

for:

BM25 vs FAISS

BM25 vs RRF

FAISS vs RRF

The generated result is saved to:

research_results/significance_test_results.csv

🧩 Component Ablation

ShopEase also evaluates the contribution of major agents/components.

Current summary:

Configuration

Groundedness

Personalization

Relevance

Overall

Full Pipeline

5.00

4.95

4.05

4.05

No CRM

5.00

2.10

4.00

4.00

No Escalation

4.95

4.90

3.90

3.95

No Memory

5.00

5.00

4.05

4.05

These scores are generated by the response-quality evaluation pipeline.

📁 Project Structure

shopease-multiagent/
│
├── app.py
├── requirements.txt
│
├── graph/
│   ├── state.py
│   ├── workflow.py
│   ├── supervisor.py
│   ├── reflection_agent.py
│   │
│   └── nodes/
│       ├── guardrail_agent.py
│       ├── intent_agent.py
│       ├── crm_agent.py
│       ├── memory_agent.py
│       ├── rag_agent.py
│       ├── rag_faiss.py
│       ├── rag_bm25.py
│       ├── retrieval_utils.py
│       ├── reranker.py
│       ├── escalation_agent.py
│       └── single_agent.py
│
├── utils/
│   ├── timing.py
│   └── guardrails.py
│
├── config/
│   └── calibration.json
│
├── data/
│   ├── customers.csv
│   ├── policies/
│   ├── policies.txt
│   ├── test_queries_clean.csv
│   └── escalation_test_set.csv
│
├── research_results/
│   ├── evaluation_results_bm25.csv
│   ├── evaluation_results_faiss.csv
│   ├── evaluation_results_hybrid.csv
│   ├── hybrid_rrf_fair_results.csv
│   ├── retrieval_comparison_summary.csv
│   ├── significance_test_results.csv
│   ├── confusion_matrix.png
│   ├── confusion_matrix_rrf_fair.png
│   ├── accuracy_comparison.png
│   ├── latency_comparison.png
│   └── retrieval_ablation_comparison.png
│
├── evaluate.py
├── hybrid_rrf_fair.py
├── significance_test.py
├── baseline_compare.py
├── ablation_study.py
├── llm_judge_eval.py
├── evaluate_escalation.py
├── calibrate_confidence.py
├── retrieval_ablation_graph.py
├── compare_agents.py
├── accuracy_graph.py
├── latency_graph.py
├── confusion_matrix.py
├── classification_report.py
└── final_results.py

⚙️ Installation

1. Clone the Repository

git clone https://github.com/tiwariaakashprg-droid/shopease-multiagent.git
cd shopease-multiagent

2. Create a Virtual Environment

Windows

python -m venv venv
venv\Scripts\activate

Linux / macOS

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

🦙 Ollama Setup

ShopEase uses local LLM inference through Ollama.

Install the required models:

ollama pull llama3.2

ollama pull nomic-embed-text

Verify:

ollama list

You should have:

llama3.2
nomic-embed-text

▶️ Run ShopEase

Make sure Ollama is running.

Then:

streamlit run app.py

Open:

http://localhost:8501

🧪 Evaluation

Quick evaluation

Run one retrieval configuration:

python evaluate.py --mode hybrid

or:

python evaluate.py --mode bm25
python evaluate.py --mode faiss

🔬 Full Research Evaluation

Recommended order:

1. Evaluate BM25, FAISS and main hybrid pipeline

python evaluate.py --mode all

2. Run fair RRF-only ablation

python hybrid_rrf_fair.py

This produces:

research_results/hybrid_rrf_fair_results.csv

3. Statistical significance

python significance_test.py

4. Baseline comparison

python baseline_compare.py --n 100

For the final paper, increase --n if sufficient compute time is available.

5. Component ablation

python ablation_study.py --n 20

6. Response-quality evaluation

python llm_judge_eval.py --n 30

7. Escalation-agent evaluation

python evaluate_escalation.py

8. Regenerate main charts/reports

python confusion_matrix.py
python classification_report.py
python accuracy_graph.py
python latency_graph.py

9. Retrieval ablation chart

python retrieval_ablation_graph.py

10. Final result aggregation

python final_results.py

📦 Research Outputs

The evaluation pipeline generates:

research_results/

including:

Accuracy comparison

Latency comparison

Confusion matrix

Classification report

Retrieval ablation

Component ablation

Wrong-prediction analysis

Statistical significance results

Evaluation CSVs

These files can be directly used to prepare the Results and Discussion section of the research paper.

🎯 Example Queries

Order Tracking

Where is my order?

Refund

I want a refund for my order.

Damaged Product

My product arrived damaged. What should I do?

Multiple Policies

My laptop is damaged. Can I return it and get a refund?

Escalation

I have complained several times and I want to take legal action.

The last example can trigger the Human-in-the-Loop escalation mechanism.

🔬 Research Motivation

Modern enterprise customer-support systems must handle:

Large volumes of customer queries

Multiple enterprise policies

Ambiguous user language

Personalized customer information

Safety-sensitive requests

Hallucination risks

Escalation requirements

ShopEase investigates how:

Generative AI + Multi-Agent Systems + Hybrid RAG + CRM Context + Conversational Memory + Human-in-the-Loop

can be combined into a modular enterprise customer-support architecture.

🚀 Future Improvements

Potential extensions include:

FastAPI backend

React / Next.js frontend

PostgreSQL-based CRM

Redis-based conversational memory

Qdrant/Pinecone vector database

Docker containerization

Cloud deployment

Authentication and authorization

LangSmith/OpenTelemetry observability

Improved reflection and self-correction

Advanced reranking

Voice-based customer support

Multilingual support

Real enterprise CRM integration

Production monitoring and analytics

📄 Research

This repository contains the implementation and experimental evaluation of:

ShopEase: A Generative AI-Based Multi-Agent Framework for Enterprise Customer Support Using Hybrid Retrieval-Augmented Generation

The project investigates the combination of:

Generative AI + Multi-Agent Systems + Hybrid RAG + CRM + Conversational Memory + Human-in-the-Loop AI

for enterprise customer-support automation.

👨‍💻 Author

Aakash Kumar Tiwari

M.Tech — Computer Science and Data Processing
Indian Institute of Technology Kharagpur

Research Interests:

Artificial Intelligence
Machine Learning
Generative AI
Agentic AI
Retrieval-Augmented Generation

GitHub:

@tiwariaakashprg-droid

<div align="center">

🛍️ ShopEase

Multi-Agent AI × Hybrid RAG × Enterprise Customer Support

Built with Python • LangGraph • LangChain • LLaMA 3.2 • FAISS • BM25 • Ollama • Streamlit

</div>