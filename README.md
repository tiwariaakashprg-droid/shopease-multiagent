# 🛍️ ShopEase — Multi-Agent AI Customer Support System

A Generative AI-based Multi-Agent Customer Support System using LangGraph, Hybrid Retrieval-Augmented Generation (FAISS + BM25), LLaMA 3.2, CRM integration, conversational memory, Human-in-the-Loop escalation, and response reflection.

---

## 📌 Overview

ShopEase is an enterprise customer-support framework designed to automate e-commerce customer queries through a coordinated multi-agent architecture.

Instead of relying on a single LLM for the complete task, ShopEase decomposes customer-support reasoning into specialized agents for:

- Guardrail and safety screening
- Intent, sentiment, and urgency analysis
- CRM/customer-context retrieval
- Conversational memory
- Enterprise policy retrieval
- Human-in-the-Loop escalation
- Response generation
- Response reflection/validation

The system combines dense semantic retrieval using FAISS with sparse lexical retrieval using BM25. The retrieval pipelines additionally evaluate Reciprocal Rank Fusion (RRF), weighted RRF, and cross-encoder reranking.

---

## 🆕 Research & Engineering Improvements

### 1. Retrieval-driven classification

The evaluation pipeline predicts the policy category from the retrieved policy documents, rather than relying on a hardcoded intent-to-policy mapping. This makes the comparison between BM25, FAISS, RRF, Weighted RRF, and hybrid retrieval configurations a retrieval-based experiment.

The common retrieval utilities are implemented in:

```
graph/nodes/retrieval_utils.py
```

### 2. RRF-based Hybrid Retrieval

The retrieval pipeline:

```
Customer Query
      │
      ├───────────────┐
      ▼               ▼
    FAISS            BM25
  Dense Search    Lexical Search
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
```

### Retrieval configurations

| Configuration | Description |
|---|---|
| BM25-only | Sparse lexical retrieval |
| FAISS-only | Dense semantic retrieval |
| RRF-only (Fair) | FAISS + BM25 → RRF, without cross-encoder |
| Weighted RRF | FAISS + BM25 → weighted RRF fusion |
| RRF + Cross-Encoder | FAISS + BM25 → RRF → cross-encoder reranking |
| Top-10 Hybrid + Cross-Encoder | Larger candidate set followed by cross-encoder reranking |

The RRF-only fair configuration is kept separate from the cross-encoder configurations so that the retrieval ablation does not accidentally mix the effects of rank fusion and reranking.

---

## 🤖 Multi-Agent Architecture

ShopEase uses a LangGraph StateGraph.

```
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
                    │       HITL       │
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
```

### Important implementation detail

The CRM Agent and Memory Agent run in parallel after the Intent Agent because they operate on independent state fields. LangGraph then performs a fan-in before the RAG Agent. This reduces unnecessary sequential execution and enables per-agent timing analysis.

### 🧩 Agents

| Agent | Responsibility |
|---|---|
| Guardrail Agent | PII detection/redaction and prompt-injection detection |
| Intent Agent | Intent, sentiment, and urgency classification |
| CRM Agent | Customer profile and order-context retrieval, backed by a SQLite database (`data/shopease.db`) queried via parameterized SQL in `utils/db.py` (auto-seeded from `data/customers.csv` on first run) |
| Memory Agent | Conversation-history/context retrieval |
| Hybrid RAG Agent | Enterprise policy retrieval and retrieval-driven classification |
| Escalation Agent | Determines whether HITL intervention is required |
| Supervisor Agent | Generates the final customer-support response |
| Reflection Agent | Validates/refines the generated response |

All agents communicate through a shared `AgentState`.

---

## 🔍 Hybrid Retrieval-Augmented Generation

### 1. FAISS — Dense Retrieval

FAISS performs semantic similarity search using embeddings generated by `nomic-embed-text`. This helps retrieve relevant policies even when the customer's wording differs from the wording in the policy documents.

### 2. BM25 — Sparse Retrieval

BM25 performs lexical retrieval and is useful for exact policy terms such as:

- refund
- return
- damaged
- shipping
- cancellation

### 3. Reciprocal Rank Fusion

The ranked results from FAISS and BM25 are combined using Reciprocal Rank Fusion (RRF):

```
FAISS Top-k
     +
BM25 Top-k
     │
     ▼
RRF Fusion
     │
     ▼
Fused Ranking
```

### 4. Weighted RRF

Weighted RRF combines the ranked outputs of FAISS and BM25 while assigning different weights to the retrieval sources. This configuration is evaluated separately to determine whether weighting the retrieval sources improves policy classification performance.

### 5. Cross-Encoder Reranking

The hybrid configurations can rerank retrieved candidates using `cross-encoder/ms-marco-MiniLM-L-6-v2`.

- **Hybrid + Cross-Encoder** applies RRF followed by cross-encoder reranking.
- **Top-10 Hybrid + Cross-Encoder** evaluates a larger candidate set before cross-encoder reranking.

---

## 🛡️ Guardrails & Human-in-the-Loop

The **Guardrail Agent** runs before the downstream agents. It handles:

- PII detection/redaction
- Prompt-injection/jailbreak detection
- Safety-sensitive requests

The **Escalation Agent** can route cases requiring human judgment to a Human-in-the-Loop workflow. Examples include:

- Legal or regulatory threats
- Repeated unresolved complaints
- High-risk requests
- Sensitive customer situations
- Requests requiring human judgment

---

## 🖥️ Streamlit Application

ShopEase includes an interactive Streamlit interface with:

- Customer selection
- Customer profile and order information
- Interactive support chat
- Intent/sentiment/urgency trace
- CRM information
- RAG retrieval status
- HITL escalation status
- Session statistics
- Research Evaluation Dashboard
- Accuracy and latency comparison
- Retrieval ablation
- Component ablation
- Confusion matrices
- Classification reports
- Error analysis
- Statistical analysis

Run the application using:

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

The application provides both the customer-support interface and the research evaluation dashboard.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/tiwariaakashprg-droid/shopease-multiagent.git
cd shopease-multiagent
```

### 2. Create a Virtual Environment

**Windows**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 🦙 Ollama Setup

ShopEase uses local LLM inference through Ollama. Install the required models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Verify the installed models:

```bash
ollama list
```

You should have:

```
llama3.2
nomic-embed-text
```

### ▶️ Run ShopEase

Make sure Ollama is running, then start the Streamlit application:

```bash
streamlit run app.py
```

Open the application at:

```
http://localhost:8501
```

---

## 🧪 Evaluation

The final research evaluation uses **2,632 customer-support queries**. The generated evaluation files are stored under:

```
research_results/final_2632_evaluation/
```

Each retrieval configuration is evaluated on the same query set.

### Quick Evaluation

To evaluate a specific retrieval configuration:

```bash
python evaluate.py --mode hybrid
python evaluate.py --mode bm25
python evaluate.py --mode faiss
```

### 🔬 Full Research Evaluation

The repository contains separate scripts for retrieval evaluation, fair RRF comparison, statistical testing, component ablation, and result generation. A typical research workflow is:

**1. Evaluate Retrieval Configurations**

```bash
python evaluate.py --mode all
```

This evaluates the supported baseline and hybrid retrieval configurations: BM25, FAISS, Fair RRF, Weighted RRF, Hybrid + Cross-Encoder, and Top-10 Hybrid + Cross-Encoder.

**2. Fair RRF Evaluation**

```bash
python hybrid_rrf_fair.py
```

Stored as `research_results/final_2632_evaluation/evaluation_results_fair_rrf.csv`. This experiment evaluates FAISS + BM25 rank fusion without mixing in the Cross-Encoder reranking stage.

**3. Weighted RRF Evaluation**

Stored as `research_results/final_2632_evaluation/evaluation_results_weighted_rrf.csv`. Weighted RRF combines sparse and dense retrieval signals using weighted rank fusion. Final accuracy: **85.07%** (2,239 / 2,632 correct).

**4. Statistical Significance Testing**

```bash
python significance_test.py
```

Outputs stored under `research_results/final_2632_evaluation/statistical_tests/`, including `statistical_significance_tests.csv/.txt` and `v2` variants. McNemar's test compares paired predictions produced by different retrieval configurations on the same test queries.

**5. Baseline Comparison**

```bash
python baseline_compare.py --n 100
```

For a larger experiment, increase `--n` if sufficient compute resources and execution time are available.

**6. Component Ablation**

```bash
python ablation_study.py --n 20
```

Outputs stored under `research_results/final_2632_evaluation/ablation_study/`, including `ablation_queries_2632.csv`, `ablation_study_results_2632.csv`, `ablation_study_summary_2632.csv/.txt`, and `ablation_comparison_100.png`.

**7. Response-Quality Evaluation**

```bash
python llm_judge_eval.py --n 30
```

Evaluates generated responses using: Groundedness, Personalization, Relevance, and Overall quality.

**8. Escalation-Agent Evaluation**

```bash
python evaluate_escalation.py
```

Evaluates the Human-in-the-Loop escalation behavior of the Escalation Agent.

**9. Regenerate Evaluation Visualizations**

```bash
python confusion_matrix.py
python classification_report.py
python accuracy_graph.py
python latency_graph.py
```

**10. Retrieval Ablation Graph**

```bash
python retrieval_ablation_graph.py
```

Outputs: `research_results/final_2632_evaluation/graphs/retrieval_ablation_comparison_2632.png` and `.csv`.

**11. Final Result Aggregation**

```bash
python final_results.py
```

Aggregates the generated evaluation results for reporting and research analysis.

---

## 📊 Research Evaluation Dashboard

The Streamlit application also exposes a research-oriented dashboard using the generated files in `research_results/final_2632_evaluation/`.

The dashboard summarizes the final evaluation on **2,632 valid test queries** and includes:

- BM25-only
- FAISS-only
- RRF-only (Fair)
- Weighted RRF
- RRF + Cross-Encoder
- Top-10 Hybrid + Cross-Encoder

along with: accuracy comparison, latency comparison, retrieval ablation, component ablation, confusion matrices, classification reports, error analysis, and statistical significance analysis.

---

## 📈 Experimental Evaluation

The final evaluation was performed on **2,632 customer-support queries**, comparing six retrieval configurations:

1. BM25
2. FAISS
3. Fair RRF
4. Weighted RRF
5. Hybrid + Cross-Encoder
6. Top-10 Hybrid + Cross-Encoder

### 🏆 Main Retrieval Results

| Retrieval Configuration | Accuracy | Correct Predictions | Avg. Latency |
|---|---:|---:|---:|
| BM25-only | **55.74%** | 1,467 / 2,632 | **0.000494 s** |
| FAISS-only | **85.37%** | 2,247 / 2,632 | **0.046205 s** |
| RRF-only (Fair) | **82.29%** | 2,166 / 2,632 | **Not recorded** |
| Weighted RRF | **85.07%** | 2,239 / 2,632 | **0.041632 s** |
| RRF + Cross-Encoder | **83.24%** | 2,191 / 2,632 | **0.142449 s** |
| Top-10 Hybrid + Cross-Encoder | **81.88%** | 2,155 / 2,632 | **0.248934 s** |

**Accuracy calculation** (from the final evaluation CSV files):

- BM25: 1,467 / 2,632 = **55.74%**
- FAISS: 2,247 / 2,632 = **85.37%**
- Fair RRF: 2,166 / 2,632 = **82.29%**
- Weighted RRF: 2,239 / 2,632 = **85.07%**
- Hybrid + Cross-Encoder: 2,191 / 2,632 = **83.24%**
- Top-10 Hybrid + Cross-Encoder: 2,155 / 2,632 = **81.88%**

### ⏱️ Latency Analysis

Latency was recorded independently for the retrieval configurations where timing information was available.

| Experiment | Mean Latency | Median Latency | Minimum Latency | Maximum Latency | Std. Deviation |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.000494 s | 0.000410 s | 0.000198 s | 0.022928 s | 0.000548 s |
| FAISS | 0.046205 s | 0.044860 s | 0.028356 s | 0.951277 s | 0.019848 s |
| Fair RRF | Not recorded | Not recorded | Not recorded | Not recorded | Not recorded |
| Weighted RRF | 0.041632 s | 0.039754 s | 0.025253 s | 0.332629 s | 0.014609 s |
| Hybrid + CE | 0.142449 s | 0.111598 s | 0.077048 s | 15.210859 s | 0.445607 s |
| Top-10 Hybrid + CE | 0.248934 s | 0.177453 s | 0.092927 s | 18.114652 s | 0.502003 s |

The Fair RRF evaluation contains 2,632 predictions, but latency was not recorded for that experiment. Its latency is therefore reported as **Not recorded** rather than estimated, and no latency value is imputed or fabricated.

Latency results are stored in:

```
research_results/final_2632_evaluation/latency_analysis/latency_summary.csv
research_results/final_2632_evaluation/latency_analysis/latency_summary.txt
```

### 📈 Accuracy Comparison

Stored in `research_results/final_2632_evaluation/graphs/accuracy_comparison_2632.csv`, visualized in `.../accuracy_comparison_2632.png`.

| Retrieval Method | Correct | Total | Accuracy |
|---|---:|---:|---:|
| BM25 | 1467 | 2632 | 55.74% |
| FAISS | 2247 | 2632 | 85.37% |
| Fair RRF | 2166 | 2632 | 82.29% |
| Weighted RRF | 2239 | 2632 | 85.07% |
| Hybrid + CE | 2191 | 2632 | 83.24% |
| Top-10 Hybrid + CE | 2155 | 2632 | 81.88% |

FAISS provides the highest accuracy among the evaluated configurations, closely followed by Weighted RRF. The Cross-Encoder configurations do not outperform FAISS or Weighted RRF on this final evaluation set.

---

## 📉 Per-Class Performance

Classification reports are stored in `research_results/final_2632_evaluation/classification_reports/`:

- `classification_report_bm25.txt`
- `classification_report_faiss.txt`
- `classification_report_fair_rrf.txt`
- `classification_report_weighted_rrf.txt`
- `classification_report_hybrid.txt`
- `classification_report_top10_hybrid.txt`

These reports provide precision, recall, F1-score, and support for each policy category.

---

## 🎯 Confusion Matrix Analysis

Confusion matrices are generated separately for all six retrieval configurations, available under `research_results/final_2632_evaluation/confusion_matrices/`:

- `confusion_matrix_bm25.png` / `.csv`
- `confusion_matrix_faiss.png` / `.csv`
- `confusion_matrix_fair_rrf.png` / `.csv`
- `confusion_matrix_weighted_rrf.png` / `.csv`
- `confusion_matrix_hybrid.png` / `.csv`
- `confusion_matrix_top10_hybrid.png` / `.csv`

This allows detailed analysis of correct policy classification, confusion between similar policy categories, false positives, false negatives, and class-specific retrieval behavior.

---

## 🔬 Error Analysis

Stored in `research_results/final_2632_evaluation/error_analysis/`. For each configuration, two file types are generated:

**Wrong Predictions** (queries where predicted ≠ expected policy):
`bm25_wrong_queries.csv`, `faiss_wrong_queries.csv`, `fair_rrf_wrong_queries.csv`, `weighted_rrf_wrong_queries.csv`, `hybrid_wrong_queries.csv`, `top10_hybrid_wrong_queries.csv`

**Error Pairs** (most common expected → predicted error combinations):
`bm25_error_pairs.csv`, `faiss_error_pairs.csv`, `fair_rrf_error_pairs.csv`, `weighted_rrf_error_pairs.csv`, `hybrid_error_pairs.csv`, `top10_hybrid_error_pairs.csv`

Error analysis is useful for identifying difficult policy boundaries and understanding where retrieval-based classification fails.

---

## 📊 Retrieval Ablation Study

| Configuration | Description |
|---|---|
| BM25-only | Sparse lexical retrieval |
| FAISS-only | Dense semantic retrieval |
| Fair RRF | FAISS + BM25 → RRF |
| Weighted RRF | Weighted FAISS + BM25 → RRF |
| Hybrid + CE | RRF → Cross-Encoder reranking |
| Top-10 Hybrid + CE | Larger candidate set → Cross-Encoder reranking |

| Method | Accuracy |
|---|---:|
| BM25 | **55.74%** |
| FAISS | **85.37%** |
| Fair RRF | **82.29%** |
| Weighted RRF | **85.07%** |
| Hybrid + CE | **83.24%** |
| Top-10 Hybrid + CE | **81.88%** |

Visualization: `research_results/final_2632_evaluation/graphs/retrieval_ablation_comparison_2632.png`
Numerical data: `research_results/final_2632_evaluation/graphs/retrieval_ablation_comparison_2632.csv`

---

## 🧩 Component Ablation

ShopEase also evaluates the contribution of major components of the multi-agent architecture — how removing individual components affects response quality across:

- Groundedness
- Personalization
- Relevance
- Overall response quality

Generated by the response-quality evaluation pipeline. Final files under `research_results/final_2632_evaluation/ablation_study/`:

- `ablation_queries_2632.csv`
- `ablation_study_results_2632.csv`
- `ablation_study_summary_2632.csv`
- `ablation_study_summary_2632.txt`
- `ablation_comparison_100.png`

Intended to measure the practical contribution of CRM context, conversational memory, and escalation/HITL components to the overall support workflow.

---

## 🧪 Statistical Significance Analysis

Pairwise retrieval comparisons are evaluated using **McNemar's test**, appropriate here because the retrieval configurations produce paired predictions on the same set of customer queries.

Outputs stored under `research_results/final_2632_evaluation/statistical_tests/`:

- `statistical_significance_tests.csv`
- `statistical_significance_tests.txt`
- `statistical_significance_tests_v2.csv`
- `statistical_significance_tests_v2.txt`

Used to determine whether observed differences between retrieval configurations are statistically significant rather than simply caused by different prediction outcomes on the evaluation set.

---

## 📁 Final Research Results Structure

```text
research_results/
└── final_2632_evaluation/
    │
    ├── evaluation_results_bm25.csv
    ├── evaluation_results_faiss.csv
    ├── evaluation_results_fair_rrf.csv
    ├── evaluation_results_weighted_rrf.csv
    ├── evaluation_results_hybrid.csv
    ├── evaluation_results_top10_hybrid.csv
    │
    ├── ablation_study/
    │   ├── ablation_queries_2632.csv
    │   ├── ablation_study_results_2632.csv
    │   ├── ablation_study_summary_2632.csv
    │   ├── ablation_study_summary_2632.txt
    │   └── ablation_comparison_100.png
    │
    ├── classification_reports/
    │   ├── classification_report_bm25.txt
    │   ├── classification_report_faiss.txt
    │   ├── classification_report_fair_rrf.txt
    │   ├── classification_report_weighted_rrf.txt
    │   ├── classification_report_hybrid.txt
    │   └── classification_report_top10_hybrid.txt
    │
    ├── confusion_matrices/
    │   ├── confusion_matrix_bm25.csv
    │   ├── confusion_matrix_bm25.png
    │   ├── confusion_matrix_faiss.csv
    │   ├── confusion_matrix_faiss.png
    │   ├── confusion_matrix_fair_rrf.csv
    │   ├── confusion_matrix_fair_rrf.png
    │   ├── confusion_matrix_weighted_rrf.csv
    │   ├── confusion_matrix_weighted_rrf.png
    │   ├── confusion_matrix_hybrid.csv
    │   ├── confusion_matrix_hybrid.png
    │   ├── confusion_matrix_top10_hybrid.csv
    │   └── confusion_matrix_top10_hybrid.png
    │
    ├── error_analysis/
    │   ├── bm25_error_pairs.csv
    │   ├── bm25_wrong_queries.csv
    │   ├── faiss_error_pairs.csv
    │   ├── faiss_wrong_queries.csv
    │   ├── fair_rrf_error_pairs.csv
    │   ├── fair_rrf_wrong_queries.csv
    │   ├── weighted_rrf_error_pairs.csv
    │   ├── weighted_rrf_wrong_queries.csv
    │   ├── hybrid_error_pairs.csv
    │   ├── hybrid_wrong_queries.csv
    │   ├── top10_hybrid_error_pairs.csv
    │   └── top10_hybrid_wrong_queries.csv
    │
    ├── graphs/
    │   ├── accuracy_comparison_2632.csv
    │   ├── accuracy_comparison_2632.png
    │   ├── latency_comparison_2632.csv
    │   ├── latency_comparison_2632.png
    │   ├── retrieval_ablation_comparison_2632.csv
    │   └── retrieval_ablation_comparison_2632.png
    │
    ├── latency_analysis/
    │   ├── latency_summary.csv
    │   └── latency_summary.txt
    │
    └── statistical_tests/
        ├── statistical_significance_tests.csv
        ├── statistical_significance_tests.txt
        ├── statistical_significance_tests_v2.csv
        └── statistical_significance_tests_v2.txt
```

---

## 📦 Evaluation CSV Format

Each retrieval evaluation CSV contains fields such as:

| Field | Description |
|---|---|
| `query` | Customer question |
| `expected` | Ground-truth policy |
| `predicted` | Retrieved-policy prediction |
| `correct` | 1 or 0 |
| `confidence` | Prediction confidence |
| `retrieval_method` | Retrieval configuration |
| `latency` | Processing time |

This enables query-level analysis of expected policy, predicted policy, correctness, model confidence, retrieval configuration, and per-query latency.

---

## 📊 Final Accuracy Summary

```
BM25-only
Accuracy : 55.74%
Correct  : 1467 / 2632

FAISS-only
Accuracy : 85.37%
Correct  : 2247 / 2632

Fair RRF
Accuracy : 82.29%
Correct  : 2166 / 2632

Weighted RRF
Accuracy : 85.07%
Correct  : 2239 / 2632

Hybrid + Cross-Encoder
Accuracy : 83.24%
Correct  : 2191 / 2632

Top-10 Hybrid + Cross-Encoder
Accuracy : 81.88%
Correct  : 2155 / 2632
```

The final results show that dense semantic retrieval with FAISS provides a substantial improvement over BM25-only lexical retrieval on the evaluated customer-support queries. Weighted RRF achieves a comparable accuracy to FAISS while combining dense and sparse retrieval signals. The Cross-Encoder configurations introduce additional reranking computation and, in this evaluation, do not outperform the strongest FAISS/Weighted-RRF configurations. These results are therefore reported as an empirical comparison rather than assuming that additional reranking necessarily improves accuracy.

---

## 🎯 Example Queries

| Category | Example |
|---|---|
| Order Tracking | "Where is my order?" |
| Refund | "I want a refund for my order." |
| Damaged Product | "My product arrived damaged. What should I do?" |
| Multiple Policies | "My laptop is damaged. Can I return it and get a refund?" |
| Escalation | "I have complained several times and I want to take legal action." |

The last example can trigger the Human-in-the-Loop escalation mechanism.

---

## 🔬 Research Motivation

Modern enterprise customer-support systems must handle:

- Large volumes of customer queries
- Multiple enterprise policies
- Ambiguous user language
- Personalized customer information
- Safety-sensitive requests
- Hallucination risks
- Escalation requirements

ShopEase investigates how **Generative AI + Multi-Agent Systems + Hybrid RAG + CRM Context + Conversational Memory + Human-in-the-Loop** can be combined into a modular enterprise customer-support architecture.

The retrieval experiments further investigate how sparse retrieval, dense retrieval, rank fusion, weighted rank fusion, and cross-encoder reranking affect policy retrieval and retrieval-driven classification.

---

## 🚀 Future Improvements

Potential extensions include:

- FastAPI backend
- React / Next.js frontend
- Swap SQLite for PostgreSQL/MySQL in production
- Redis-based conversational memory
- Qdrant/Pinecone vector database
- Docker containerization
- Cloud deployment
- Authentication and authorization
- LangSmith/OpenTelemetry observability
- Improved reflection and self-correction
- Advanced reranking strategies
- Learned retrieval weighting
- Better handling of low-frequency/Unknown policies
- Voice-based customer support
- Multilingual support
- Real enterprise CRM integration
- Production monitoring and analytics

---

## 📄 Research

This repository contains the implementation and experimental evaluation of:

**ShopEase: A Generative AI-Based Multi-Agent Framework for Enterprise Customer Support Using Hybrid Retrieval-Augmented Generation**

The project investigates the combination of **Generative AI + Multi-Agent Systems + Hybrid RAG + CRM + Conversational Memory + Human-in-the-Loop AI** for enterprise customer-support automation.

The final experimental evaluation compares BM25, FAISS, Fair RRF, Weighted RRF, RRF + Cross-Encoder, and Top-10 Hybrid + Cross-Encoder on a common set of **2,632 customer-support queries**.

### 🎯 Reproducibility

For reproducibility, the final evaluation files preserve query-level information including `query`, `expected`, `predicted`, `correct`, `confidence`, `retrieval_method`, and `latency`. This enables researchers to independently inspect individual predictions, correct/incorrect queries, retrieval method behavior, confidence values, per-query latency, error patterns, confusion matrices, and statistical comparisons.

All six retrieval configurations are evaluated using the same 2,632-query evaluation set.

### 📚 Research Use

The generated evaluation artifacts can be used directly for: research-paper result tables, accuracy comparisons, retrieval ablation studies, error analysis, confusion-matrix analysis, latency analysis, statistical significance testing, and component-ablation analysis.

The final research artifacts are maintained separately from the main application files to keep the customer-support application and research evaluation workflow modular.

---

## 👨‍💻 Author

**Aakash Kumar Tiwari**
M.Tech — Computer Science and Data Processing
Indian Institute of Technology Kharagpur

**Research Interests:** Artificial Intelligence · Machine Learning · Generative AI · Agentic AI · Retrieval-Augmented Generation

**GitHub:** [@tiwariaakashprg-droid](https://github.com/tiwariaakashprg-droid)

<div align="center">

### 🛍️ ShopEase
Multi-Agent AI × Hybrid RAG × Enterprise Customer Support

Built with Python • LangGraph • LangChain • LLaMA 3.2 • FAISS • BM25 • Ollama • Streamlit

</div>