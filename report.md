# RAG-Based Vietnamese Law Question Answering System

**Course:** Large Language Models — Assignment  
**Author:** minhbuibhm  
**Date:** April 2026

---

## Abstract

This report presents a Retrieval-Augmented Generation (RAG) system for Vietnamese legal question answering, built on 16,439 legal documents and evaluated on 2,745 Q&A pairs. Starting from a baseline pipeline (`vietnamese-sbert` + FAISS + `Qwen1.5-1.8B`), we improve the system across four dimensions: embedding model, chunking strategy, retrieval method, and prompt engineering. Each configuration is evaluated on 100+ Q&A pairs using Cosine Similarity, Jaccard, Token Overlap, BLEU, and ROUGE-L. Results show that [**TODO: fill after experiments**].

---

## 1. Introduction

Vietnamese legal QA requires precise retrieval of specific legislative clauses from formal, terminology-heavy documents. Without retrieval augmentation, LLMs tend to hallucinate legal information — a critical failure mode in this domain. RAG addresses this by grounding generation in retrieved passages.

Starting from the course-provided baseline, we systematically improve each RAG component: embedding model, chunking, retrieval strategy, and prompt design. The corpus covers labor law, administrative decisions, and civil regulations from multiple governmental levels.

---

## 2. Data Exploration & Understanding

### 2.1 Legal Documents Corpus

The corpus consists of **16,439 Vietnamese legal text files** covering multiple domains of law from various governmental levels.

**Corpus statistics:**

| Metric | Value |
|--------|-------|
| Total documents | 16,439 |
| Total characters | 329,117,214 (~329M) |
| Mean doc length | 20,020 chars |
| Median doc length | 12,271 chars |
| Std dev | 29,448 chars |
| Min length | 737 chars |
| Max length | 798,241 chars |
| 25th percentile | 6,668 chars |
| 75th percentile | 22,313 chars |

**Document type distribution:**

| Document Type | Count | Percentage |
|--------------|-------|------------|
| Quyết định (QĐ) | 8,829 | 53.7% |
| Khác (Other) | 5,214 | 31.7% |
| UBND | 1,455 | 8.8% |
| Thông tư (TT) | 388 | 2.4% |
| Nghị định (NĐ) | 308 | 1.9% |
| Chính phủ (CP) | 115 | 0.7% |
| Chỉ thị (CT) | 79 | 0.5% |
| Thông báo (TB) | 39 | 0.2% |
| Nghị quyết (NQ) | 12 | 0.07% |

The corpus is dominated by Quyết định (Decisions, 53.7%) and administrative documents. High variance in document length (std = 29,448) means a single fixed chunk size may not suit all document types — motivating our chunking experiments in Section 4.2.

**Top keywords:** lao động (labor), quyết định (decision), chính phủ (government) — confirming the labor/administrative law focus. Note: raw unigram tokenization splits Vietnamese compound words; bigram analysis would yield more meaningful terms.

*See: [doc_type_distribution.png](report/results/doc_type_distribution.png), [doc_length_distribution.png](report/results/doc_length_distribution.png), [top_keywords.png](report/results/top_keywords.png)*

### 2.2 RES Dataset (Q&A Pairs)

The evaluation dataset contains **2,745 question-answer pairs** covering diverse legal topics.

**Q&A statistics:**

| Metric | Questions | Answers |
|--------|-----------|---------|
| Total pairs | 2,745 | 2,745 |
| Min length (chars) | 41 | 62 |
| Max length (chars) | 322 | 9,858 |
| Mean length (chars) | 141.0 | 1,619.3 |
| Median length (chars) | 137.0 | 1,426.0 |
| Mean length (words) | 30.6 | 350.9 |

Questions are concise (avg 30.6 words) while answers are ~11x longer (avg 350.9 words), often quoting legal articles verbatim. This means retrieval precision is critical — the system must find the exact passage among 188K chunks to produce accurate, detailed answers.

*See: [qa_length_distribution.png](report/results/qa_length_distribution.png), [sample_qa_pairs.csv](report/results/sample_qa_pairs.csv)*

### 2.3 Chunking Analysis

The baseline chunking strategy uses `RecursiveCharacterTextSplitter` with chunk_size=2000, overlap=50.

**Chunking statistics (baseline configuration):**

| Metric | Value |
|--------|-------|
| Total chunks | 188,226 |
| Total documents | 16,439 |
| Mean chunks/doc | 11.45 |
| Median chunks/doc | 7.0 |
| Max chunks/doc | 452 |
| Mean chunk length | 1,756 chars |
| Median chunk length | 1,893 chars |
| Min chunk length | 20 chars |
| Max chunk length | 2,000 chars |

188K chunks with median length ~1,893 chars (near the 2,000 max). The overlap of only 50 chars (2.5%) may cause information loss at chunk boundaries — motivating the overlap experiments in Section 4.2.

*See: [chunk_length_distribution.png](report/results/chunk_length_distribution.png)*

---

## 3. Baseline System

### 3.1 Baseline Configuration

| Component | Configuration |
|-----------|--------------|
| Chunking | RecursiveCharacterTextSplitter, chunk_size=2000, overlap=50 |
| Embedding Model | `keepitreal/vietnamese-sbert` |
| Vector DB | FAISS (local) |
| Top-k retrieval | k = 5 |
| LLM | `Qwen/Qwen1.5-1.8B` |
| Framework | LangChain |
| Evaluation set | 100 Q&A pairs |

### 3.2 Baseline Prompt Template

The baseline uses a simple Vietnamese instruction prompt that grounds the LLM on retrieved context and explicitly discourages hallucination:

```
Bạn là một trợ lý thông minh. Trả lời câu hỏi dựa trên ngữ cảnh sau.
Nếu không đủ thông tin, hãy nói rõ điều đó, không được tự bịa đặt.

Ngữ cảnh:
{context}

Câu hỏi: {question}
Trả lời ngắn gọn và chính xác nhất có thể:
```

### 3.3 Evaluation Metrics

All metrics compare generated answer (`ŷ`) against ground-truth answer (`y`). We use both semantic and lexical metrics to capture different aspects of answer quality:

| Metric | What it measures | Formula |
|--------|-----------------|---------|
| Cosine Similarity | Semantic similarity via embeddings | `cos(v_ŷ, v_y)` |
| Jaccard Similarity | Word-level set overlap | `\|W_ŷ ∩ W_y\| / \|W_ŷ ∪ W_y\|` |
| Token Overlap | Recall-oriented word overlap | `\|W_ŷ ∩ W_y\| / \|W_y\|` |
| BLEU | N-gram precision + brevity penalty | Standard BLEU |
| ROUGE-L | LCS-based F1 | `2PR/(P+R)` where P,R from LCS |

### 3.4 Baseline Results

Evaluated on 100 Q&A pairs (Cell 14 of `Baseline.ipynb`), with `return_full_text=False` and `max_new_tokens=256`.

| Metric | Baseline Score |
|--------|---------------|
| Cosine Similarity | 0.5697 |
| Jaccard Similarity | 0.1307 |
| Token Overlap | 0.1890 |
| BLEU | 0.1462 |
| ROUGE-L | 0.1537 |

**Observations:**
- **Cosine Similarity (0.57):** Moderate — the model captures the general topic but lacks precision on specific legal content.
- **Token Overlap (0.19) & Jaccard (0.13):** Low lexical overlap with ground truth, indicating generated answers diverge significantly in wording.
- **BLEU (0.15) & ROUGE-L (0.15):** Low but non-trivial — some n-gram and subsequence matches exist, suggesting partial content relevance.
- The LLM tends to repeat the question or generate generic text rather than citing specific legal articles (visible in output logs).

---

## 4. Improvements to RAG Pipeline

> **[TODO]** — Fill after running experiments. Each subsection: describe what was tried, show results table, explain selection rationale.

### 4.1 Embedding Model

Compare `vietnamese-sbert` (baseline) against models with stronger Vietnamese/multilingual capabilities. Rebuild FAISS index for each model, evaluate on same 100 Q&A pairs.

| Embedding Model | Cosine Sim | Jaccard | Token Overlap | BLEU | ROUGE-L |
|----------------|-----------|---------|--------------|------|---------|
| vietnamese-sbert (baseline) | 0.5697 | 0.1307 | 0.1890 | 0.1462 | 0.1537 |
| [candidate 1] | — | — | — | — | — |
| [candidate 2] | — | — | — | — | — |

**Selected:** [TODO] **Rationale:** [TODO]

### 4.2 Chunking Strategy

Test whether smaller chunks with larger overlap improve retrieval precision (at the cost of more chunks in the index).

| chunk_size | overlap | #chunks | Cosine Sim | ROUGE-L |
|-----------|---------|---------|-----------|---------|
| 2000 | 50 | 188K | 0.5697 | 0.1537 |
| 1000 | 200 | ~350K | — | — |
| 500 | 100 | ~650K | — | — |

**Selected:** [TODO]

### 4.3 Retrieval Strategy

Tune top-k and test hybrid retrieval (BM25 lexical + vector semantic).

| Strategy | k | Cosine Sim | ROUGE-L |
|----------|---|-----------|---------|
| Vector only (baseline) | 5 | 0.5697 | 0.1537 |
| Vector only | 3/7/10 | — | — |
| BM25 + Vector hybrid | 5 | — | — |

**Selected:** [TODO]

### 4.4 Prompt Engineering

Test prompt variants targeting the legal QA domain.

| Prompt Variant | Cosine Sim | ROUGE-L |
|---------------|-----------|---------|
| Baseline (simple instruction) | 0.5697 | 0.1537 |
| Chain-of-thought | — | — |
| Few-shot with examples | — | — |

**Selected:** [TODO]

---

## 5. Final Results & Comparison

> **[TODO]** — Fill after all experiments.

### 5.1 Best Configuration vs Baseline

| Component | Baseline | Improved |
|-----------|---------|---------|
| Embedding | vietnamese-sbert | — |
| Chunk size / overlap | 2000 / 50 | — |
| Retrieval | Vector k=5 | — |
| LLM | Qwen1.5-1.8B | Qwen1.5-1.8B |
| Prompt | Simple instruction | — |

### 5.2 Metric Comparison

| Metric | Baseline | Improved | Δ |
|--------|---------|---------|---|
| Cosine Similarity | 0.5697 | — | — |
| Jaccard Similarity | 0.1307 | — | — |
| Token Overlap | 0.1890 | — | — |
| BLEU | 0.1462 | — | — |
| ROUGE-L | 0.1537 | — | — |

*Evaluated on 100 Q&A pairs.*

### 5.3 Error Analysis

> [TODO: 2-3 examples where improvement helped, 1-2 where it didn't, with analysis of why]

---

## 6. Demo

> [TODO: 5-10 sample questions covering different legal domains, showing retrieved context + generated answer]

---

## 7. Conclusion

> [TODO: Which component had the largest impact? What are the limitations? What would you try next?]

---

## 8. Reproducibility

### Repository Structure

```
252_llm_asm/
├── Baseline.ipynb          # Main notebook (15 cells: setup, data exploration, FAISS build, evaluation)
├── Improvements.ipynb      # Improvement experiments (embedding, chunking, retrieval, prompt)
├── Dataset/export_1/       # 16,439 Vietnamese legal .txt documents
├── res.csv                 # 2,745 Q&A evaluation pairs
├── RES.xlsx                # Original Q&A source file
├── report/
│   └── results/            # Generated charts and JSON stats
├── faiss_db/               # Built FAISS index (generated by notebook)
└── report.md               # This report
```

### How to Run

1. **Environment:** Kaggle notebook with GPU T4 enabled, or local machine with CUDA GPU.

2. **Install dependencies:**
```bash
pip install langchain langchain-community langchain-huggingface faiss-cpu \
            sentence-transformers pandas numpy tqdm matplotlib scikit-learn openpyxl rank_bm25
```

3. **Run `Baseline.ipynb`** cells in order (Cell 0→14). Key stages:
   - Cells 0–9: Setup + data exploration + chunking
   - Cells 10–12: Build FAISS index + embeddings (~1h50m on CPU, ~30min on GPU)
   - Cells 13–14: Load/verify FAISS + run RAG evaluation (~8.5min for 30 pairs)

4. **For Kaggle:** `os.chdir('/kaggle/working/rag_chatbot_asm')` before executing.

---

## References

1. Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.
2. Lin. "ROUGE: A Package for Automatic Evaluation of Summaries." ACL Workshop 2004.
3. Papineni et al. "BLEU: a Method for Automatic Evaluation of Machine Translation." ACL 2002.
4. Manning et al. "Introduction to Information Retrieval." Cambridge University Press, 2008.
5. Johnson et al. "Billion-scale similarity search with GPUs." IEEE Transactions on Big Data, 2021. (FAISS)
6. Reimers & Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP 2019.
