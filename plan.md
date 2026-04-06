# Implementation Plan - RAG Law Chatbot

> Aligned with `todo.md`. Chi tiết implementation xem Step 2 trong todo.

---

## Progress

| Step | Description | Status |
|------|-------------|--------|
| 1 | Baseline evaluation | Done (30 pairs). Re-run needed: 100 pairs + fix `return_full_text` |
| 2 | Improvement experiments | Not started. Code spec in `todo.md` Step 2 |
| 3 | Best config final eval | Blocked by Step 2 |
| 4 | Report + Demo | Blocked by Step 3 |

---

## Architecture

```
Baseline.ipynb (15 cells)          Improvements.ipynb (to create)
├── Cell 0-3:  Setup               ├── Setup: imports, data, LLM, metrics
├── Cell 4-9:  Data exploration     ├── Exp A: Embedding model comparison
├── Cell 10-12: FAISS build         ├── Exp B: Chunking strategy
├── Cell 13: FAISS verify           ├── Exp C: Retrieval (top-k + BM25 hybrid)
└── Cell 14: RAG evaluation         ├── Exp D: Prompt engineering
                                    └── Aggregate: comparison table
```

## Environment

- **Runtime:** Kaggle GPU T4 (9h session limit, ~30min idle timeout)
- **Key timings (from baseline run):**
  - FAISS build: ~1h50m (CPU), ~30min (GPU)
  - Eval 30 pairs: ~8.5min (~17s/question)
  - Eval 100 pairs: ~28min estimated
- **Strategy:** Use N=50 for intermediate experiments, N=100 for final eval only

## Known Issues to Fix in Improvements.ipynb

1. **`return_full_text=True` (Critical):** Baseline LLM output includes prompt+context → inflates Token Overlap and Cosine Similarity. Fix: add `pipeline_kwargs={"return_full_text": False}` to `HuggingFacePipeline`.

2. **`max_length=1024` truncates context:** 5 chunks × ~440 tokens = ~2200 tokens input, exceeds limit. Fix: remove `max_length`, use only `max_new_tokens=512`.

3. **`df.head(30)` insufficient:** Requirement is minimum 100 pairs. Fix: `df.head(100)`.

---

## Output Files (expected after all steps)

```
report/results/
├── [DONE] doc_length_distribution.png, doc_type_distribution.png, top_keywords.png
├── [DONE] qa_length_distribution.png, sample_qa_pairs.csv
├── [DONE] chunk_length_distribution.png
├── [TODO] exp_A_embedding.csv
├── [TODO] exp_B_chunking.csv
├── [TODO] exp_C_retrieval.csv
├── [TODO] exp_D_prompt.csv
└── [TODO] comparison_table.csv    ← baseline vs best improved
```
