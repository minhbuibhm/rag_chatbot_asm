# Implementation Plan - RAG Law Chatbot

> Aligned with `todo.md`. Chi tiết implementation xem Step 2 trong todo.

---

## Progress

| Step | Description | Status |
|------|-------------|--------|
| 1 | Baseline evaluation | **Done** (100 pairs, `return_full_text=False`) |
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
  - Eval 100 pairs: ~28min (~17s/question)
- **Strategy:** Use N=50 for intermediate experiments, N=100 for final eval only

## Fixed Issues (already applied in Baseline.ipynb)

1. ~~`return_full_text=True`~~ → Fixed: `pipeline_kwargs={"return_full_text": False}`
2. ~~`max_length=1024` truncates context~~ → Fixed: uses `max_new_tokens=256`
3. ~~`df.head(30)` insufficient~~ → Fixed: `df.head(100)`

**Improvements.ipynb must carry these fixes forward** — copy the corrected LLM init from Cell 14.

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
