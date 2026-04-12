# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Coursework repository for a **Vietnamese Legal Question-Answering RAG system** (HCMUT 252 — LLM assignment). The task: build a RAG pipeline over ~16,400 Vietnamese legal documents, evaluate on 2,745 Q&A pairs, and systematically improve retrieval + generation vs. a provided baseline. Grading: improvement (3pt) + demo (2pt) + report (5pt).

The work is organized as a sequence of Jupyter experiments (A: embedding, B: chunking, C: retrieval, D: prompt) with a final 100-pair evaluation comparing baseline vs. best config. Primary runtime is **Kaggle GPU T4** (9h session, ~30min idle timeout) — local dev is read/edit only; heavy cells are not meant to run on Windows.

## Architecture

Two-notebook pipeline, sharing the same data + metrics:

```
Baseline.ipynb            Improvements.ipynb
  └─ setup                  └─ setup (same imports, same metric fns as Baseline Cell 13)
  └─ data exploration       └─ Exp A: embedding model (3 candidates, rebuild FAISS each)
  └─ chunk + build FAISS    └─ Exp B: chunk size/overlap (3 configs, rebuild FAISS each)
  └─ eval 100 pairs ───┐    └─ Exp C: retrieval (top-k sweep + BM25+vector RRF hybrid)
                       │    └─ Exp D: prompt variants (baseline vs structured vs few-shot)
                       │    └─ Step 3: final baseline-vs-best eval on 100 pairs
                       ↓
                  report.md §3.4 ←───── experiment CSVs in report/results/ ────→ report.md §4–§5
```

**Key design decisions baked into `Improvements.ipynb`:**

- `build_or_load_faiss(emb_model, model_name, chunk_size, chunk_overlap)` is the single entry point for any FAISS index. It checks, in order: Kaggle baseline dataset → Kaggle improvements dataset → local `faiss_db/` or `WORK_DIR` cache → build from scratch. Always save new indexes to `WORK_DIR` (writable on Kaggle). **Do not bypass this helper** when adding new configs.
- LLM is loaded **once** at the top of the notebook with `pipeline_kwargs={"return_full_text": False}` and `max_new_tokens=256`. Earlier buggy configs (`return_full_text=True`, `max_length=1024` truncating context, `df.head(30)` too small) are fixed — do not re-introduce.
- `N_EVAL = 50` for intermediate experiments A/B/C/D (speed); `N_FINAL = 100` only for the Step-3 official comparison. Keep this split — it is the basis of the report's timing/cost budget.
- Metric embedding and retrieval embedding are the same object in each experiment (see Exp A caveat: e5-large's Cosine=0.90 is inflated by self-similarity — ROUGE-L/BLEU/Jaccard are the fair comparators).
- Kaggle detection (`ON_KAGGLE`, `WORK_DIR`, `KAGGLE_FAISS`, `KAGGLE_IMPROVEMENTS`) is set in the setup cell and every subsequent cell assumes these constants exist.

## Important Files

| File | Role |
|---|---|
| `requirements.md` | Assignment spec — problem statement, metrics, grading rubric. **Source of truth for what must be delivered.** |
| `plan.md` | Step-by-step implementation plan, progress tracker, environment notes. |
| `todo.md` | Detailed execution checklist with per-experiment notes and gotchas. |
| `report.md` | **Deliverable** — the final report (Abstract → Data → Baseline → Improvements → Results → Demo → Conclusion → Reproducibility). |
| `revision_plan.md` | Current in-flight plan for syncing report.md with Improvements.ipynb results. |
| `Baseline.ipynb` | Baseline pipeline: data exploration → FAISS build → eval on 100 pairs. Output values feed §3.4 of the report. |
| `Improvements.ipynb` | **Main experiment notebook.** Runs 4 experiments (A/B/C/D, N=50) + Step-3 final eval (N=100). Outputs all `exp_*.csv` and `comparison_table.csv`. |
| `Improvements_base.ipynb` | Earlier draft of Improvements.ipynb (kept for reference). |
| `res.csv` | 2,745 Q&A pairs (columns: `Câu Hỏi`, `Trả lời`). Evaluation ground truth. |
| `RES.xlsx` | Original Excel source for `res.csv`. |
| `RAG_Assignment.pdf` | Original assignment PDF from the course. |
| `rag_baseline_metrics.csv` | Per-row baseline eval output from `Baseline.ipynb`. |

## Important Folders

| Folder | Contents |
|---|---|
| `Dataset/export_1/` | 16,439 Vietnamese legal `.txt` documents (Quyết định, Thông tư, Nghị định, UBND decisions, etc.). Not committed — loaded on Kaggle from the `llm-rag-asm` dataset. |
| `faiss_db/` | Pre-built baseline FAISS index (vietnamese-sbert, chunk 2000/50, ~188K chunks). Loaded directly on Kaggle via `/kaggle/input/.../llm-rag-asm/faiss_db`. |
| `report/results/` | All generated artifacts: data exploration charts (`*_distribution.png`, `top_keywords.png`), JSON stats (`corpus_stats.json`, `chunk_stats.json`, `res_stats.json`), experiment CSVs (`exp_A_embedding.csv`, `exp_B_chunking.csv`, `exp_C_retrieval.csv`, `exp_D_prompt.csv`), `comparison_table.csv`, `final_evaluation_100pairs.csv`. Referenced from `report.md` figures/tables. |

## Kaggle Dataset Layout (read-only inputs)

- `/kaggle/input/datasets/minhbhm/llm-rag-asm/faiss_db` — baseline FAISS (vietnamese-sbert, 2000/50)
- `/kaggle/input/datasets/minhbhm/llm-rag-asm/embeddings_cache.pkl` — baseline embeddings cache
- `/kaggle/input/datasets/minhbhm/llm-rag-asm-improvements/faiss_db_<model>_<size>_<overlap>/` — pre-built FAISS for non-baseline configs (e.g. `faiss_db_multilingual-e5-large_2000_50/`, `faiss_db_vietnamese-sbert_1000_200/`)

**Updating cached FAISS on Kaggle:** use dataset "New Version" (not Create New Dataset) to preserve the slug `llm-rag-asm` / `llm-rag-asm-improvements` that the notebook hardcodes.

## Common Operations

Notebooks are not meant to be run end-to-end locally (FAISS build is ~1h50m CPU, eval ~28min per 100 pairs on T4). Typical local dev is editing cells via Python JSON manipulation.

**Inspect cell structure of a notebook:**

```bash
python3 -X utf8 << 'PYEOF'
import json
with open('Improvements.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])[:120]
    has_out = 'yes' if c.get('outputs') else 'no'
    print(f"Cell {i} [{c['cell_type']}] out={has_out} | {src}")
PYEOF
```

**Read a specific cell's output (to extract experiment results without rerunning):**

```bash
python3 -X utf8 << 'PYEOF'
import json
with open('Improvements.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
for out in nb['cells'][N].get('outputs', []):
    if 'text' in out:
        print(''.join(out['text']))
PYEOF
```

**Edit a cell's source in-place** (write full new source list of strings, preserve trailing `\n` on each line except possibly the last):

```bash
python3 -X utf8 << 'PYEOF'
import json
with open('Improvements.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
nb['cells'][N]['source'] = ["line1\n", "line2\n", ...]
with open('Improvements.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
PYEOF
```

**Environment bootstrap on Kaggle** (first cell):

```python
!pip install -q langchain langchain-community langchain-huggingface langchain-text-splitters
!pip install -q faiss-cpu sentence-transformers rank_bm25
!pip install -q pandas numpy tqdm scikit-learn
```

## Working Conventions

- **Shell is bash, not cmd/PowerShell** — use Unix paths (`/dev/null`), forward slashes, single quotes for heredocs.
- **Never rebuild a FAISS index locally** — it takes hours and will fill disk. Always check if a cached index exists in Kaggle input datasets first; the notebook's `build_or_load_faiss()` already handles this.
- When syncing `report.md` with notebook results, **read the cell outputs** (already persisted in the `.ipynb` JSON) rather than re-executing — all experiment numbers are already captured in outputs.
- The Step-3 re-run baseline numbers (same session as the best config) differ slightly from the standalone `Baseline.ipynb` baseline. Use the Step-3 numbers for the official §5 comparison; keep §3.4 with a footnote.

## Current Status (as of 2026-04-12)

- Baseline eval: **Done** (100 pairs).
- Experiments A/B/C/D: **Done** (N=50, results in `report/results/exp_*.csv`).
- Final eval (100 pairs, baseline vs best config): **Done** — all 5 metrics improved (+11% to +36%).
- Report writing: **In progress** — §4–§7 still contain `[TODO]` placeholders; plan in `revision_plan.md`.
- Demo section (5–10 sample Q&A): **Not started**.

**Best config:** `vietnamese-sbert` + chunk `1000/200` + vector retrieval `k=10` + baseline prompt.
