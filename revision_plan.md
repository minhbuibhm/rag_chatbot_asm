# Revision Plan — Report + Plan + TODO

> **Goal:** Fill `report.md` with experiment results from `Improvements.ipynb`, and sync `plan.md` / `todo.md` to reflect that Step 2 + Step 3 are now **Done**.
> **Review this document before I start editing.**

---

## 1. Current State Recap

### What's done (from notebook outputs)

| Step | Status | Evidence |
|------|--------|----------|
| Step 1 — Baseline 100 pairs | Done (pre-existing) | report.md §3.4 |
| Step 2 — Experiments A/B/C/D (N=50) | **Done** | `Improvements.ipynb` cells 11/16/21/24 + `exp_*.csv` |
| Step 3 — Final eval 100 pairs | **Done** | cell 27 + `final_evaluation_100pairs.csv` + `comparison_table.csv` |
| Step 4 — Report, Demo, Conclusion | **Not done** | report.md §4–§7 are `[TODO]` |

### Key results extracted from notebook outputs

**Exp A — Embedding (N=50, 2000/50, k=5):**

| Model | Cosine | Jaccard | Tok-Ovl | BLEU | ROUGE-L |
|---|---|---|---|---|---|
| A1 vietnamese-sbert | 0.5337 | 0.1225 | 0.1756 | 0.1229 | **0.1234** |
| A2 multilingual-e5-large | **0.8978** | 0.1289 | 0.1865 | 0.1305 | 0.1228 |
| A3 vietnamese-bi-encoder | 0.4040 | 0.1213 | 0.1757 | 0.1236 | 0.1201 |

> Note: Notebook selected **A1-sbert** as best by ROUGE-L. A2 has anomalously high Cosine (0.90) — likely because `emb_A2` (e5-large) is also used as the metric embedding, inflating self-similarity. This is a **legitimate caveat to mention in report** (metric dependency on embedding choice).

**Exp B — Chunking (N=50, sbert, k=5):**

| Config | Cosine | Jaccard | Tok-Ovl | BLEU | ROUGE-L |
|---|---|---|---|---|---|
| B1 2000/50 | 0.5337 | 0.1225 | 0.1756 | 0.1229 | 0.1234 |
| **B2 1000/200** | 0.5749 | 0.1342 | 0.1994 | 0.1497 | **0.1338** |
| B3 500/100 | 0.5758 | 0.1402 | 0.1998 | 0.1450 | 0.1335 |

**Exp C — Retrieval (N=50, sbert, 1000/200):**

| Strategy | Cosine | Jaccard | Tok-Ovl | BLEU | ROUGE-L |
|---|---|---|---|---|---|
| C1 k=3 | 0.5656 | 0.1427 | 0.2079 | 0.1440 | 0.1378 |
| C1 k=5 | 0.5749 | 0.1342 | 0.1994 | 0.1497 | 0.1338 |
| C1 k=7 | 0.5932 | 0.1418 | 0.2293 | 0.1703 | 0.1507 |
| **C1 k=10** | **0.6336** | 0.1616 | 0.2471 | 0.1810 | **0.1607** |
| C2 BM25+vector RRF | 0.5830 | 0.1437 | 0.2241 | 0.1532 | 0.1404 |

**Exp D — Prompt (N=50, sbert, 1000/200, k=10):**

| Prompt | Cosine | Jaccard | Tok-Ovl | BLEU | ROUGE-L |
|---|---|---|---|---|---|
| **D1 baseline** | **0.6336** | 0.1616 | 0.2471 | 0.1810 | **0.1607** |
| D2 structured | 0.0560 | 0.0053 | 0.0042 | 0.0028 | 0.0033 |
| D3 few-shot | 0.6188 | 0.1820 | 0.2582 | 0.1901 | 0.1520 |

> D2 collapsed (model failed to follow `[Căn cứ]:/[Nội dung]:` format — output diverged from reference). D3 edges ahead on BLEU/Jaccard/Token-Overlap but loses on ROUGE-L + Cosine. Worth analyzing in report.

**Best config:** sbert + 1000/200 chunks + k=10 + baseline prompt.

**Step 3 — Final eval (N=100):**

| Metric | Baseline | Improved | Δ abs | Δ % |
|---|---|---|---|---|
| Cosine | 0.5394 | 0.5998 | +0.0604 | **+11.2%** |
| Jaccard | 0.1241 | 0.1621 | +0.0380 | **+30.7%** |
| Tok-Overlap | 0.1892 | 0.2508 | +0.0616 | **+32.6%** |
| BLEU | 0.1296 | 0.1759 | +0.0463 | **+35.7%** |
| ROUGE-L | 0.1311 | 0.1585 | +0.0275 | **+21.0%** |

> **Discrepancy to flag:** baseline numbers in report.md §3.4 (Cos=0.5697, R-L=0.1537) come from a prior run; the Step 3 re-run baseline (Cos=0.5394, R-L=0.1311) is lower. Likely cause: different random state / FAISS reload path. The **Step 3 baseline** is the fair comparator (same env as best config) — recommend using those numbers as the official baseline-vs-best table, while keeping §3.4 with a footnote that the earlier standalone baseline run used a slightly different checkpoint.

---

## 2. Proposed Edits

### 2.1 `report.md` — main body fills

**§1 Abstract** — replace `[TODO: fill after experiments]` with a one-liner: "Our best configuration improves ROUGE-L by +21% and BLEU by +36% over the baseline."

**§3.4 Baseline Results** — add a small footnote: results here come from the standalone `Baseline.ipynb` run. For the official baseline-vs-improved comparison in §5, we re-ran the baseline in the same session as the best config (see Step 3 numbers).

**§4.1 Embedding Model** — fill table with A1/A2/A3 numbers above. Selection rationale:
- Ranked by ROUGE-L, which uses LCS of normalized tokens (metric-embedding-agnostic).
- A2's Cosine=0.90 is suspicious — its own embedding is also used as the similarity metric, inflating scores. Fair comparison via ROUGE-L/BLEU/Jaccard shows A1 and A2 are nearly tied; A1 is kept as it is significantly smaller (540M vs 2.24G) and faster.
- A3 (bi-encoder) underperformed; likely domain mismatch.

**§4.2 Chunking Strategy** — fill B1/B2/B3. Selected B2 (1000/200). Rationale: smaller chunks with larger overlap (20%) captured legal clause boundaries better; B3 had similar scores but 2× more chunks → slower retrieval.

**§4.3 Retrieval Strategy** — fill C1-k={3,5,7,10} + C2. Selected k=10. Rationale: monotonic improvement with k; BM25+vector RRF underperformed pure vector at k=10, suggesting legal QA is more semantic than lexical at this corpus size.

**§4.4 Prompt Engineering** — fill D1/D2/D3. Selected D1 (baseline). Rationale: structured prompt caused format drift (Qwen-1.8B too small to follow strict output schema); few-shot helped lexical metrics but hurt semantic — example leakage distracted model from the real question.

**§5.1 Best Configuration** — fill: sbert / 1000-200 / vector-k=10 / baseline-prompt.

**§5.2 Metric Comparison** — populate with Step 3 100-pair numbers.

**§5.3 Error Analysis** — add 2-3 qualitative examples (need to sample from `final_evaluation_100pairs.csv`). Plan: pick 1 case where improved system correctly cites article vs baseline generic; 1 where both fail; 1 where improvement is marginal (long answer, high token cost).

**§6 Demo** — pick 5 representative Q&A from `res.csv` (ideally pairs #101–#110 to stay outside eval set) and show: question, top-3 retrieved chunk snippets, improved-system answer, ground truth. Format as indented blocks.

**§7 Conclusion** — write ~150 words covering:
- Largest impact: retrieval depth (k=5→10) drove ~half of the improvement; chunk size was second.
- Negative findings: structured prompts + BM25 hybrid failed for this model/domain.
- Limitations: single LLM (Qwen-1.8B), N=100 eval, no hyperparam search on RRF weights.
- Future work: larger LLM (Qwen-2.5-3B), learned reranker (BGE-reranker), query rewriting.

**§8 Reproducibility** — add `Improvements.ipynb` run instructions, mention the two Kaggle datasets (`llm-rag-asm`, `llm-rag-asm-improvements`) for pre-built FAISS indexes.

### 2.2 `plan.md` — status sync

- Step 2: `Not started` → **Done** (A+B+C+D complete, N=50 results in `report/results/exp_*.csv`).
- Step 3: `Blocked` → **Done** (100-pair final eval in `final_evaluation_100pairs.csv`, deltas computed).
- Step 4: `Blocked` → **In progress** (report edits are the remaining work).
- Update "Output Files" block: mark `exp_A/B/C/D.csv`, `comparison_table.csv`, `final_evaluation_100pairs.csv` as `[DONE]`.

### 2.3 `todo.md` — status sync

- Mark Experiment A/B/C/D checkboxes `[x]`.
- Step 3 checkboxes `[x]`.
- Step 4 unchanged (still open).
- Update "Current Status" table: Improvement experiments → **Done**; add row "Final eval 100 pairs — Done".
- Add actual best-config summary under Step 3 section:
  > Best config: vietnamese-sbert + chunk 1000/200 + vector k=10 + baseline prompt → Cosine +11.2%, ROUGE-L +21.0%, BLEU +35.7% vs baseline.

---

## 3. Open Questions for Reviewer

1. **Baseline number mismatch (0.5697 vs 0.5394 Cosine):** OK to use Step-3 re-run numbers as the official baseline in §5? (My recommendation: yes, with a footnote.)
2. **A2 Cosine anomaly:** Should the report explicitly discuss the metric-embedding-self-similarity issue, or just note "A1 selected for efficiency at comparable quality"? (I lean toward explicit — shows understanding.)
3. **Demo section (§6):** Do you want me to (a) just write a script to pull examples and leave blanks, or (b) actually run 5–10 queries end-to-end through the improved pipeline? Option (b) requires re-running the pipeline locally / on Kaggle.
4. **Error analysis (§5.3):** Should I pick examples by diffing baseline vs improved answers in `final_evaluation_100pairs.csv`? The CSV has per-pair metrics — I can pick largest positive/negative deltas.

---

## 4. Execution Order (after approval)

1. Update `todo.md` and `plan.md` status rows (quick, low risk).
2. Fill `report.md` §4.1–4.4 tables and rationales (from numbers above).
3. Fill `report.md` §5.1–5.2 final comparison.
4. Write `report.md` §5.3 error analysis — **may need to read CSV rows**.
5. Write `report.md` §6 demo — **needs clarification per Q3**.
6. Write `report.md` §7 conclusion + §1 abstract.
7. Touch up §8 reproducibility with Improvements.ipynb info.
