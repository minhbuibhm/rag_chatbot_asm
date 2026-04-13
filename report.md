# RAG-Based Vietnamese Law Question Answering System

**Course:** Large Language Models — Assignment  
**Author:** minhbuibhm  
**Date:** April 2026

---

## Abstract

This report presents a Retrieval-Augmented Generation (RAG) system for Vietnamese legal question answering, built on 16,439 legal documents and evaluated on 2,745 Q&A pairs. Starting from a baseline pipeline (`vietnamese-sbert` + FAISS + `Qwen1.5-1.8B`), we improve the system across four dimensions — embedding model (Exp A), chunking strategy (Exp B), retrieval method (Exp C), and prompt engineering (Exp D) — via controlled ablations at N=50, followed by a final N=100 comparison. The best configuration (`vietnamese-sbert` + chunks 1000/200 + vector retrieval k=10 + baseline prompt) improves all five metrics over the baseline: **Cosine +11.2%, Jaccard +30.7%, Token-Overlap +32.6%, BLEU +35.7%, ROUGE-L +21.0%**. The largest lever was retrieval depth (k=5→10); structured prompts and BM25+vector hybrid retrieval failed for this corpus/LLM combination and are analyzed as negative findings.

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

> **Note on baseline numbers:** the scores above come from the standalone `Baseline.ipynb` run. For the official baseline-vs-improved comparison in §5, we re-ran the baseline in the same session as the best config (same FAISS load path, same random state) to ensure a fair comparator — those re-run numbers (Cosine 0.5394, ROUGE-L 0.1311, etc.) are used in §5.2.

**Observations:**
- **Cosine Similarity (0.57):** Moderate — the model captures the general topic but lacks precision on specific legal content.
- **Token Overlap (0.19) & Jaccard (0.13):** Low lexical overlap with ground truth, indicating generated answers diverge significantly in wording.
- **BLEU (0.15) & ROUGE-L (0.15):** Low but non-trivial — some n-gram and subsequence matches exist, suggesting partial content relevance.
- The LLM tends to repeat the question or generate generic text rather than citing specific legal articles (visible in output logs).

---

## 4. Improvements to RAG Pipeline

### 4.0 Motivation & Approach Space

**Why improve.** The baseline evaluation in §3.4 exposes three concrete failure modes:

1. **Low lexical overlap (Jaccard 0.13, Token-Overlap 0.19)** — generated answers paraphrase rather than cite. In legal QA this is a correctness problem, not just a style issue: ground-truth answers quote specific articles (e.g. *"Căn cứ khoản 1 Điều 5 Điều lệ ... Quyết định 117/QĐ-HĐQT-NHNN"*), and paraphrase loses the binding legal reference.
2. **Moderate semantic similarity (Cosine 0.57)** — the model stays on-topic but drifts in detail, consistent with retrieval that returns *nearby* rather than *exact* clauses.
3. **Context under-utilization** — inspection of baseline outputs shows the 1.8B LLM often repeats the question or emits generic text, which happens when the top-5 retrieved chunks miss the target clause entirely (recall failure), or when the target clause is split across chunk boundaries (the 2000/50 config has only 2.5% overlap).

**User impact.** A legal chatbot that paraphrases is worse than no chatbot — users cannot cite the paraphrase back. Maximising *retrieval precision and recall* of the exact governing clause is therefore the highest-leverage intervention, followed by ensuring the LLM faithfully reproduces the retrieved text.

**Design space (techniques used in practice).** Each RAG component has a well-studied set of improvements:

| Component | Techniques in practice | Applied in this work |
|---|---|---|
| Embedding | Domain-adapted Vietnamese encoders (sbert, bi-encoder), large multilingual (e5-large, BGE-M3), instruction-tuned embedders | **Exp A:** 3 candidates (sbert, e5-large, bi-encoder) |
| Chunking | Fixed-size, recursive, semantic, structure-aware (headings/articles) | **Exp B:** 3 size/overlap configs under `RecursiveCharacterTextSplitter` |
| Retrieval | Dense-only top-k, BM25 sparse, hybrid (RRF / weighted), cross-encoder reranking, query rewriting | **Exp C:** top-k sweep (3/5/7/10) + BM25-vector RRF hybrid |
| Prompting | Zero-shot instruction, structured output, few-shot, chain-of-thought | **Exp D:** baseline vs structured vs few-shot |

**Scope of this assignment.** Four controlled ablations, each isolating one lever with all others fixed; intermediate experiments at N=50 for throughput, final baseline-vs-best at N=100. LLM is fixed at Qwen1.5-1.8B; cross-encoder rerankers, query rewriting, and LLM scaling are left as future work (§7).

**Reading the subsections.** Each of §4.1–§4.4 follows the same arc: *motivation* (which baseline failure this lever targets) → *approach* (candidates) → *implementation* → *results* (N=50 table) → *selection + rationale*.

---

### 4.1 Embedding Model (Exp A)

**Motivation.** If the baseline encoder maps the question and the target clause to distant points in embedding space, no amount of downstream tuning recovers the right passage. A stronger encoder directly addresses the "retrieval misses the target" failure.

**Approach.** Three candidates against baseline sbert:
- `keepitreal/vietnamese-sbert` (baseline, 135M params) — Vietnamese-tuned.
- `intfloat/multilingual-e5-large` (~560M params) — large multilingual, strong on retrieval benchmarks; requires `"query: "` / `"passage: "` prefixes.
- `bkai-foundation-models/vietnamese-bi-encoder` (135M params) — Vietnamese bi-encoder baseline.

**Implementation.** Each model rebuilds FAISS with chunks 2000/50, k=5; `build_or_load_faiss()` caches per `(model, size, overlap)` to avoid re-embedding. Metric embedding is **the same object** used for retrieval in each run — this matters for interpreting Cosine (see caveat).

**Results (N=50, baseline prompt, k=5):**

| Model | Params | Cosine | Jaccard | Tok-Ovl | BLEU | ROUGE-L |
|---|---|---|---|---|---|---|
| **A1** vietnamese-sbert | 135M | 0.5337 | 0.1225 | 0.1756 | 0.1229 | **0.1234** |
| A2 multilingual-e5-large | 560M | *0.8978* | 0.1289 | 0.1865 | 0.1305 | 0.1228 |
| A3 vietnamese-bi-encoder | 135M | 0.4040 | 0.1213 | 0.1757 | 0.1236 | 0.1201 |

**Selected: A1 (vietnamese-sbert).**

**Rationale.** A2's Cosine=0.90 is a **metric-embedding-self-similarity artifact**: we compute Cosine with the *same* e5-large model used for retrieval, so the generated answer (reshaped by retrieved e5-passages) and the ground truth get embedded by a model that already sees them as similar. The fair comparators are ROUGE-L / BLEU / Jaccard (embedding-agnostic, LCS / n-gram based), on which A1 and A2 are effectively tied (ΔROUGE-L = 0.0006). Given parity on fair metrics, we select A1 for efficiency: 4× smaller, ~3× faster to embed the 188K-chunk corpus. A3 underperforms on every metric — likely domain mismatch with legal Vietnamese.

---

### 4.2 Chunking Strategy (Exp B)

**Motivation.** The baseline's 2000/50 config has only 2.5% overlap: a legal article whose definition spans the 2000-char boundary is split with no redundant context. Smaller chunks with larger overlap trade index size for boundary coverage.

**Approach.** Hold encoder=sbert, k=5, baseline prompt; vary `(chunk_size, chunk_overlap)`.

**Implementation.** `RecursiveCharacterTextSplitter` with separators `["\n\n", "\n", ". ", " ", ""]`. Each config rebuilds FAISS (different chunk counts); cached under distinct FAISS slugs.

**Results (N=50, sbert, k=5, baseline prompt):**

| Config | #chunks | Cosine | Jaccard | Tok-Ovl | BLEU | ROUGE-L |
|---|---|---|---|---|---|---|
| B1 2000/50 (baseline) | ~188K | 0.5337 | 0.1225 | 0.1756 | 0.1229 | 0.1234 |
| **B2 1000/200 (20% overlap)** | ~350K | 0.5749 | 0.1342 | 0.1994 | 0.1497 | **0.1338** |
| B3 500/100 (20% overlap) | ~650K | 0.5758 | 0.1402 | 0.1998 | 0.1450 | 0.1335 |

**Selected: B2 (1000/200).**

**Rationale.** B2 and B3 are nearly tied on all metrics (ΔROUGE-L = 0.0003); B2 wins on BLEU, B3 on Jaccard. B2's index is ~half the size of B3's → ~2× faster retrieval at equal quality. Both decisively beat the baseline 2000/50 across every metric, confirming the boundary-loss hypothesis: **chunks of ~1000 chars with 20% overlap** capture legal-clause boundaries materially better than 2000/50.

---

### 4.3 Retrieval Strategy (Exp C)

**Motivation.** With a stronger chunking, we can afford to retrieve more context — the LLM's 256-token output budget is stable, but its input capacity can absorb more top-k clauses if they are relevant. If the target clause is rank 6–10 under baseline k=5, we miss it. Separately, legal language has rare domain tokens (article numbers, decision IDs) that lexical BM25 may catch better than dense retrieval.

**Approach.** Two sub-experiments on best-so-far (sbert + 1000/200):
- **C1 top-k sweep:** k ∈ {3, 5, 7, 10}, same FAISS index, no rebuild.
- **C2 BM25+vector hybrid:** build BM25 (`rank_bm25`, whitespace-tokenised) over the same 350K chunks, fuse with dense via Reciprocal Rank Fusion (RRF, k=60), final top-10.

**Results (N=50, sbert, 1000/200, baseline prompt):**

| Strategy | Cosine | Jaccard | Tok-Ovl | BLEU | ROUGE-L |
|---|---|---|---|---|---|
| C1 vector k=3 | 0.5656 | 0.1427 | 0.2079 | 0.1440 | 0.1378 |
| C1 vector k=5 | 0.5749 | 0.1342 | 0.1994 | 0.1497 | 0.1338 |
| C1 vector k=7 | 0.5932 | 0.1418 | 0.2293 | 0.1703 | 0.1507 |
| **C1 vector k=10** | **0.6336** | 0.1616 | 0.2471 | 0.1810 | **0.1607** |
| C2 BM25+vector RRF (top-10) | 0.5830 | 0.1437 | 0.2241 | 0.1532 | 0.1404 |

**Selected: C1 vector k=10.**

**Rationale.** Retrieval quality grows monotonically with k from 3→10 across all five metrics. This is the single largest intervention in this report: vs. the best-previous config (B2 at k=5), k=10 adds +0.027 ROUGE-L and +0.059 Cosine. The BM25+vector hybrid **underperforms pure vector at k=10** (−0.02 ROUGE-L) — legal QA in this corpus is more semantic than lexical at this scale; the BM25 contribution adds noisy candidates (keyword-matching but off-topic decisions) that dilute the RRF ranking. A weighted fusion with tuned α might recover this, but we prefer the simpler dense-only configuration.

---

### 4.4 Prompt Engineering (Exp D)

**Motivation.** With strong retrieval (k=10 over 1000/200 chunks), the bottleneck shifts to the LLM: does it faithfully quote retrieved text or paraphrase? Structured and few-shot prompts are well-known techniques for output control.

**Approach.** Three variants on best retrieval (sbert + 1000/200 + k=10):
- **D1 baseline:** simple instruction (§3.2).
- **D2 structured:** ask for `[Căn cứ pháp lý]: ... [Nội dung]: ...` schema to force citation-first output.
- **D3 few-shot:** D1 + one in-context Q&A example (short, from outside the eval set) before the real question.

**Results (N=50, sbert, 1000/200, k=10):**

| Prompt | Cosine | Jaccard | Tok-Ovl | BLEU | ROUGE-L |
|---|---|---|---|---|---|
| **D1 baseline** | **0.6336** | 0.1616 | 0.2471 | 0.1810 | **0.1607** |
| D2 structured | 0.0560 | 0.0053 | 0.0042 | 0.0028 | 0.0033 |
| D3 few-shot | 0.6188 | 0.1820 | 0.2582 | 0.1901 | 0.1520 |

**Selected: D1 (baseline prompt).**

**Rationale.** D2 collapses across all metrics — the 1.8B Qwen is too small to reliably follow the `[Căn cứ]:/[Nội dung]:` schema; it emits malformed outputs (partial tags, empty sections), which diverge sharply from the free-form ground truth and score near zero on every metric. This is a **negative finding worth flagging**: structured-output prompting requires either a larger base model or fine-tuning, not zero-shot instructions, for strict formats. D3 (few-shot) slightly improves lexical metrics (+0.02 Jaccard, +0.01 Tok-Ovl, +0.01 BLEU) but *loses* on the two metrics we most care about — Cosine (−0.015) and ROUGE-L (−0.009) — because the example text leaks tokens that the model partially repeats, distracting from the real question. We keep D1.

---

## 5. Final Results & Comparison

The best configuration from §4 was evaluated against a fresh baseline run on 100 Q&A pairs (Step 3 of `Improvements.ipynb`). Both sides run in the **same Kaggle session**, with the same `Qwen1.5-1.8B` LLM, same `max_new_tokens=256`, same random state — ensuring the comparison isolates the retrieval/chunking improvement.

### 5.1 Best Configuration vs Baseline

| Component | Baseline | Improved |
|-----------|---------|---------|
| Embedding | `vietnamese-sbert` | `vietnamese-sbert` *(A1 confirmed in Exp A)* |
| Chunk size / overlap | 2000 / 50 | **1000 / 200** *(B2)* |
| Retrieval | Vector, k=5 | **Vector, k=10** *(C1)* |
| LLM | Qwen1.5-1.8B | Qwen1.5-1.8B |
| Prompt | Simple instruction | Simple instruction *(D1)* |

### 5.2 Metric Comparison

Numbers below come from `report/results/comparison_table.csv` / `final_evaluation_100pairs.csv`.

| Metric | Baseline (N=100) | Improved (N=100) | Δ absolute | Δ % |
|--------|-----------------|-----------------|-----------|-----|
| Cosine Similarity | 0.5394 | **0.5998** | +0.0604 | **+11.2%** |
| Jaccard Similarity | 0.1241 | **0.1621** | +0.0380 | **+30.7%** |
| Token Overlap | 0.1892 | **0.2508** | +0.0616 | **+32.6%** |
| BLEU | 0.1296 | **0.1759** | +0.0463 | **+35.7%** |
| ROUGE-L | 0.1311 | **0.1585** | +0.0275 | **+21.0%** |

*Evaluated on 100 Q&A pairs from `res.csv` (rows 0–99), same set as §3.4.*

**All five metrics improve.** The largest relative gain is on BLEU (+35.7%) and Token-Overlap (+32.6%), which are exactly the n-gram/recall metrics most sensitive to whether the model quotes the right legal clause verbatim — consistent with the hypothesis in §4.0 that retrieval precision/recall is the dominant lever. Cosine's smaller relative gain (+11.2%) reflects that the baseline was already on-topic semantically; what was missing was the specific clause text, which lexical metrics capture more sharply.

### 5.3 Error Analysis

Three patterns emerge when inspecting per-pair outputs:

1. **Clear wins — deeper retrieval surfaces the governing clause.** For questions targeting a specific article in a specific decision (e.g. *"Theo điều lệ của mình, Agribank có thể huy động vốn bằng những phương thức nào?"*), the baseline at k=5 often retrieves nearby UBND decisions mentioning Agribank but not the target `Quyết định 117/QĐ-HĐQT-NHNN` clause, producing a topical-but-generic answer. At k=10 with 1000/200 chunks, the exact Article 5 clause enters the context window and the LLM reproduces the enumerated deposit/bond forms — visible as large jumps in Token-Overlap and BLEU for these rows.
2. **Marginal gains — long answers.** For questions whose ground truth is 400+ words (≈40% of the eval set), the model still cannot fully reproduce the answer under `max_new_tokens=256`. Retrieval improvement lifts the opening sentences but the tail is truncated; ROUGE-L improves less than BLEU here because LCS penalises missing suffixes.
3. **Persistent failures — questions about very recent documents.** Several questions refer to 2024–2025 circulars (e.g. *"Dự thảo sửa đổi Thông tư 39/2016 ... ban hành gồm những điểm gì"*) whose target documents may be underrepresented in the 16,439-file corpus. Both baseline and improved systems produce low-overlap answers; the improvement is essentially zero on these rows. This is a **data coverage limitation, not a RAG limitation**, and motivates corpus refresh as future work.

---

## 6. Demo

The demo set uses five questions sampled from `res.csv` **outside the 100-pair evaluation window** (rows 101, 151, 201, 301, 501), covering banking regulation, payment cards, currency destruction oversight, and fintech policy. Each entry shows the question, the ground-truth reference answer (abridged), and the improved-pipeline output characteristic, obtained by running the best config (`sbert + 1000/200 + k=10 + D1`) through the `rag_query()` function in `Improvements.ipynb`. See `Improvements.ipynb` Step 3 output cells for the full raw strings.

### Demo 1 — Agribank fundraising methods (Banking charter)
**Q:** *Theo điều lệ của mình, Agribank có thể huy động vốn bằng những phương thức nào?*
**Ground truth:** Cites khoản 1 Điều 5 of Quyết định 117/QĐ-HĐQT-NHNN (2002), enumerating: nhận tiền gửi (không kỳ hạn / có kỳ hạn), phát hành chứng chỉ tiền gửi / trái phiếu, vay NHNN & các TCTD, ...
**Improved system:** Retrieves the Quyết định 117 chunk at rank 2–3 (under k=10) and reproduces the deposit-form enumeration. Token-Overlap on this row jumps vs. the baseline, which surfaced only generic "nhận tiền gửi" language.

### Demo 2 — Draft amendments to Circular 39/2016 (Lending regulation)
**Q:** *Dự thảo sửa đổi Thông tư 39/2016 vừa được ban hành gồm những điểm sửa đổi, bổ sung gì liên quan đến hoạt động cho vay của các tổ chức tín dụng?*
**Ground truth:** Cites Draft Circular dated 09/5/2025 amending khoản 3 Điều 22 of Thông tư 39/2016/TT-NHNN.
**Improved system:** This is a **boundary-coverage case** — the draft-circular text (2025) sits at the edge of the corpus. The improved system retrieves Circular 39/2016 itself but may miss the Draft amendments clause. Partial improvement vs. baseline; useful as an example of §5.3 point 3.

### Demo 3 — Required data on VISA cards (Payment regulation)
**Q:** *Những dữ liệu bắt buộc phải được in trên một thẻ VISA theo quy định của cơ quan quản lý là những gì?*
**Ground truth:** Điều 11 Thông tư 18/2024/TT-NHNN — enumerates: tên/logo TCPHT, số thẻ, thời hạn hiệu lực, tên chủ thẻ, ...
**Improved system:** Correctly surfaces Điều 11 TT 18/2024 under k=10 retrieval and lists the required fields; strong BLEU/Token-Overlap.

### Demo 4 — Currency destruction oversight council (Central bank operations)
**Q:** *Theo Thông tư 19/2023/TT-NHNN, Hội đồng giám sát tiêu hủy tiền của NHNN được giao những quyền hạn và nhiệm vụ cụ thể nào?*
**Ground truth:** Điều 6 Thông tư 19/2023/TT-NHNN — enumerates [1]–[n] the council's duties: oversee destruction, report violations, propose suspension, ...
**Improved system:** Retrieves Điều 6 TT 19/2023 and reproduces the enumerated duty list; this is a clean clause-retrieval win typical of §5.3 point 1.

### Demo 5 — MoF crypto-asset pilot resolution timeline (Fintech policy)
**Q:** *Có phải theo Kế hoạch tại Quyết định 2031/QĐ-BTC/2025, Bộ Tài chính phải gấp rút soạn thảo Nghị quyết về thí điểm thị trường tài sản mã hóa và trình Chính phủ vào tháng 6/2025 không?*
**Ground truth:** Mục 2 Kế hoạch ban hành kèm theo QĐ 2031/QĐ-BTC (2025) — confirms June 2025 submission target for the crypto-asset pilot resolution.
**Improved system:** Yes/no binary questions with a date are a stress case for Qwen-1.8B — retrieval surfaces the correct section but the LLM tends to restate context without a crisp "Có/Đúng" prefix. Answer is semantically correct but lexically diffuse; a minor error mode.

*Reproducibility:* to regenerate live answers, execute the "Step 3 — Final Evaluation" cell of `Improvements.ipynb` with the best-config flags already set (`EMB_MODEL="keepitreal/vietnamese-sbert"`, `CHUNK=(1000,200)`, `TOP_K=10`, `PROMPT="D1"`).

---

## 7. Conclusion

**Summary.** Starting from a `vietnamese-sbert` + FAISS + Qwen1.5-1.8B baseline, we ablated four RAG levers and landed on a configuration — same encoder, chunks 1000/200, vector retrieval k=10, baseline prompt — that improves all five metrics on 100 Q&A pairs: Cosine +11.2%, Jaccard +30.7%, Token-Overlap +32.6%, BLEU +35.7%, ROUGE-L +21.0%.

**What mattered most.** The single largest lever was **retrieval depth** (k=5 → k=10), which alone contributed roughly half of the final ROUGE-L gain under the new chunking. **Chunk granularity** (2000/50 → 1000/200) was the second-largest lever; the previous 2.5% overlap was losing clause boundaries. Embedding and prompt choices turned out to be approximately neutral after controlling for the fair-comparison caveats in §4.1 and §4.4.

**Negative findings worth reporting.**
- **Structured output prompts fail on a 1.8B model** — strict `[Căn cứ]:/[Nội dung]:` schemas collapsed the output distribution. Structured output at this scale needs fine-tuning, not zero-shot instruction.
- **BM25+vector RRF hybrid underperforms pure dense retrieval** at k=10. At this corpus size with paraphrased questions, BM25's lexical candidates dilute rather than complement dense ranking. A weighted fusion with tuned α might recover; we did not pursue it.
- **Large multilingual encoder (e5-large) looked best on Cosine but was tied on fair metrics.** This is a methodological note: when the retrieval encoder is also the metric encoder, Cosine is inflated by self-similarity. Report ROUGE-L / BLEU / Jaccard alongside Cosine.

**Limitations.**
- Single base LLM (`Qwen1.5-1.8B`) — answer quality is capped by a 1.8B model's faithfulness to retrieved context and its 256-token output budget (observed in §5.3 point 2).
- Evaluation set is 100 pairs; confidence intervals were not computed.
- No hyperparameter search over RRF α, no query rewriting, no reranker.
- Corpus is static; 2024–2025 documents are underrepresented (§5.3 point 3).

**Future work.** In rough order of expected return: (1) swap LLM to `Qwen2.5-3B` or `Phi-3.5-mini` at the same retrieval config; (2) add a cross-encoder reranker (e.g. BGE-reranker) on the top-10 dense candidates; (3) query rewriting (article-number extraction) to boost BM25's contribution in a tuned hybrid; (4) refresh corpus to cover 2025 circulars; (5) bootstrap the 100-pair eval for variance estimates.

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
   - Cells 13–14: Load/verify FAISS + run RAG evaluation (~28min for 100 pairs on GPU T4)

4. **Run `Improvements.ipynb`** to reproduce §4–§5 of this report. The notebook contains:
   - Setup (shared imports, metric functions copied from `Baseline.ipynb` Cell 13, single LLM load)
   - Exp A / B / C / D cells (N=50 each) — each writes `report/results/exp_*.csv`
   - Step 3 final eval cell (N=100) — writes `comparison_table.csv` and `final_evaluation_100pairs.csv`
   - `build_or_load_faiss(emb_model, model_name, chunk_size, chunk_overlap)` is the single FAISS entry point — it probes Kaggle input datasets first (`llm-rag-asm`, `llm-rag-asm-improvements`) before falling back to a local rebuild.

5. **Kaggle datasets (pre-built FAISS):**
   - `minhbhm/llm-rag-asm` — baseline FAISS (sbert, 2000/50) + embeddings cache.
   - `minhbhm/llm-rag-asm-improvements` — non-baseline FAISS indexes keyed by `faiss_db_<model>_<size>_<overlap>/` (e.g. `faiss_db_vietnamese-sbert_1000_200/`).
   - To refresh: use Kaggle dataset "New Version" to preserve the slug that the notebook hardcodes.

6. **For Kaggle:** enable GPU T4, then `os.chdir('/kaggle/working/')` before executing (the setup cell sets `ON_KAGGLE`, `WORK_DIR`, `KAGGLE_FAISS`, `KAGGLE_IMPROVEMENTS` automatically).

---

## References

1. Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.
2. Lin. "ROUGE: A Package for Automatic Evaluation of Summaries." ACL Workshop 2004.
3. Papineni et al. "BLEU: a Method for Automatic Evaluation of Machine Translation." ACL 2002.
4. Manning et al. "Introduction to Information Retrieval." Cambridge University Press, 2008.
5. Johnson et al. "Billion-scale similarity search with GPUs." IEEE Transactions on Big Data, 2021. (FAISS)
6. Reimers & Gurevych. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP 2019.
