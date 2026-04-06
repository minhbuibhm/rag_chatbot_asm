# TODO - RAG Law Chatbot Assignment

> **Goal:** Cải thiện RAG pipeline so với baseline, chạy evaluation >= 100 pairs, viết report đầy đủ.
> **Grading:** Improvement 3pt + Demo 2pt + Report 5pt (Data 1 + Method 3 + Format 1)
> **Constraints:** LLM < 4B params | Min 100 Q&A pairs | 5 metrics bắt buộc

---

## Current Status

| Phase | Status | Note |
|-------|--------|------|
| Data exploration | Done | Cells 5-9 chạy xong, charts saved |
| FAISS build | Done | 176,504 chunks, ~1h50m on CPU |
| Baseline evaluation | **Done (30 pairs)** | Cần re-run với 100 pairs + fix `return_full_text` |
| Improvement experiments | **Not started** | `Improvements.ipynb` cần implement |
| Report | Draft (Section 2+3 done) | Baseline scores đã điền, cần kết quả experiments |
| Demo | Not started | Cần improved pipeline |

**Baseline scores (30 pairs):** Cosine=0.674, Jaccard=0.181, TokenOverlap=0.706, BLEU=0.097, ROUGE-L=0.078
**Known issue:** `return_full_text=True` đang inflate Token Overlap và Cosine Similarity.

---

## Execution Plan

### Step 1: Chạy baseline evaluation (prerequisite cho mọi thứ)
- [x] Chạy baseline eval trên 30 pairs → scores saved to `rag_baseline_metrics.csv`
- [x] Ghi baseline scores vào report Section 3.4
- [ ] Re-run với `df.head(100)` + fix `return_full_text=False` để có scores chính xác

### Step 2: Implement improvement experiments

**Tạo file mới `Improvements.ipynb`** — tách riêng khỏi Baseline.ipynb để dễ quản lý.

**Cấu trúc notebook:**
1. Setup cell: import, load `res.csv` với `df.head(100)`, load LLM (`Qwen1.5-1.8B`), define metric functions (copy từ Cell 13 của Baseline.ipynb)
2. Một cell riêng cho mỗi experiment
3. Cell cuối: aggregate results → bảng so sánh → lưu `report/results/comparison_table.csv`

---

**Experiment A — Embedding Model**
- [ ] Implement và chạy

Thử lần lượt, mỗi model: load embeddings → build FAISS → eval 100 pairs → lưu scores.
```
Candidates: intfloat/multilingual-e5-large, bkai-foundation-models/vietnamese-bi-encoder
```
⚠️ **Lưu ý:** `multilingual-e5-large` yêu cầu prefix `"query: "` / `"passage: "` khi embed — phải wrap trong subclass hoặc custom embed function. Nếu thiếu prefix, scores sẽ thấp hơn thực tế.
⚠️ Build FAISS tốn ~10-15 phút/model trên GPU T4. Dùng cache (`embeddings_cache_<modelname>.pkl`) để tránh rebuild khi rerun.

---

**Experiment B — Chunking Strategy**
- [ ] Implement và chạy

Dùng embedding model tốt nhất từ Experiment A. Thử 3 configs với cùng splitter (`RecursiveCharacterTextSplitter`):
```
(chunk_size=2000, overlap=50)   ← baseline, đã có faiss_db/
(chunk_size=1000, overlap=200)
(chunk_size=500,  overlap=100)
```
⚠️ Mỗi config phải build FAISS index mới (save sang `faiss_db_1000/`, `faiss_db_500/`). Không dùng lại index baseline vì embedding dimensions khác nếu đổi model.

---

**Experiment C — Retrieval Strategy**
- [ ] Implement và chạy

Dùng best config từ A+B. Hai nhóm:

*C1 — Top-k tuning:* chạy eval với k = 3, 5, 7, 10. Không cần rebuild FAISS, chỉ đổi tham số.

*C2 — BM25 Hybrid:* build BM25 index trên cùng tập chunks. Query → top-k BM25 + top-k vector → deduplicate → merge context → eval.
```python
from rank_bm25 import BM25Okapi
# Score = alpha * bm25_score + (1-alpha) * vector_score, thử alpha=0.3, 0.5
```
⚠️ BM25 cần tokenized corpus — tokenize bằng `text.split()` là đủ cho tiếng Việt ở mức này.

---

**Experiment D — Prompt Engineering**
- [ ] Implement và chạy

Thử 3 variants trên best config từ A+B+C. Chỉ thay phần prompt trong `rag_query()`:
```
D1 (baseline): "Trả lời ngắn gọn và chính xác nhất có thể:"
D2 (structured): "Trả lời theo format: [Căn cứ pháp lý]: ... [Nội dung]: ..."
D3 (few-shot): thêm 1 ví dụ Q&A vào prompt trước câu hỏi thực
```
⚠️ Few-shot ăn thêm token budget — kiểm tra `max_length` không bị truncate context.

---

**Output bắt buộc sau Step 2:**
- `report/results/exp_A_embedding.csv` — scores của từng embedding model
- `report/results/exp_B_chunking.csv` — scores của từng chunking config
- `report/results/exp_C_retrieval.csv` — scores của top-k và hybrid
- `report/results/exp_D_prompt.csv` — scores của từng prompt variant
- `report/results/comparison_table.csv` — bảng tổng hợp baseline vs best improved

### Step 3: Chọn best config, chạy final evaluation
- [ ] Combine best components từ Step 2
- [ ] Chạy full eval >= 100 pairs, tạo bảng so sánh baseline vs improved

### Step 4: Hoàn thiện report & demo
- [ ] Điền kết quả vào report Sections 3.4, 4, 5
- [ ] Viết Section 6 (Demo): chọn 5-10 câu hỏi mẫu, show input/output
- [ ] Viết Section 7 (Conclusion)
- [ ] Export PDF

---

## Reference

**Deliverables checklist:**
- [ ] Code chạy được (notebook với cả baseline + improvements)
- [ ] Demo (5-10 sample Q&A)
- [ ] Report PDF
- [ ] Bảng metrics so sánh baseline vs improved

**Files quan trọng:**
- `Baseline.ipynb` — notebook chính, cần thêm cells cho experiments
- `report.md` — report draft, Section 2 done, còn lại cần kết quả
- `report/results/` — 10 chart/json files từ data exploration (done)
- `faiss_db/` — FAISS index baseline (cần verify trên Kaggle)
- `res.csv` — 2,745 Q&A pairs

**Experiment candidates (chi tiết):**
- Embedding: `keepitreal/vietnamese-sbert` (baseline) vs `intfloat/multilingual-e5-large`, `bkai-foundation-models/vietnamese-bi-encoder`, `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base`
- Chunking: 2000/50 (baseline) vs 500/100, 1000/200
- Top-k: 5 (baseline) vs 3, 7, 10
- Hybrid: vector-only (baseline) vs BM25+vector (`rank_bm25`)
- LLM: `Qwen1.5-1.8B` (baseline) vs `Qwen2.5-3B`, `Phi-3.5-mini`
- Prompt: simple instruction (baseline) vs few-shot, chain-of-thought, structured output
