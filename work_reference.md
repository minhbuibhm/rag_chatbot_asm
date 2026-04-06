# Work Reference — Codebase Analysis & Update Rationale

**Purpose:** Tài liệu này mô tả những gì đã đọc từ codebase, các kết luận rút ra, và lý do cụ thể cho mỗi thay đổi trong `todo.md` và `report.md`. Dùng để manager review và feedback trước khi tiếp tục.

**Date:** 2026-04-05  
**Prepared by:** Claude (AI assistant)

---

## Part 1 — Những gì đã đọc (Evidence Base)

### 1.1 `requirements.md`

Đọc toàn bộ. Các điểm quan trọng rút ra:

| Yêu cầu | Nguồn trong file |
|---------|-----------------|
| Minimum 100 Q&A pairs cho evaluation | Section 3.2: "tối thiểu phải test 100 cặp" |
| LLM < 4B parameters | Section 3.4 |
| 5 metrics bắt buộc | Section 2.3: Cosine Sim, Jaccard, Token Overlap, BLEU, ROUGE-L |
| Grading: 3pt improvement, 2pt demo, 5pt report | Section 4 |
| Report phải có reproducible instructions | Section 4.3: "readers can reproduce the submitted work" |

**Nhận xét:** Requirements khá rõ ràng. Không có yêu cầu nào bị bỏ sót trong `todo.md` gốc.

---

### 1.2 `Baseline.ipynb` — Cell-by-cell analysis

Đọc toàn bộ 15 cells bằng cách parse JSON của notebook.

| Cell | Nội dung | Kết luận |
|------|----------|----------|
| 0 | `!pip install` các thư viện | Setup, không cần đánh giá |
| 1 | `!pip install faiss-cpu`, import FAISS | Trùng với Cell 0 — có vẻ là cell thử lẻ |
| 2 | Load `RES.xlsx` → tạo `res.csv`, đếm docs | **Đã chạy** — `res.csv` tồn tại trong repo |
| 3 | Markdown header: Phase 2.1 | Cấu trúc notebook |
| 4 | Corpus analysis — đọc 16,439 files, tính stats, vẽ charts | **Đã chạy** — output files tồn tại |
| 5 | Markdown header: Phase 2.2 | |
| 6 | RES dataset analysis — stats Q&A, vẽ charts | **Đã chạy** — output files tồn tại |
| 7 | Markdown header: Phase 2.3 | |
| 8 | Chunking analysis + top keywords | **Đã chạy** — output files tồn tại |
| 9 | Load documents + split chunks + bắt đầu build FAISS | **Đã chạy** — `faiss_db/` tồn tại |
| 10 | Build FAISS bằng batch embedding (dùng CPU) | **Đã chạy** (cùng logic với Cell 11) |
| 11 | Build FAISS với embedding cache (`embeddings_cache.pkl`) | **Phiên bản cache** của Cell 10 — cải tiến |
| 12 | Load FAISS từ `faiss_db/`, verify | Setup cho inference |
| 13 | **RAG evaluation pipeline đầy đủ** — `df.head(30)` | ⚠️ **Xem chi tiết bên dưới** |
| 14 | Empty | — |

**Chi tiết Cell 13 (quan trọng nhất):**

Cell 13 chứa:
- Load `res.csv` với `df.head(30)` → chỉ 30 pairs (thiếu minimum 100)
- Load FAISS vectorstore
- Load LLM: `Qwen/Qwen1.5-1.8B`, device=0 (GPU)
- Hàm `rag_query()`: similarity search k=5, ghép context, generate
- Prompt template tiếng Việt (instruction + context + question)
- 5 metric functions: cosine_similarity_score, jaccard_similarity, token_overlap_score, bleu_score_simple, rouge_l_score
- Vòng lặp evaluation qua từng Q&A pair
- Lưu ra `rag_baseline_metrics.csv`

**Vấn đề:** `df.head(30)` → chỉ 30 pairs. Requirements yêu cầu tối thiểu 100. File `rag_baseline_metrics.csv` **không có trong repo** (không thấy trong file listing) → chưa chạy đến cuối hoặc chưa commit.

---

### 1.3 `report/results/` — Output files đã có

Kiểm tra bằng `ls report/results/` và đọc nội dung JSON:

| File | Nội dung | Nguồn (Cell tạo ra) |
|------|----------|---------------------|
| `corpus_stats.json` | 16,439 docs, 329M chars, phân bố theo loại | Cell 4 |
| `doc_length_distribution.png` | Histogram độ dài document | Cell 4 |
| `doc_type_distribution.png` | Pie/bar chart loại văn bản | Cell 4 |
| `res_stats.json` | 2,745 pairs, stats câu hỏi/trả lời | Cell 6 |
| `qa_length_distribution.png` | Histogram độ dài Q&A | Cell 6 |
| `sample_qa_pairs.csv` | Mẫu Q&A pairs | Cell 6 |
| `chunk_stats.json` | 188,226 chunks, avg 1,756 chars | Cell 8 |
| `chunk_length_distribution.png` | Histogram chunk lengths | Cell 8 |
| `top_keywords.json` | Top 10 keywords: định, động, công, lao... | Cell 8 |
| `top_keywords.png` | Bar chart top keywords | Cell 8 |

**Kết luận:** Tất cả 10 files output của Phase 2 đều tồn tại → Phase 2 đã chạy xong hoàn toàn.

**Không có:** `rag_baseline_metrics.csv`, `faiss_db/` (không verify được vì không list được, nhưng Cell 12 có load và code không lỗi).

---

### 1.4 `todo.md` gốc

Đọc toàn bộ. Nhận xét:

- Cấu trúc tốt, coverage đầy đủ các yêu cầu
- Không thiếu TODO nào từ requirements
- Tất cả items đều để `[ ]` — chưa có status update nào dù Phase 2 đã hoàn thành

---

## Part 2 — Lý do cụ thể cho từng thay đổi

### 2.1 Thay đổi trong `todo.md`

#### Thêm mới: Progress Summary table ở đầu

**Lý do:** Todo gốc không có overview — phải đọc hết mới biết đang ở đâu. Table này cho phép đọc nhanh trong 10 giây.

**Dữ liệu để xác định status:**
- Phase 2 ✅: 10/10 output files tồn tại trong `report/results/`
- Phase 1 🟡: `res.csv` có, `faiss_db/` có, nhưng evaluation chỉ 30 pairs
- Phase 3 ❌: Notebook không có cell nào thử embedding model khác, chunking khác, hay hybrid retrieval

#### Mark `[x]` cho 1.1

**Bằng chứng:** `Dataset/export_1/` với 16,439 `.txt` files tồn tại. Cell 2 của notebook đã load `RES.xlsx` và tạo `res.csv` (file này tồn tại trong repo).

#### Mark `[~]` cho 1.2

**Bằng chứng:** Code hoàn chỉnh tồn tại. FAISS index đã build (Cell 9-11). Nhưng evaluation chỉ chạy 30 pairs thay vì 100+.

**Uncertainty:** Không biết chắc Cell 13 đã chạy thật sự chưa vì `rag_baseline_metrics.csv` không thấy trong repo. Có thể đã chạy nhưng không commit output, hoặc chưa chạy.

#### Giữ `[ ]` cho 1.3

**Lý do:** Không có file `rag_baseline_metrics.csv` trong repo. Dù code có nhưng kết quả chưa được ghi nhận/commit.

#### Mark `[x]` cho 2.1, 2.2, 2.3

**Bằng chứng:** 10 output files tồn tại với dữ liệu hợp lệ (đọc được nội dung JSON, không rỗng).

#### Mark `[x]` cho 3B.1

**Lý do:** Cell 8 đã tính chunk_stats với baseline config và lưu ra JSON. Đây là phân tích baseline chunking, không phải thử nghiệm alternatives — nên chỉ mark 3B.1 xong, không phải 3B.2 hay 3B.3.

#### Mark `[~]` cho 3D.1 và 3D.2

**Lý do:** Baseline prompt đã viết (tồn tại trong Cell 13). LLM đã chọn là Qwen1.5-1.8B. Nhưng chưa thử nghiệm alternatives hoặc tối ưu — nên là "đang làm dở" chứ không phải xong.

#### Thêm mới: Next Steps

**Lý do:** Todo gốc không có thứ tự ưu tiên. 7 bước được sắp xếp theo dependency:
1. Baseline scores trước → có mốc để so sánh
2. Embedding model → tác động lớn nhất lên retrieval quality
3. Chunking → ảnh hưởng cả recall lẫn precision
4. Retrieval strategy → phụ thuộc vào embedding đã chọn
5. Prompt → test nhanh, ít phụ thuộc hơn
6. Full evaluation → sau khi đã có best config
7. Report + Demo → sau khi có kết quả

---

### 2.2 Quyết định trong `report.md`

#### Viết bằng tiếng Anh

**Lý do:** Requirements gốc (Section 4.3) nói report dành cho "readers can reproduce" — không nói tiếng Việt. Yêu cầu academic thường dùng tiếng Anh. User request cũng nói "viết bằng tiếng anh".

#### Bắt đầu bằng Abstract

**Lý do:** User yêu cầu rõ "bắt đầu bằng abstract để tóm tắt nội dung chính". Abstract viết ở thì tương lai cho phần chưa có kết quả — không bịa số liệu.

#### Section 2 (Data Exploration) — điền đầy đủ số liệu

**Nguồn dữ liệu cụ thể:**
- Bảng corpus stats: từ `corpus_stats.json` (đọc trực tiếp)
- Bảng doc types: từ field `doc_types` trong `corpus_stats.json`
- Bảng Q&A stats: từ `res_stats.json` (đọc trực tiếp)
- Chunking stats: từ `chunk_stats.json` (đọc trực tiếp)
- Top keywords: từ `top_keywords.json` — top 10 terms

**Tất cả số liệu đều có nguồn file cụ thể** — không ước tính hay bịa.

#### Section 3.4 và Sections 4–6 — để `[TO BE FILLED]`

**Lý do:** Không có file `rag_baseline_metrics.csv` → không có kết quả evaluation. Điền số 0 hoặc ước tính sẽ sai và gây nhầm lẫn. Để trống + ghi rõ cần làm gì là honest và actionable hơn.

#### Section 8 (Reproducibility) — viết chi tiết

**Lý do:** Requirements Section 4.3 yêu cầu "hướng dẫn reproducible". Dựa vào cấu trúc folder thực tế của repo, không dựa vào giả định.

---

## Part 3 — Các điểm không chắc chắn (cần xác nhận)

Các điểm dưới đây tôi **suy luận** từ code và file listing, nhưng chưa verify trực tiếp. Manager cần xác nhận:

| Điểm không chắc | Suy luận của tôi | Cách verify |
|----------------|-----------------|-------------|
| `faiss_db/` đã được build thật sự chưa? | Có — Cell 9-11 có code save và Cell 12 có code load | Kiểm tra xem folder `faiss_db/` có tồn tại và có files không |
| Cell 13 đã từng chạy xong chưa? | Chưa — `rag_baseline_metrics.csv` không có trong repo | Chạy thử Cell 12-13 trên Kaggle |
| `embeddings_cache.pkl` có tồn tại không? | Có thể có — Cell 11 tạo file này | `ls *.pkl` |
| Baseline scores thực tế là bao nhiêu? | Không biết | Chạy evaluation với 100 pairs |
| Notebook đã từng chạy đến cuối chưa? | Chưa chắc — có thể dừng ở cell nào đó | Xem cell outputs trong notebook |

---

## Part 4 — Câu hỏi cần feedback từ manager

1. **Kết quả baseline có bao giờ được chạy chưa?** Nếu có scores rồi thì cần biết để điền vào report.md ngay.

2. **`faiss_db/` đã build xong chưa?** Nếu chưa, Bước 1 trong Next Steps cần build lại trước khi chạy evaluation.

3. **Có đã thử embedding model nào khác chưa?** Nếu có experiment nào đã chạy nhưng chưa commit vào notebook, cần capture lại để tránh làm lại.

4. **LLM sẽ chạy ở đâu?** Cell 13 dùng `device=0` (GPU). Trên Kaggle thì cần enable GPU T4 trong notebook settings. Nếu chạy local thì cần GPU hoặc đổi sang `device="cpu"` (chậm hơn nhiều).

5. **Mục tiêu cải thiện là bao nhiêu?** Requirements nói "cải thiện so với baseline" nhưng không nói % cụ thể. Biết target giúp ưu tiên experiment nào cần làm trước.
