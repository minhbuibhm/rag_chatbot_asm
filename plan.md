# Implementation Plan - RAG Law Chatbot Assignment

> Mục tiêu: Thực hiện từng phase, mỗi bước tạo output lưu vào `report/results/` để tái sử dụng cho report.

---

## Tổng quan kiến trúc

```
Phase 1 (Setup & Baseline) → Phase 2 (Data Exploration) → Phase 3 (Improvements) → Phase 4 (Demo) → Phase 5 (Report)
```

**Môi trường:** Colab (GPU) - notebook đã install xong dependencies (cell 0, 1).
**Data:** `Dataset/export_1/` (16,439 .txt files), `RES.xlsx` (~2,745 Q&A pairs).
**Baseline config:** vietnamese-sbert, chunk 2000/overlap 50, FAISS, k=5, Qwen1.5-1.8B.

---

## Phase 1: Setup & Chạy Baseline

### 1.1 Adapt notebook cho local/Colab
- **Việc cần làm:** Sửa cell 2 (data loading) để đọc từ đúng path (`Dataset/export_1/`, `RES.xlsx` thay vì Google Drive mount).
- **Code changes:**
  - Đọc RES.xlsx bằng `pd.read_excel()` thay vì CSV từ Google Sheets.
  - Cập nhật `DOCUMENTS_PATH` trỏ đúng folder.

### 1.2 Chạy baseline evaluation trên ≥100 Q&A pairs
- **Việc cần làm:** Sửa `df.head(30)` → `df.head(100)` (hoặc nhiều hơn) trong cell 5.
- **Chạy cell 3** (build FAISS) → **cell 4** (verify) → **cell 5** (evaluate).
- Nếu đã có `faiss_db/` saved, skip cell 3 và load trực tiếp từ cell 4.

### 1.3 Ghi lại baseline scores
- **Output cần lưu:**
  - `report/results/baseline_metrics.csv` — metrics từng câu hỏi
  - `report/results/baseline_summary.txt` — mean scores (Cosine Sim, Jaccard, Token Overlap, BLEU, ROUGE-L)
  - `report/results/baseline_config.md` — bảng cấu hình baseline

**→ Report output:** Bảng "Baseline Configuration" + Bảng "Baseline Evaluation Results"

---

## Phase 2: Data Exploration & Understanding (1 pt)

### 2.1 Phân tích Legal Documents Corpus
- **Việc cần làm:** Viết code thống kê trong notebook cell mới.
- **Thống kê cần thu thập:**
  - Tổng số documents: 16,439 (đã biết)
  - Phân bố độ dài documents (chars): min, max, mean, median, std
  - Histogram độ dài documents
  - Phân loại theo tên file prefix (TT, QĐ, NĐ, CP...) → nhóm theo loại văn bản
  - Top 20 từ khóa phổ biến (sau khi bỏ stopwords)
- **Output lưu:**
  - `report/results/corpus_stats.json` — các con số thống kê
  - `report/results/doc_length_distribution.png` — histogram
  - `report/results/doc_type_distribution.png` — bar chart phân loại văn bản
  - `report/results/top_keywords.png` — word frequency chart

### 2.2 Phân tích RES Dataset (Q&A pairs)
- **Việc cần làm:** Đọc RES.xlsx, phân tích.
- **Thống kê cần thu thập:**
  - Tổng số Q&A pairs
  - Độ dài câu hỏi: min, max, mean (chars + words)
  - Độ dài câu trả lời: min, max, mean (chars + words)
  - Phân bố độ dài câu hỏi và câu trả lời (histogram)
  - Mẫu 5-10 cặp Q&A đại diện
- **Output lưu:**
  - `report/results/res_stats.json` — thống kê Q&A
  - `report/results/qa_length_distribution.png` — histogram
  - `report/results/sample_qa_pairs.csv` — mẫu Q&A đại diện

### 2.3 Phân tích Chunks (sau khi split)
- **Thống kê:** Tổng chunks (176,506), phân bố độ dài chunks, chunks/document ratio.
- **Output:** `report/results/chunk_stats.json`, `report/results/chunk_length_distribution.png`

**→ Report output:** Section "Data Exploration" với tables + figures ở trên.

---

## Phase 3: Cải thiện RAG Pipeline (3 pts Improvement + 3 pts Method)

### Chiến lược cải thiện (theo thứ tự ưu tiên impact)

### 3A. Embedding Model (Impact: CAO)
- **Thí nghiệm:**
  | # | Model | Dim | Ghi chú |
  |---|---|---|---|
  | 0 | `keepitreal/vietnamese-sbert` (baseline) | 768 | Baseline |
  | 1 | `bkai-foundation-models/vietnamese-bi-encoder` | 768 | BKAI Vietnamese |
  | 2 | `intfloat/multilingual-e5-base` | 768 | Multilingual E5 |
  | 3 | `intfloat/multilingual-e5-large` | 1024 | Larger E5 |
  | 4 | `BAAI/bge-m3` | 1024 | SOTA multilingual |
- **Đánh giá:** Với mỗi model, build FAISS → chạy retrieval trên 50-100 questions → đo retrieval quality (có ground truth answer xuất hiện trong retrieved chunks không).
- **Output:** `report/results/embedding_comparison.csv`, `report/results/embedding_comparison.png`

### 3B. Chunking Strategy (Impact: TRUNG BÌNH)
- **Thí nghiệm:**
  | # | Chunk Size | Overlap | Method |
  |---|---|---|---|
  | 0 | 2000 | 50 | Recursive (baseline) |
  | 1 | 1000 | 200 | Recursive |
  | 2 | 500 | 100 | Recursive |
  | 3 | 1000 | 200 | Sentence-based |
- **Đánh giá:** Số chunks tạo ra, retrieval quality với cùng embedding model.
- **Output:** `report/results/chunking_comparison.csv`, `report/results/chunking_comparison.png`

### 3C. Retrieval Strategy (Impact: CAO)
- **Thí nghiệm:**
  1. **Top-k tuning:** k = 3, 5, 7, 10, 15
  2. **Hybrid retrieval:** BM25 (keyword) + FAISS (semantic) → weighted combination
  3. **Reranking:** Dùng cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2` hoặc tương đương) để rerank top-k results
  4. **Query expansion:** Dùng LLM rewrite query trước khi retrieve
- **Đánh giá:** Retrieval recall@k, final answer metrics.
- **Output:**
  - `report/results/topk_comparison.csv`
  - `report/results/hybrid_vs_vector.csv`
  - `report/results/reranking_impact.csv`

### 3D. LLM & Prompt Engineering (Impact: CAO)
- **LLM candidates (< 4B params):**
  | # | Model | Params | Ghi chú |
  |---|---|---|---|
  | 0 | `Qwen/Qwen1.5-1.8B` (baseline) | 1.8B | Baseline |
  | 1 | `Qwen/Qwen2.5-3B-Instruct` | 3B | Newer, instruct-tuned |
  | 2 | `microsoft/Phi-3.5-mini-instruct` | 3.8B | Strong reasoning |
  | 3 | `google/gemma-2-2b-it` | 2B | Good multilingual |
- **Prompt engineering:**
  - Baseline prompt → thêm system role "legal expert"
  - Thêm instruction "trả lời bằng tiếng Việt", "trích dẫn điều luật cụ thể"
  - Few-shot: thêm 1-2 ví dụ Q&A mẫu
  - Chain-of-thought: "hãy suy nghĩ từng bước"
- **Output:**
  - `report/results/llm_comparison.csv`
  - `report/results/prompt_comparison.csv`

### 3E. Final Improved System
- **Kết hợp best config** từ 3A-3D.
- **Chạy evaluation trên ≥100 Q&A pairs** với improved system.
- **Output:**
  - `report/results/improved_metrics.csv` — metrics từng câu hỏi
  - `report/results/improved_summary.txt` — mean scores
  - `report/results/baseline_vs_improved.csv` — bảng so sánh side-by-side
  - `report/results/baseline_vs_improved.png` — bar chart so sánh
  - `report/results/error_analysis.csv` — cases cải thiện tốt + cases chưa tốt
  - `report/results/improvement_config.md` — cấu hình improved system

**→ Report output:** Section "Method" + "Evaluation Results" với tất cả tables/figures trên.

---

## Phase 4: Demo (2 pts)

### 4.1 Chuẩn bị demo
- Chọn 5-10 câu hỏi mẫu đa dạng chủ đề (dân sự, lao động, hành chính...).
- Tạo notebook cell hoặc script riêng chạy clean.

### 4.2 Demo format
- Mỗi câu hỏi hiển thị: Question → Retrieved passages (top-3) → Generated answer.
- So sánh baseline answer vs improved answer cho cùng câu hỏi.

### 4.3 (Optional) Simple UI
- Gradio/Streamlit interface nếu có thời gian.

### Output lưu:
- `report/results/demo_questions.csv` — câu hỏi mẫu + answers
- `report/results/demo_comparison.csv` — baseline vs improved cho demo questions
- Screenshots nếu có UI

**→ Report output:** Section "Demonstration" với bảng so sánh demo.

---

## Phase 5: Report (5 pts)

### Cấu trúc Report đề xuất

```
1. Introduction
   - Problem statement
   - Objectives

2. Data Exploration & Understanding (1 pt)
   - 2.1 Legal Documents Corpus Analysis
     → dùng: corpus_stats.json, doc_length_distribution.png, doc_type_distribution.png
   - 2.2 RES Dataset Analysis
     → dùng: res_stats.json, qa_length_distribution.png, sample_qa_pairs.csv
   - 2.3 Chunking Analysis
     → dùng: chunk_stats.json, chunk_length_distribution.png

3. Baseline System (phần của Method)
   → dùng: baseline_config.md, baseline_summary.txt

4. Method - Improvements (3 pts)
   - 4.1 Embedding Model Selection
     → dùng: embedding_comparison.csv/png
   - 4.2 Chunking Strategy Optimization
     → dùng: chunking_comparison.csv/png
   - 4.3 Retrieval Improvements
     → dùng: topk_comparison.csv, hybrid_vs_vector.csv, reranking_impact.csv
   - 4.4 LLM Selection & Prompt Engineering
     → dùng: llm_comparison.csv, prompt_comparison.csv
   - 4.5 Final System Configuration
     → dùng: improvement_config.md

5. Evaluation Results
   → dùng: baseline_vs_improved.csv/png, error_analysis.csv

6. Demonstration
   → dùng: demo_comparison.csv

7. Conclusion & Future Work

8. References
```

### Format & Clarity (1 pt)
- Headings rõ ràng, figures có caption, tables có label.
- Hướng dẫn reproduce: liệt kê steps chạy notebook.
- References đầy đủ (4 papers từ requirements + thêm model papers).

---

## Mapping: Output files → Report sections

| File | Report Section |
|---|---|
| `baseline_config.md` | 3. Baseline |
| `baseline_summary.txt` | 3. Baseline |
| `baseline_metrics.csv` | 5. Evaluation |
| `corpus_stats.json` | 2.1 Corpus Analysis |
| `doc_length_distribution.png` | 2.1 Corpus Analysis |
| `doc_type_distribution.png` | 2.1 Corpus Analysis |
| `top_keywords.png` | 2.1 Corpus Analysis |
| `res_stats.json` | 2.2 RES Analysis |
| `qa_length_distribution.png` | 2.2 RES Analysis |
| `sample_qa_pairs.csv` | 2.2 RES Analysis |
| `chunk_stats.json` | 2.3 Chunking Analysis |
| `chunk_length_distribution.png` | 2.3 Chunking Analysis |
| `embedding_comparison.csv/png` | 4.1 Embedding |
| `chunking_comparison.csv/png` | 4.2 Chunking |
| `topk_comparison.csv` | 4.3 Retrieval |
| `hybrid_vs_vector.csv` | 4.3 Retrieval |
| `reranking_impact.csv` | 4.3 Retrieval |
| `llm_comparison.csv` | 4.4 LLM |
| `prompt_comparison.csv` | 4.4 Prompt |
| `baseline_vs_improved.csv/png` | 5. Evaluation |
| `improved_summary.txt` | 5. Evaluation |
| `error_analysis.csv` | 5. Evaluation |
| `improvement_config.md` | 4.5 Final Config |
| `demo_comparison.csv` | 6. Demonstration |

---

## Thứ tự thực hiện (recommended)

```
1. [Phase 1.1] Adapt notebook paths           ~30 min
2. [Phase 2]   Data exploration code           ~2 hrs  → lưu tất cả charts/stats
3. [Phase 1.2] Chạy baseline evaluation 100+   ~1-2 hrs (GPU time)
4. [Phase 1.3] Lưu baseline scores             ~10 min
5. [Phase 3A]  Test embedding models           ~3-5 hrs (mỗi model rebuild FAISS)
6. [Phase 3B]  Test chunking strategies        ~2-3 hrs
7. [Phase 3C]  Test retrieval strategies       ~2-3 hrs
8. [Phase 3D]  Test LLM + prompts             ~2-3 hrs
9. [Phase 3E]  Combine best → final eval      ~1-2 hrs
10. [Phase 4]  Prepare demo                    ~1 hr
11. [Phase 5]  Write report từ saved outputs   ~3-5 hrs
```

**Tổng ước tính:** ~15-25 hrs (phụ thuộc GPU speed).
**Tip:** Phase 3A-3D có thể làm song song nếu có nhiều GPU/Colab sessions.
