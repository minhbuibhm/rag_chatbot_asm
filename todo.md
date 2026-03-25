# TODO - RAG Assignment: Law Chatbot

> Tracking file cho toàn bộ assignment. Cập nhật status khi hoàn thành từng task.
> Status: `[ ]` = chưa làm | `[~]` = đang làm | `[x]` = xong

---

## Phase 1: Setup & Baseline (Ưu tiên cao)

- [ ] **1.1** Lấy dữ liệu từ Google Drive (legal documents + res.csv)
- [ ] **1.2** Copy baseline Colab notebook, chạy thành công lần đầu
- [ ] **1.3** Ghi lại baseline scores (Cosine Sim, Jaccard, Token Overlap, BLEU, ROUGE-L) làm mốc so sánh

---

## Phase 2: Data Exploration & Understanding (1 pt Report)

- [ ] **2.1** Phân tích legal documents corpus
  - [ ] Thống kê số lượng files, phân bố theo lĩnh vực (dân sự, lao động, hành chính...)
  - [ ] Phân tích độ dài trung bình, min, max của documents
  - [ ] Tìm hiểu thuật ngữ pháp luật phổ biến (word frequency)
- [ ] **2.2** Phân tích RES dataset (res.csv)
  - [ ] Số lượng Q&A pairs, phân bố theo chủ đề
  - [ ] Độ dài trung bình câu hỏi / câu trả lời
  - [ ] Đánh giá diversity và difficulty của questions
- [ ] **2.3** Visualize các phân tích trên (charts/tables cho report)

---

## Phase 3: Cải thiện RAG Pipeline (3 pts Improvement + 3 pts Method)

### 3A. Embedding Model

- [ ] **3A.1** Nghiên cứu các embedding models phù hợp tiếng Việt
  - Candidates: `bkai-foundation-models/vietnamese-bi-encoder`, `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base`, `intfloat/multilingual-e5-large`, ...
- [ ] **3A.2** So sánh retrieval quality giữa các embedding models
- [ ] **3A.3** Chọn embedding model tốt nhất, ghi lại lý do

### 3B. Chunking Strategy

- [ ] **3B.1** Thử nghiệm các chunking strategies khác nhau
  - Baseline: 2000 chars, overlap 50
  - Thử: chunk nhỏ hơn (500-1000), overlap lớn hơn (100-200)
  - Thử: sentence-based chunking, recursive chunking
- [ ] **3B.2** Đánh giá ảnh hưởng chunking lên retrieval quality

### 3C. Retrieval Strategy

- [ ] **3C.1** Tối ưu top-k (thử k = 3, 5, 7, 10)
- [ ] **3C.2** Thử hybrid retrieval (BM25 + vector search)
- [ ] **3C.3** Thử reranking retrieved passages (cross-encoder)
- [ ] **3C.4** Thử query expansion / transformation

### 3D. LLM & Prompt Engineering

- [ ] **3D.1** Chọn LLM < 4B params (candidates: `Qwen2.5-3B`, `Phi-3.5-mini`, `Vistral-7B-Chat` nếu fit...)
- [ ] **3D.2** Thiết kế prompt template tối ưu cho legal Q&A
- [ ] **3D.3** Thử các cách tích hợp context (stuff, map-reduce, refine)

### 3E. Evaluation & So sánh

- [ ] **3E.1** Chạy evaluation trên tối thiểu 100 Q&A pairs
- [ ] **3E.2** Tạo bảng so sánh metrics: baseline vs improved system
- [ ] **3E.3** Phân tích cases cải thiện tốt và cases chưa tốt (error analysis)

---

## Phase 4: Demo (2 pts)

- [ ] **4.1** Chọn 5-10 câu hỏi mẫu đại diện cho các chủ đề khác nhau
- [ ] **4.2** Tạo demo script/notebook chạy clean, không lỗi
- [ ] **4.3** Đảm bảo output readable, relevant, correct
- [ ] **4.4** (Optional) Tạo simple UI (Gradio/Streamlit) cho demo

---

## Phase 5: Report (5 pts)

- [ ] **5.1** Viết phần Data Exploration & Understanding (1 pt)
- [ ] **5.2** Viết phần Method - mô tả chi tiết kỹ thuật cải thiện (3 pts)
  - [ ] Embedding model selection & rationale
  - [ ] Chunking strategy
  - [ ] Retrieval improvements
  - [ ] Prompt engineering
  - [ ] Bảng evaluation results so sánh baseline
- [ ] **5.3** Format & Clarity (1 pt)
  - [ ] Cấu trúc rõ ràng, headings, figures/tables
  - [ ] Hướng dẫn reproducible (cách chạy lại code)
  - [ ] References đầy đủ

---

## Deliverables Checklist

- [ ] Code chạy được (Colab notebook hoặc Python scripts)
- [ ] Demo notebook/script
- [ ] Report (PDF)
- [ ] Bảng metrics so sánh baseline vs improved

---

## Notes

- LLM phải < 4B parameters
- Tối thiểu test 100 cặp Q&A, báo cáo số lượng đã dùng
- Focus chính: retrieval pipeline, không chỉ đổi LLM
- Ghi lại mọi experiment results để dùng cho report
