# Retrieval Augmented Generation Assignment - Building a Simple Law Chatbot

**Author:** Tho Quan
**Version:** 1.0, November 9, 2025

---

## 1. Problem Statement

Xây dựng hệ thống hỏi đáp pháp luật (law question-answering system) sử dụng kỹ thuật RAG để cải thiện so với baseline được cung cấp. Tập trung vào:

- Nâng cao **retrieval component** và tích hợp thông tin truy xuất vào quá trình sinh câu trả lời.
- Lựa chọn và tinh chỉnh **embeddings** cho semantic search tốt hơn.
- Thiết kế và tối ưu **retrieval methods** (vector-based, hybrid retrieval).
- Phương pháp chọn, xếp hạng, lọc retrieved passages để cải thiện answer relevance.

**Mục tiêu chính:** Cải thiện các metric đo lường so với baseline, chứng minh hiệu quả của kỹ thuật RAG.

---

## 2. RAG Overview và Evaluation Metrics

### 2.1. Kiến trúc RAG

```
Documents → Chunked Texts → Generate Embeddings → Vector DB
                                                        ↓
Prompt → Prompt Embedding → Most relevant text passages (context)
                                                        ↓
                                    Prompt + Context → LLM → Result
```

### 2.2. Các thành phần chính

- **Document Retrieval:** Tìm passages liên quan nhất từ legal corpus.
- **Context Integration:** Cấu trúc và chọn lọc retrieved passages để đóng góp hiệu quả vào answer generation.
- **Evaluation:** Đo lường mức cải thiện so với baseline bằng các objective metrics.

### 2.3. Evaluation Metrics

#### Cosine Similarity
Đo similarity giữa vector embeddings của generated answer và ground truth:
```
CosineSim(ŷ, y) = (v_ŷ · v_y) / (‖v_ŷ‖ · ‖v_y‖)
```

#### Token Overlap / Jaccard Similarity
Đo word-level overlap giữa generated answer và reference answer:
```
Jaccard(ŷ, y) = |W_ŷ ∩ W_y| / |W_ŷ ∪ W_y|
TokenOverlap(ŷ, y) = |W_ŷ ∩ W_y| / |W_y|
```

#### BLEU
Đo n-gram precision của generated answer so với reference answer:
```
BLEU(ŷ, y) = BP · |W_ŷ ∩ W_y| / |W_ŷ|
BP = 1 nếu |ŷ| ≥ |y|, exp(1 − |y|/|ŷ|) nếu |ŷ| < |y|
```

#### ROUGE-L
Đo longest common subsequence (LCS) giữa ŷ và y:
```
P = LCS(ŷ, y) / |ŷ|
R = LCS(ŷ, y) / |y|
ROUGE-L = 2 · P · R / (P + R)
```

---

## 3. Baseline

### 3.1. Cấu hình Baseline

| Thành phần | Chi tiết |
|---|---|
| **Chunking** | ~2000 characters/chunk, overlap 50 characters |
| **Embedding Model** | `keepitreal/vietnamese-sbert` |
| **Vector DB** | FAISS (local) |
| **Top-k retrieval** | k = 5 |
| **LLM** | `Qwen/Qwen1.5-1.8B` |
| **Framework** | LangChain |

### 3.2. Documents (Dữ liệu)

- **Legal Documents:** ~16,440 text files, bao gồm nhiều lĩnh vực pháp luật (dân sự, lao động, hành chính) từ nhiều cấp chính quyền.
- **RES Dataset:** ~2,745 cặp question-answer để đánh giá hệ thống RAG.
  - Có thể dùng subset nhỏ hơn, nhưng **tối thiểu phải test 100 cặp** và báo cáo số lượng đã dùng.

### 3.3. Code Snippets Baseline

#### Load RES Dataset
```python
import pandas as pd

df = pd.read_csv("res.csv")
df = df.head(100)  # Tối thiểu 100 cặp Q&A
print(f"Total number of questions: {len(df)}")
```

#### Embedding Model Initialization
```python
from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL = "keepitreal/vietnamese-sbert"
embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cuda"}
)
```

#### LLM Initialization
```python
from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen1.5-1.8B",
    task="text-generation",
    device=0,
    model_kwargs={"temperature": 0, "max_length": 1024}
)
```

### 3.4. Ràng buộc

- Tự do chọn **embedding model** bất kỳ.
- Tự do chọn **LLM dưới 4 billion parameters** phù hợp tài nguyên.
- Mục tiêu chính: cải thiện retrieval và integration pipeline của RAG.

### 3.5. Chạy Baseline

1. Copy source code từ Colab notebook vào Google Drive cá nhân.
2. Add thư mục RAG Assignment folder vào Google Drive dưới dạng shortcut.

**Source:** Google Drive folder (documents) + Colab source code (baseline).

---

## 4. Grading (Tổng 10 điểm)

### 4.1. Improvements compared to baseline - 3 pts

Đánh giá mức cải thiện so với baseline trên ít nhất 1 trong các metric:
- Cosine similarity
- Token-level overlap (Token Overlap, Jaccard Similarity)
- BLEU hoặc ROUGE-L

### 4.2. Demonstration - 2 pts

Demo hệ thống RAG trên một tập nhỏ câu hỏi mẫu:
- Clarity và readability của responses.
- Relevance và correctness của answers.
- Dễ chạy demo (scripts/notebooks chạy không lỗi lớn).

### 4.3. Report - 5 pts

| Tiêu chí | Điểm |
|---|---|
| **Data Exploration & Understanding** | 1 pt |
| **Method to improve RAG vs baseline** | 3 pts |
| **Format & Clarity** | 1 pt |

#### Data Exploration & Understanding (1 pt)
- Khám phá dataset, phân tích đặc điểm (số lượng questions, diversity, difficulty), hiểu cấu trúc.

#### Method to improve RAG vs baseline (3 pts)
- Mô tả kỹ thuật cải thiện retrieval hoặc generation:
  - Lựa chọn/tinh chỉnh embeddings và retrievers.
  - Prompt engineering cho LLM.
  - Tích hợp retrieved context với generation step.
  - Các cải tiến khác có hỗ trợ bởi quantitative metrics.
- **Bắt buộc** trình bày evaluation results cho thấy cải thiện so với baseline.

#### Format & Clarity (1 pt)
- Report có cấu trúc rõ ràng, headings, figures/tables nếu cần, hướng dẫn reproducible.

---

## 5. References

1. Lewis et al. "Retrieval-augmented generation for knowledge-intensive NLP tasks." NIPS 2020.
2. Lin. "ROUGE: A package for automatic evaluation of summaries." 2004.
3. Manning et al. "Introduction to Information Retrieval." Cambridge University Press, 2008.
4. Papineni et al. "BLEU: a method for automatic evaluation of machine translation." ACL 2002.

---

## Tóm tắt yêu cầu kỹ thuật

- **Input:** ~16,440 legal documents (text files) + ~2,745 Q&A pairs (res.csv)
- **Output:** Hệ thống RAG cải thiện so với baseline
- **Constraints:**
  - LLM < 4B parameters
  - Tối thiểu test 100 cặp Q&A
  - Embedding model tùy chọn
- **Deliverables:**
  1. Code chạy được (Colab/script)
  2. Demo trên câu hỏi mẫu
  3. Report phân tích chi tiết
- **Metrics cần cải thiện:** Cosine Similarity, Jaccard/Token Overlap, BLEU, ROUGE-L
