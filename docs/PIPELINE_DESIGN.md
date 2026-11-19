# Video–Image Retrieval Pipeline

> **Quy mô:** ~200.000 keyframes  
> **Đơn vị lập chỉ mục:** `keyframe_id` (ảnh tĩnh dùng `image_id`)  
> **Thiết kế:** Offline indexing và online retrieval tách biệt; OCR được chạy độc lập, không cần tạo lại caption hay visual embedding.

## 1. Kiến trúc tổng thể

```text
                 VIDEO / IMAGE DATASET
                          |
          Keyframes + mapping + metadata
           (video_id, keyframe_id, timestamp)
                          |
                 OFFLINE PROCESSING
                          |
       +------------------+-------------------+
       |                  |                   |
  Visual branch     Caption branch        OCR branch
  SigLIP/OpenCLIP   Caption generator      PaddleOCR
       |            (InternVL: đề xuất)        |
  Visual vectors         |                OCR text
       |              Captions                 |
       |                  |                   |
       |          MiniLM text encoder         |
       |                  |                   |
       |          Semantic vectors            |
       |                  |                   |
    Milvus             FAISS              OCR JSONL
  visual index       384D index                |
       |                  |                  BM25
       |                  |                   |
       +------------------+-------------------+
                          |
                   ONLINE RETRIEVAL
                          |
              Text query / Image query
                          |
            Query planner / router (LLM)
                          |
               Parallel retrieval
             /          |           \
       Visual ANN   Semantic ANN    BM25
        (Milvus)      (FAISS)    (caption + OCR)
             \          |           /
                 Candidate union
              + keyframe_id mapping
                          |
                   RRF fusion
                          |
             Deduplication / filtering
                          |
             Qwen VLM reranking (Top-N)
                          |
            Temporal grouping / matching
                          |
             Final ranked keyframes
              / video segments / images
```

**Lưu ý:** Planner LLM là thành phần tùy chọn cho truy vấn phức tạp; truy vấn đơn giản có thể đi thẳng tới các retriever. Với truy vấn chuỗi sự kiện, temporal matching có thể cần được thực hiện **trước** reranking hoặc phối hợp với reranking. BM25 có thể index riêng `caption_text` và `ocr_text`, sau đó hợp nhất kết quả.

## 2. Offline indexing pipeline

### A. Visual retrieval

```text
Keyframes / Images
    → SigLIP hoặc OpenCLIP (image encoder)
    → Visual embeddings (chiều vector theo checkpoint; ví dụ 768D)
    → Milvus vector collection
```

### B. Semantic retrieval từ caption

```text
Keyframes / Images
    → Caption generator (InternVL nếu chọn dùng)
    → captions.json / captions.jsonl
    → MiniLM text encoder (384D với checkpoint phù hợp)
    → FAISS semantic index
```

- Nếu caption đã tồn tại, **tái sử dụng caption hiện có**; không cần chạy lại bước sinh caption.
- SigLIP/OpenCLIP là **image–text embedding model**, không phải caption generator theo nghĩa thông thường. Cần ghi đúng tên model thực sự đã dùng để sinh caption.

### C. OCR retrieval (chạy bổ sung, độc lập)

```text
200.000 keyframes hiện có
    → PaddleOCR (text detection + recognition)
    → ocr_results.jsonl
    → Text normalization / filtering
    → BM25 OCR index
```

- OCR chạy **offline trên ảnh keyframe gốc**, không chạy trên caption JSON.
- Không cần tính lại SigLIP embeddings, caption hay MiniLM embeddings chỉ vì bổ sung OCR.
- Lưu các trạng thái `processed`, `no_text`, `failed` để hỗ trợ resume.

### D. Keyword retrieval từ caption

```text
captions.jsonl
    → Text normalization / tokenization
    → BM25 caption index
```

### E. Temporal metadata

```text
keyframe_id
    ↔ video_id / image_id
    ↔ timestamp_ms
    ↔ shot_id (nếu có)
    ↔ image_path
```

Temporal retrieval sử dụng metadata, thứ tự keyframes và logic ghép chuỗi sự kiện; **không mặc định cần một mô hình học sâu riêng**.

## 3. Online query pipeline

1. **Input:** Truy vấn văn bản hoặc ảnh.
2. **Query planner/router (tùy chọn):** Phân tích ý định, cụm từ cần tìm chính xác, điều kiện thời gian; bật các nhánh phù hợp.
3. **Parallel candidate retrieval:**
   - **Visual:** SigLIP/OpenCLIP query encoder → Milvus ANN search.
   - **Semantic:** MiniLM text encoder → FAISS ANN search trên caption embeddings.
   - **Keyword:** BM25 trên caption và OCR text.
4. **Candidate union:** Chuẩn hóa `keyframe_id`, lấy metadata, gộp kết quả trùng.
5. **Fusion:** Reciprocal Rank Fusion (RRF) để kết hợp ranking từ các nhánh.
6. **Reranking:** Qwen VLM reranker chấm lại shortlist Top-N khi cần; có thể bỏ qua để giảm latency.
7. **Temporal retrieval:** Filter/group theo `video_id`, `timestamp_ms`, `shot_id`; kiểm tra thứ tự sự kiện nếu query yêu cầu.
8. **Output:** Danh sách keyframes hoặc video segments được xếp hạng, kèm thumbnail, timestamp và video/image ID.

> **Image-query routing:** Query dạng ảnh có thể dùng trực tiếp visual branch; MiniLM/BM25 chỉ áp dụng khi có text query hoặc văn bản được tạo/trích xuất phù hợp.

## 4. Mô hình và công nghệ

| Thành phần | Model / Công nghệ | Vai trò | Trạng thái |
|---|---|---|---|
| Keyframe extraction & mapping | Pipeline hiện có | Lấy keyframes, ID, timestamp | Đã có |
| Caption generation | **InternVL** (mô hình đề xuất; cần xác nhận model đã chạy thực tế) | Tạo mô tả keyframe | Caption hiện có có thể tái sử dụng |
| Visual embedding | **SigLIP / OpenCLIP** | Mã hóa ảnh và text vào không gian visual–text | Theo pipeline hiện có |
| Visual index | **Milvus** | ANN search visual embeddings | Theo pipeline hiện có |
| Semantic text embedding | **MiniLM**, 384D nếu đúng checkpoint | Mã hóa caption và text query | Theo pipeline hiện có |
| Semantic index | **FAISS** | ANN search caption embeddings | Theo pipeline hiện có |
| OCR | **PaddleOCR** | Phát hiện và nhận dạng chữ trên keyframes | Bổ sung offline |
| Keyword retrieval | **BM25** | Truy vấn caption/OCR theo từ khóa | Bổ sung hoặc mở rộng |
| Fusion | **RRF** | Hợp nhất ranking giữa các nhánh | Online |
| Query planning | **Planner LLM** (tùy chọn) | Query decomposition / routing | Online |
| Multimodal reranking | **Qwen VLM** | Chấm lại top candidates | Online |
| Temporal retrieval | Timestamp/shot/video matching | Nhóm đoạn, ràng buộc thứ tự | Online |

**Cấu hình cần xác nhận:** Kích thước vector SigLIP/OpenCLIP phụ thuộc checkpoint, **không mặc định là 768D**. MiniLM là 384D chỉ với các checkpoint phù hợp. Text query phải được mã hóa bằng **đúng encoder tương ứng** với index mà nó truy vấn.

## 5. Quy ước dữ liệu giữa các nhánh

**Khóa liên kết thống nhất:** `keyframe_id`.

```json
{
  "keyframe_id": "video001_kf00042",
  "video_id": "video001",
  "timestamp_ms": 125000,
  "image_path": "keyframes/video001/kf00042.jpg",
  "caption_text": "A person standing near a restaurant",
  "ocr_text": "PHO HA NOI",
  "ocr_status": "processed"
}
```

**Các artifact nên tách riêng:**

```text
artifacts/
├── keyframe_manifest.jsonl       # ID, path, video_id, timestamp
├── captions.jsonl                # keyframe_id, caption_text, model_version
├── ocr_results.jsonl             # keyframe_id, ocr_text, boxes, scores, status
├── faiss_semantic/               # MiniLM caption index + ID mapping
├── milvus_visual/                # Milvus collection / release metadata
└── bm25/                         # Caption/OCR index + ID mapping
```

**Nguyên tắc:** OCR bổ sung một artifact và một retrieval branch mới. **Giữ nguyên các artifact caption, MiniLM/FAISS và SigLIP/Milvus đã hoàn thành**, trừ khi có thay đổi model, dữ liệu hoặc schema đòi hỏi tái lập chỉ mục.
