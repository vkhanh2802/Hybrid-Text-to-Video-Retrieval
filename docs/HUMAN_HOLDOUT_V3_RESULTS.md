# Holdout v3 do người dùng gán nhãn — đánh giá condition evidence

## Trạng thái

**Đã hoàn tất đánh giá sau khi freeze nhãn và bản dịch được người dùng duyệt.**
Không có lỗi chấm artifact. Hai chế độ có cùng metric chính; bộ condition
evidence không đạt điều kiện FPR phải giảm, vì vậy **giữ mặc định `off`**.

Không xem đây là bằng chứng toàn hệ thống đã có accuracy70,83%: số đó là
balanced accuracy trên các pair được gán nhãn, không phải retrieval accuracy.

## Kết quả trên holdout mới

### Phân loại event–frame

Các tỷ lệ dưới đây là **macro trung bình theo26 query**, mỗi query có cả hai
lớp. 2 unknown judgments bị loại khỏi metric, không đổi thành negative.

| Metric | Baseline | Semantic guard |
|---|---:|---:|
| Macro TPR (nhận positive) | 49,36% | 49,36% |
| Macro FPR (chấp nhận nhầm negative) | 7,69% | 7,69% |
| Macro balanced accuracy | 70,83% | 70,83% |
| TP / TN / FP / FN (số pair) | 17 / 79 / 7 / 25 | 17 / 79 / 7 / 25 |

Không lấy TP17/42 để đối chiếu trực tiếp với macro TPR49,36%: đó là micro
recall, có trọng số khác khi query có số positive khác nhau.

Condition states trên130 pair: **2 consistent, 0 contradicted, 128 unknown**.
Hai `consistent` chỉ nói thuộc tính được rule trích ra khớp caption, không
xác nhận mọi điều kiện trong event. Trên chính các pair đã gán nhãn, guard
không có contradiction để tác động, nên không đổi predictions.

### Truy hồi keyframe cho từng event (bounded diagnostic)

| Metric | Baseline | Semantic guard |
|---|---:|---:|
| Target video có trong pool | 9/26 | 9/26 |
| Known-frame Hit@1 | 4/26 (15,38%) | 4/26 (15,38%) |
| Known-frame Hit@5 | 4/26 (15,38%) | 4/26 (15,38%) |
| Known-frame Hit@20 | 6/26 (23,08%) | 6/26 (23,08%) |
| Đúng target video ở top1 | 6/26 | 6/26 |
| MRR@20 | 0,16215 | 0,16215 |

### Truy hồi chuỗi từ các event đã duyệt

| Metric | Baseline | Semantic guard |
|---|---:|---:|
| Target video có trong pool | 8/10 | 8/10 |
| Đúng target video ở top1 | 6/10 | 6/10 |
| Known-path Hit@1 | 1/10 | 1/10 |
| Known-path Hit@5 | 2/10 | 2/10 |
| Known-path Hit@20 | 2/10 | 2/10 |

Guard đã loại100 support trong các pool single-event và91 support trong các
pool sequence, nhưng **không đổi metric chính**. Hai tập có thể chứa lại cùng
event/frame; không cộng thành191 lỗi duy nhất, và không gọi đây là191 false
positives đã sửa vì phần lớn frame ngoài qrels chưa được adjudicate.

### Chẩn đoán giai đoạn thất bại, không retune

`tools/human_holdout_diagnostics.py` chỉ đọc raw score/frames đã lưu; không gọi
encoder/retrieval hoặc đổi ngưỡng. Nó thử giới hạn mask vào **known positives**
để xác định có đường đúng theo nhãn vượt gate hay không. Mask oracle này chỉ
để chẩn đoán, **không được đưa vào retrieval hoặc báo là dự đoán của hệ thống**.

| Event-frame diagnostic | Số query |
|---|---:|
| Video mẫu không vào pool | 17/26 |
| Video vào pool nhưng known positives không có support | 3/26 |
| Known-positive hit trong top20 | 6/26 |

| Sequence diagnostic | Số query |
|---|---:|
| Video mẫu không vào pool | 2/10 |
| Video vào pool nhưng không có full path known positives vượt support gate | 6/10 |
| Known-positive full-path hit trong top20 | 2/10 |

Trên tập này không thấy mất known-positive do budget frame hoặc gap timestamp.
Không gán mọi path khác qrels là sai về thị giác: nhãn chỉ bao phủ cửa sổ hẹp,
có thể còn frame hợp lệ chưa được người dùng xem.

128 unknown condition cases gồm:90 không trích được loại điều kiện hỗ trợ,
25 có query binding/ngôn ngữ mơ hồ theo rule,9 có caption binding/ngôn ngữ mơ
hồ,4 thiếu thuộc tính cần thiết trong đối chiếu. Đây là **coverage quá hẹp**,
không phải một verifier tổng quát cho các event người dùng viết.

### Quyết định và hướng tiếp theo

- Điều kiện số lượng26 pair-queries/10 sequences đạt, không có scoring failure;
  TPR/frame/path không regression nhưng **FPR không giảm**. Không đạt adoption gate.
- Giữ `condition_evidence=off` và baseline0,25/0,45. Không tune threshold,
  mở rộng synonyms hoặc sửa rule trên holdout đã quan sát này.
- Điểm nghẽn đo được là **candidate recall cho event riêng lẻ** và **support
  recall cho known-positive paths**. Một bộ chặn contradiction caption-only
  không thể cứu target bị loại khỏi pool hoặc frame đúng đang không vượt gate.
- Hướng thử tiếp theo cần một **development mới**: phân rã mô tả nhiều chi tiết
  thành truy vấn retrieval ngắn nhưng vẫn giữ nguyên điều kiện để kiểm chứng;
  đo recall ở seed/pool trước khi tăng độ mạnh của bộ lọc. Đây mới là đề xuất,
  chưa triển khai hoặc chọn profile bằng kết quả holdout này.
- Không cần người dùng gán nhãn lại bộ hiện tại hay chạy Kaggle cho lượt đã xong.
  Nếu kế hoạch tiếp theo thật sự cần model nặng/sinh caption mới, dừng và báo
  người dùng trước. Holdout v3 đã được quan sát, không tái sử dụng như untouched.

Đã có bước tiếp theo mở rộng retrieval/object scoring và gói VLM Kaggle, tách
development với human-v3 regression: [FULL_PIPELINE_VLM_BENCHMARK.md](FULL_PIPELINE_VLM_BENCHMARK.md).
Lượt VLM đó không làm bộ v3 trở thành holdout mới và chưa phải đánh giá LLM
planner khi API chưa cấu hình.

## Dữ liệu, phê duyệt và khóa nhãn

- Packet: `benchmark_results/human_holdout_v3_seed2026110203_v2/`.
- 12 video được lấy mẫu, 60/60 ảnh đọc được; người dùng giữ 11 nhóm và bỏ qua
  `drive_02` vì ảnh không rõ. Không thay video hoặc tạo nhãn bù.
- **26 event queries**, **130 judgments: 42 positive, 86 negative, 2 unknown**.
- 10 nhóm có 2–3 event; mỗi nhóm có ít nhất một path positive đúng thứ tự và
  giới hạn gap 60s theo kiểm tra trước retrieval.
- Nhãn do **người dùng quan sát ảnh**. Caption không được dùng làm ground truth.
  Bản dịch tiếng Anh được đề xuất từ câu tiếng Việt của người dùng, không xem
  ảnh/caption/scores để thêm chi tiết. Người dùng đã giải nghĩa 7 mục và phê
  duyệt 19 bản dịch còn lại trước freeze.
- Không sửa `human_annotations.json`, câu tiếng Việt, các judgment, thứ tự event
  hoặc nhóm bỏ qua. Bản tiếng Anh/phê duyệt được lưu thành các artifact mới.

File gốc người dùng nộp SHA-256:

```text
f6d884c23346e5e9bd3881b4663c0c2927a4848c73fdb70814729c38b244928d
```

Frozen labels: `frozen_human_labels_v1.json`, SHA-256:

```text
2449a46318bdb2925efea39146b70f43b152097eecbe29a8ae4ebe35a1f9af77
```

Provenance phê duyệt nằm trong `translation_confirmations_v1/`,
`final_translation_approval_v1.json`, `translation_final_v1/` và trường
`human_annotations.translation_approval` của frozen labels.

## Thiết kế so sánh đã giữ nguyên

Hai chế độ duy nhất:

1. **Baseline (`off`)**: visual >=0,25 **hoặc** semantic >=0,45.
2. **`semantic_guard`**: giữ gate baseline, chỉ loại support caption-only khi
   quy tắc `garment_attributes_v2` phát hiện mâu thuẫn tường minh. Visual vượt
   ngưỡng vẫn giữ support; unknown không bị chuyển thành negative.

Không thay rule/threshold, trọng số local0,6/0,4, round-robin pool12 video,
seed50 hits/stream/event, budget1500 frame/video, DP, gap60s hoặc penalty0,001.
Không chọn profile dựa trên kết quả holdout. Hash inference sources và config
được đối chiếu với protocol trước/sau run.

Runner mới `tools/human_holdout_evaluation.py` giữ provenance **human/holdout**;
không đổi schema thành nhãn AI/development để vượt guard công cụ cũ. Runner
không gọi hàm chọn ngưỡng và không thêm video đáp án vào pool.

### A. Phân loại các pair đã gán nhãn

- Chấm đúng 130 event–frame pairs bằng embeddings/caption hiện có và câu tiếng
  Anh đã duyệt. Hai chế độ dùng chung raw visual/semantic table.
- 2 unknown judgments không vào confusion matrix; báo macro theo query, số
  query có cả hai lớp, TP/TN/FP/FN và tỷ lệ condition `unknown`.
- Mỗi query có cùng trọng số macro; không coi nhiều frame lân cận là nhiều
  quan sát độc lập để suy ý nghĩa thống kê.

### B. Truy hồi keyframe cho 26 event

- Mỗi event dùng SigLIP/Milvus, BM25, MiniLM để lấy seed; chọn pool bằng đúng
  selector đã khóa, chấm các frame cục bộ và xếp hạng frame có support.
- Hai chế độ có cùng seed lists, pool, frames và raw scores; chỉ khác mask khi
  caption-only support bị guard loại.
- Known-frame Hit@1/5/20 chỉ nhận frame positive đã được gán nhãn. Frame chưa
  gán nhãn không tự trở thành false positive. Top-video là metric trung gian.
- **Đây là bounded temporal event-frame diagnostic**, không phải pipeline ảnh
  đơn production có RRF/object scoring/Qwen. Không lấy metric này gọi là
  accuracy end-to-end của ứng dụng.

### C. Truy hồi 10 chuỗi

- Dùng trực tiếp các event đã được người dùng duyệt, không gọi LLM planner.
- Tái sử dụng seed hits của từng event; chọn pool mới theo tổ hợp event bằng
  selector baseline, không dùng đáp án để chọn pool.
- Chấm một raw evidence table/video cho cả hai chế độ rồi chạy cùng ordered DP.
- Một known-path hit đòi hỏi mọi event chọn positive thuộc đúng video và đúng
  thứ tự. Chỉ tìm đúng video chưa đủ; `verified=false` vẫn được giữ.
- Pool tối đa12 video nên Hit@20 ở đây bao phủ mọi path tốt nhất/video trong
  pool; không có nghĩa xếp hạng 20 video từ toàn corpus mà không bị candidate cut.

## Điều kiện xem xét áp dụng

Theo protocol đã lưu trước lấy mẫu:

- >=8 event queries có cả positive và negative; >=4 chuỗi đã gán nhãn.
- Macro FPR phải **giảm**, macro TPR không giảm.
- Known-frame Hit@20 và known-path Hit@20 không regression.
- Không có lỗi chấm artifact bị che bằng việc bỏ query.

Ngay cả khi vượt mọi điều kiện, chỉ báo `review_required_not_promoted`, không
tự sửa config. Nếu không vượt, tiếp tục `off`; không tune lại trên holdout.

## Artifact và tái lập

Output run: `benchmark_results/human_holdout_v3_evaluation_20261102/`.

- `context.json`, `source_snapshot.json`: cấu hình, hashes, model/collection.
- `pair_raw_scores.json`, `pair_predictions.json`, `pair_summary.json`.
- `drive_*_event_*.json`, `event_summary.json`.
- `drive_*_sequence.json`, `sequence_summary.json`.
- `raw_evidence/`: bảng điểm dùng chung cho hai chế độ.
- `local_artifact_sha256.json`: mapping/features fingerprints của các pool.
- `evaluation.log`, `summary.json`; nếu lỗi có `failure.json`.
- `posthoc_diagnostics.json`: stage attribution từ cache, không có model mới.

```powershell
python -B -u -m tools.human_holdout_evaluation --packet benchmark_results/human_holdout_v3_seed2026110203_v2 --labels benchmark_results/human_holdout_v3_seed2026110203_v2/frozen_human_labels_v1.json --output-dir benchmark_results/human_holdout_v3_evaluation_NEW
```

Chọn output mới, không ghi đè run. Lượt tái lập dùng lại nhãn đã xem kết quả
không phải một holdout độc lập mới. Không đổi câu, nhãn hoặc ngưỡng giữa các lượt.

## Giới hạn và tài nguyên

- Đây là câu hỏi/nhãn mới sau khi khóa quy tắc, không phải corpus chưa từng
  được retrieval truy cập. Đã loại303 video có target/sample/local pool cũ;
  global seed-hit/preflight IDs không đồng nghĩa có ground truth.
- Các cửa sổ ảnh thưa không chứng minh chuyển động liên tục; relevance chưa
  được gán đầy đủ ngoài các ảnh đã xem. Có availability/annotation selection
  bias từ nhóm người dùng bỏ qua và nguy cơ nội dung gần trùng giữa video.
- Rule chỉ xử lý một tập hẹp màu/họa tiết trang phục bằng caption tiếng Anh;
  query nhiều người, vật thể hoặc hành động khác có thể không được áp dụng.
- Không VLM/Qwen, không tải model, không đọc video gốc hoặc tái tạo embeddings
  ảnh/caption corpus. Model text encoders chỉ dùng cache đã pin trên CPU offline.
- Không ghi Drive/Milvus, không rebuild index, không commit/push. Log staging
  trong unit tests là mock/thư mục tạm, không phải publish index thật.
- Kiểm tra sau run: hash file người dùng nộp/frozen labels/inference sources vẫn
  khớp; source hashes của runner khớp snapshot; **0 caption image-hash mismatch**
  trên các pair có checksum caption. Milvus collection ID470054560685313683,
  138.796 vectors, caption vectors130.996, không đổi index.

## Kiểm thử trước khi kết luận chất lượng

- Full suite cuối, bao gồm runner/approval/posthoc diagnostics: **314 tests,
  300 đạt, 14 skipped**.
- Lượt targeted approval/human holdout runner **15/15 đạt**; posthoc diagnostics
  **3/3 đạt**. Bao gồm pair identity/missing caption và sự khác nhau giữa đúng
  video/đúng toàn path.
- Bao phủ: cấm qrels khác bản human đã duyệt, cấm English chưa duyệt, không
  giả provenance AI/dev, chỉ explicit positive mới tạo known-frame hit, không
  thêm target vào pool, cùng raw scores giữa hai chế độ, bảo toàn visual
  support, không tự duyệt câu ngoài phạm vi receipt và không auto-promote.
- Unit tests dùng ảnh/điểm giả lập; không dùng kết quả holdout để sửa quy tắc.
- `git diff --check` đạt; chỉ có cảnh báo line endings trên các file có sẵn.
