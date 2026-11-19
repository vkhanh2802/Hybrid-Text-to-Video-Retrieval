# Outline nâng cấp chất lượng tìm kiếm KIS

> 04/11/2026: [đã đo cross-encoder CPU thật](TEXT_RERANKER_CPU_RESULTS_20261104.md)
> trên cached development và human regression; gain/loss mixed, giữ mode off.

> 04/11/2026: [OCR retrieval độc lập đã tích hợp code/API](OCR_RETRIEVAL.md),
> mặc định off. Pilot Kaggle CPU 12 ảnh/12 video đã đóng gói và kiểm tra hashes;
> chưa chạy recognition thật, chưa có OCR index/accuracy. ASR vẫn chưa triển khai.

> Cập nhật triển khai 03/11/2026: event retrieval/DP và các benchmark đã có;
> [reranker nhẹ + API diagnostics](TEXT_RERANKER.md) nay đã được tích hợp code,
> mặc định off và chưa đo chất lượng bằng model thật. [Multi-path](TEMPORAL_MULTIPATH_RESULTS_20261103.md)
> đã thử nhưng không promote. OCR/ASR vẫn là hạng mục tiếp theo. Các mô tả hiện
> trạng bên dưới là snapshot lúc lập outline, không phải mọi module đều còn thiếu.

## 1. Mục tiêu và kết luận

Mục tiêu chính: đưa đáp án đúng lên **top 1–5**, đồng thời duy trì độ bao phủ
candidate và latency phù hợp. Với query nhiều hành động, đáp án phải là một
**đoạn có các sự kiện đúng thứ tự**, không chỉ một ảnh cùng chủ đề.

Ba module ưu tiên:

1. **Event retrieval + temporal ranking:** truy hồi từng sự kiện, ghép theo video/thời gian.
2. **Lightweight cross-encoder reranker:** sàng lọc sâu hơn trước VLM.
3. **OCR retrieval độc lập:** đọc và tìm nội dung bảng/chữ, không dựa vào caption suy đoán.

Trước khi thêm model, cần benchmark hiện trạng và kiểm tra độ phủ artifact.
Không tiếp tục sinh lại toàn corpus caption chỉ để hy vọng ranking sẽ tốt hơn.

Đây là thiết kế và kế hoạch thực nghiệm; các module đề xuất bên dưới chưa được
triển khai hoặc chứng minh tăng metric trên corpus hiện tại.

## 2. Pipeline đang chạy — điểm xuất phát

Theo `config/pipeline.json` và code hiện tại:

```text
Raw query
  -> Query API (mặc định Groq / openai/gpt-oss-20b)
  -> English stream queries + entities + events + temporal constraints
  -> SigLIP/Milvus + BM25 caption + MiniLM/FAISS caption
  -> Normalized-max giữa variants trong mỗi stream
  -> Weighted RRF giữa ba stream
  -> Object boost
  -> Qwen pointwise rerank top 20 khi đọc được ảnh
  -> Top 20 frame
```

Các vị trí liên quan:

| Thành phần | Vị trí hiện tại | Hệ quả thiết kế |
|---|---|---|
| Query có events/OCR flag | `kis/structure_query.py:30–40` | Đã có đầu vào để phân loại query |
| Retrieval chỉ lấy query variants/entities | `kis/solve_kis.py:134–169` | Chưa truy hồi riêng từng event, chưa dùng temporal constraints/OCR flag |
| Cross-stream RRF | `kis/ranking.py:26–54`, `kis/solve_kis.py:181–185` | Ưu tiên thứ hạng/đồng thuận; có thể hạ một visual-only hit mạnh |
| Hai nhánh text dùng cùng caption | `kis/text.py:27–49`, `kis/text.py:149–168` | Caption sai có thể được hai nhánh cùng ủng hộ |
| Object evidence theo nhãn | `kis/objects.py:150–182` | Chưa xác nhận thuộc tính thuộc đúng người, hành động hay thứ tự |
| VLM chỉ xem đầu danh sách | `kis/solve_kis.py:56–112` | Không cứu được đáp án ngoài cửa sổ rerank |
| Chấm VLM 0–10 | `kis/vlm_reranker.py:117–147` | Một số đơn lẻ khó thể hiện event nào đúng/sai/chưa rõ |
| Diversity selector có sẵn | `kis/ranking.py:201–272` | Chưa được truyền vào luồng web hiện tại |
| Web depth/budget | `kis/web_app.py:247–261` | Depth=100/stream, VLM budget=20, output=20 |

### Dữ liệu và giới hạn xác minh

- Scope cấu hình: L01–L03 và L07–L21.
- Trong filesystem có các nhóm visual features L22–L30, nhưng chúng nằm ngoài scope.
  Có file local không đồng nghĩa ứng dụng tìm kiếm những nhóm đó.
- Caption folders hiện thấy L01–L03 và L07–L20; chưa thấy folder caption L21.
- README ghi 138.796 visual vectors/535 video, 130.996 caption vectors/506 video.
  Đây là số ghi trong README, không phải kết quả đếm/xác minh live trong lần lập outline.
- Docker daemon chưa chạy khi kiểm tra, nên chưa xác minh được Milvus live có đủ dữ liệu mới.
- Những thay đổi query API/UI chưa commit đã tồn tại trong working tree; outline không thay đổi chúng.

Không sử dụng số đo L19/L20 từ các phiên bản trước như metric của hệ thống hiện tại:
corpus, caption, fusion, scope và code đã thay đổi.

## 3. Phân loại nguyên nhân trước khi tối ưu

| Quan sát trên query có ground truth | Nguyên nhân cần điều tra | Hướng sửa |
|---|---|---|
| Frame/đoạn đúng không tồn tại trong keyframes | Sampling hoặc thiếu corpus | Bổ sung dữ liệu/frame; ranking không sửa được |
| Đúng trong exact search nhưng thiếu ở Milvus ANN | ANN/scope/index stale | Audit index và ANN recall |
| Đúng ở một nhánh nhưng mất sau fusion | Variant/fusion/object boost | Ablation và bảo vệ candidate theo stream |
| Đúng rank 30–200, top đầu cùng chủ đề nhưng sai chi tiết | Thiếu ranking phân biệt | Cross-encoder + attribute verification |
| Query cần nhiều bước, mỗi bước nằm ở ảnh khác nhau | Thiếu temporal aggregation | Event retrieval + segment ranker |
| Query chỉ rõ câu chữ nhưng caption không chứa | Thiếu OCR evidence | OCR index độc lập |
| VLM không thay đổi thứ hạng hoặc thiếu score | Ảnh thiếu, Drive lỗi, parse/OOM | Diagnostics nguồn ảnh và inference |

## 4. Giai đoạn P0 — benchmark và đồng bộ corpus

### 4.1. Corpus health snapshot

Module đề xuất: `tools/corpus_health.py`.

Ghi theo video/group:

- Mapping count, frame IDs, timestamps; feature shape/model/revision.
- Caption coverage, caption/index/mapping identity và manifest.
- Collection/alias Milvus, row counts theo group, kiểm tra identity theo mẫu.
- Scope cấu hình và scope runtime thực tế, kể cả environment overrides.
- Số ảnh VLM truy cập thành công trên mẫu candidate ở mỗi nhóm; tách Drive/local.
- Trạng thái thiếu caption khác với trạng thái artifact hỏng.

L21 visual-only cần được đánh dấu rõ trong đánh giá: không hạ cứng điểm chỉ vì
không có caption. Có thể báo riêng coverage theo channel cho nhóm này.

Nạp/rebuild bằng công cụ hiện có khi Docker đã sẵn sàng:

```powershell
python -m tools.ingest
# Khi thêm/thay visual features thuộc scope:
python -m tools.ingest --rebuild-visual
```

Sau nạp, khởi động lại service/kernel để tránh giữ caption/index cũ trong RAM.

### 4.2. Tập query và đánh giá

- Giữ 32 query hiện có để regression, nhưng chúng thiên về ảnh L19/L20.
- Tạo thêm khoảng 40–60 query từ corpus mới, gồm 4 dạng query người dùng đã đưa:
  nấu súp, học sinh đi vào trường, cho nguyên liệu theo thứ tự, giảng Tiếng Anh.
- Ground truth gồm video + `[start_seconds, end_seconds]` + các frame/event được
  xác nhận bằng ảnh/video. Không tự suy ground truth từ caption hoặc video ID.
- Gán nhãn candidate khó: cùng món ăn nhưng sai thao tác; cùng trường học nhưng
  thiếu bảng lớp/cầu thang; cùng giáo viên nhưng khác bài/công thức.
- Split theo video hoặc cụm nội dung gần trùng. Các query đã dùng chọn tham số là
  development; holdout mới chỉ dùng báo cáo cuối của mỗi phiên bản đã cố định.
- Đánh giá query planning cố định và live API thành hai track; lưu planner output
  để phân biệt lỗi translation/decomposition với lỗi retrieval.

Metric:

| Nhóm | Metric chính |
|---|---|
| Candidate generation | Known-target Hit/Recall@100/300, event coverage trước rerank |
| Top đầu | Hit@1, Hit@5, MRR@20; nDCG@10 khi có graded qrels đủ tốt |
| Đoạn video | Segment Hit@K theo overlap/IoU đã định trước; ordered-event coverage |
| VLM | Tỷ lệ score hợp lệ, ảnh thiếu, OOM/parse error, số ảnh được chấm |
| Vận hành | P50/P95 latency từng stage, VRAM, số API call và số ảnh Drive đọc |

Chỉ số với qrels chưa exhaustive phải ghi rõ là known-target metrics; không coi
mọi frame chưa gán nhãn là negative chắc chắn.

## 5. P1 — Event retrieval + temporal segment ranker

### 5.1. Query planner có chiến lược theo loại query

Module đề xuất: `kis/query_plan.py`.

Phân loại: `single_frame`, `event_sequence`, `text_critical`, `speech_critical`;
một query có thể có nhiều cờ. Từ schema hiện có tạo:

- Mô tả toàn cảnh để truy hồi broad context.
- Các subquery tiếng Anh cho từng event, giữ đúng động từ và vật thể.
- Ràng buộc thuộc tính và quan hệ có gắn chủ thể.
- `required`, `optional`, `unknown` cho các điều kiện, có dấu vết từ raw query.
- Text cần tìm nguyên văn; event order lấy từ temporal constraints.

Ví dụ p2-1:

```text
e1: beaten egg poured into simmering soup
e2: knife cuts tofu directly above the pot
e3: tofu added to soup
e4: ingredients stirred in the pot
context: soup with shredded mushrooms/bamboo shoots
order: e1 -> e2/e3 -> e4
```

Không dịch "nguyên liệu dạng ống" thành một loại thực phẩm cụ thể khi query
không khẳng định. Không thêm hành động hoặc thứ tự model tự suy diễn.

### 5.2. Temporal store và candidate windows

Module đề xuất: `kis/temporal.py`; artifact `dataset/temporal/<group>/<video>.json`.

- Dùng `pts_time` trong mapping, không lấy frame index chia một FPS cố định khi
  mapping đã có timestamp chính xác.
- Truy hồi mỗi event ở từng channel; union theo frame identity.
- Giới hạn số candidate cho mỗi event để query nhiều event không tăng chi phí vô hạn.
- Mở các cửa sổ thời gian quanh seed hits trong cùng video; thử các duration trên
  development, ví dụ 15/30/60/120 giây, thay vì một window cố định cho mọi query.
- Có shot/chapter boundary thì ưu tiên không ghép qua phần tin khác.
- Gom evidence thành segment nhưng giữ frame đạt từng event và timestamp.

### 5.3. Chấm mức khớp và thứ tự

- Tạo ma trận `event × frame`, score chuẩn hóa theo mỗi event/channel.
- Dùng monotonic alignment/dynamic programming để tìm chuỗi frame có thời gian tăng.
- Không dùng một ảnh duy nhất để tự chứng minh tất cả các thao tác tuần tự.
- Điểm segment kết hợp coverage, evidence strength, order compatibility, temporal
  span và contradiction đã được xác nhận; tune trên development.
- Không yêu cầu mọi event nằm trong cùng shot: query học sinh có thể đổi từ sân
  trường sang hành lang/cầu thang nhưng vẫn thuộc cùng đoạn nội dung.

Trả `video_id`, start/end, matched/missing events, representative frame và timeline.

Rebuild: chủ yếu thêm temporal metadata từ mapping, không phải re-embed visual corpus.

## 6. P1 — Reranker nhẹ trước VLM

Module đề xuất: `kis/text_reranker.py`.

Cross-encoder đọc cùng lúc query và caption/segment evidence, thay vì chỉ so hai
embedding độc lập. Sàng lọc một pool rộng hơn trước khi tải ảnh và gọi Qwen.

- Baseline thử nhỏ: `cross-encoder/ms-marco-MiniLM-L-6-v2` trên English query/caption.
  Đây là ứng viên thử nghiệm, không khẳng định phù hợp nhất với video caption.
- Đánh giá cả mô tả đầy đủ và điểm theo event/constraint.
- Thử pool 100/200/300 trên cùng candidate cache; không tăng depth mù quáng.
- Evidence segment dài cần chunk/event-aware aggregation, không chỉ truncate đầu.
- Caption không có thì giữ visual route/fallback, không gán relevance=0.
- Caption do InternVL sinh là evidence không hoàn hảo; cross-encoder không thể
  xác nhận lại thông tin sai trong caption.

Luồng mục tiêu: union candidate -> reranker nhẹ -> top segment/frame để VLM xem.

Rebuild: không cần đổi visual/text embeddings; thêm model ở online ranking.

## 7. P1 — OCR retrieval riêng cho bài giảng và bảng lớp

Modules đã triển khai v1: `tools/build_ocr.py`, `kis/ocr.py`; chi tiết và cách chạy:
[OCR_RETRIEVAL.md](OCR_RETRIEVAL.md). Đã có schema/checksums, literal routing,
FTS5/BM25 một line, regions content/overlay/unknown, candidate union và temporal
seeds không vượt gate. Suite: 377 passed, 14 skipped (391 tổng); OCR tests 24/24.
Đã chuẩn bị pilot 12 ảnh/12 video nhưng **recognition và quality evaluation còn chờ**.
Crop bảng, fuzzy/cross-line matching, region classification và temporal OCR-state
tracking bên dưới vẫn là mục tiêu, không phải tính năng đã hoàn tất.

- Thử PaddleOCR với cấu hình/model hỗ trợ tiếng Anh và tiếng Việt; pin phiên bản
  sau khi kiểm tra compatibility, chưa thêm dependency trong giai đoạn outline.
- Lưu raw text, normalized text, bbox, confidence, video/frame/timestamp,
  image checksum và OCR model metadata.
- Duy trì hai field: chữ ở vùng nội dung chính và overlay/ticker. Không xóa ticker
  vĩnh viễn vì một số query có thể yêu cầu chính dòng tin đó.
- Với bài giảng, ưu tiên crop bảng; với bảng số lớp, giữ chữ/số ngắn và bbox.
- Index BM25 cho exact/partial phrase; thử fuzzy matching cho lỗi OCR nhẹ.
- Giữ số, phủ định, ký hiệu công thức và nội dung gốc. Chữ OCR mờ phải là unknown,
  không dùng LLM tự điền thành câu hoàn chỉnh.
- Gom OCR theo thời gian và phát hiện thay đổi để tìm các trạng thái ví dụ ->
  giải thích -> công thức, thay vì chỉ một frame có chữ "English".

Áp dụng p2-4: tăng trọng số evidence OCR đúng câu/công thức khi `needs_ocr=true`.

Rebuild: thêm OCR artifacts/index, không cần sinh lại caption hoặc Milvus.

## 8. P2 — VLM verification có rubric theo từng điều kiện

Module đề xuất: `kis/evidence_verifier.py`, mở rộng `kis/vlm_reranker.py`.

- Single-frame query: kiểm tra subject, action, attribute và relation.
- Sequence query: xem 4–8 frame có timestamp theo thứ tự từ candidate segment.
- Trả rubric ngắn `matched / contradicted / unknown` cho từng điều kiện và các
  frame supporting; hạn chế reasoning text để giữ chi phí thấp.
- Ưu tiên nhìn ảnh; caption chỉ bổ trợ. So sánh caption-on/off để đo confirmation bias.
- Đếm riêng ảnh fetch lỗi từ Drive và model score lỗi; không biến missing image
  thành bằng chứng cảnh không liên quan.
- Audit Qwen runtime: reranker hiện còn dùng FP16 compute trong nhánh NF4 và FP16
  khi không quantize; cần kiểm tra NaN/Inf/chuỗi lặp bằng smoke, rồi chọn dtype
  dựa trên GPU. Không khẳng định lỗi đang xảy ra nếu chưa có log.

Ngân sách: cố định số ảnh và API/model call trong A/B. Thử budget 20/40 riêng,
không quy mọi gain của budget lớn hơn cho thuật toán ranking mới.

Không mặc định "verified-first" tốt hơn: profile này có sẵn nhưng cần đo lại.
Final score cho scored/unscored candidate cần chính sách rõ ràng và calibration.

## 9. P2 — Fusion và diversity có kiểm chứng

### 9.1. Fusion calibration

- A/B `rrf` hiện tại với `upgraded` weighted-sum trên cùng corpus/candidate outputs.
- Tách variant fusion và cross-stream fusion; không thay cả hai trong một bước đo.
- Theo dõi visual-only hits bị mất và caption-consensus false positives.
- Thử channel weights theo query type sau khi có nhãn; không chỉnh riêng từng query.
- L21 thiếu caption phải được phân tích riêng, tránh thưởng phạt theo coverage
  một cách ngầm định. Weight routing cần dựa trên evidence có thật.
- Learning-to-rank là bước sau khi có đủ nhãn; chưa cần thêm model học ranking ngay.

### 9.2. MMR/shot diversity

- Áp dụng trước VLM để tránh chấm nhiều ảnh gần giống nhau cùng shot.
- Giữ representatives và neighbor expansion cho người dùng xem.
- Với event sequence, chọn frame đại diện từng event trước rồi mới diversity;
  không loại bước thứ hai chỉ vì hình ảnh vẫn là cùng một nồi súp.
- Không giới hạn một ảnh/video: một video có thể có nhiều đoạn phù hợp.
- Đánh giá Hit@5/MRR cùng duplicate rate, không chỉ nhìn danh sách đa dạng hơn.

## 10. P3 — Bổ sung khi benchmark chỉ ra nhu cầu

### ASR retrieval

Whisper/faster-whisper là ứng viên cho query lời nói, bài giảng hoặc từ khóa
không xuất hiện rõ trên ảnh. Lưu transcript theo timestamp và ngôn ngữ. Câu nói
về một hành động không chứng minh hành động đó đang xuất hiện; giữ evidence audio
khác visual. Cần video/audio nguồn để build.

### Shot-aware sampling

Random gap 80–160 frame có thể bỏ thao tác nhanh như đổ trứng/cắt đậu hũ. Chỉ
re-extract các video/đoạn cần thiết sau khi chứng minh frame đúng không có. Sampling
mới tạo phiên bản frame/mapping/features riêng, không trộn frame numbering cũ.

### Embedding thay thế

Chỉ benchmark model text/visual khác sau khi tách rõ lỗi recall, caption và ranking.
Đổi embedding đòi hỏi re-embed/index corpus; đánh giá trên subset trước, giữ model
revision và vector dimensions đúng cho cả query lẫn corpus.

## 11. Kiến trúc mục tiêu

```text
Query
  -> QueryPlan: global + events + constraints + OCR/ASR flags
  -> Per-event visual/caption retrieval + optional OCR/ASR retrieval
  -> Candidate union + evidence/coverage tracing
  -> Frame/segment assembly using timestamps
  -> Lightweight reranker + ordered-event alignment
  -> Event-aware diversity
  -> Budgeted VLM image/clip verification
  -> Calibrated final ranking
  -> Top segments + representative frames + matched/missing evidence
```

UI nên có timeline, frame đại diện theo event và lý do thiếu bằng chứng. Giữ API
frame-result tương thích khi thêm segment fields. Không cần tách microservices
cho giai đoạn đầu; thêm module vào package `kis/` và tooling offline trong `tools/`.

## 12. Trình tự triển khai và điều kiện quyết định

| Bước | Nội dung | Offline/rebuild | Tiêu chí chính |
|---|---|---|---|
| 0 | Corpus health, freeze benchmark + baseline | Kiểm tra/nạp index thiếu | Ground truth nằm trong đúng scope, live counts/identity được xác minh |
| 1 | Query decomposition + temporal windows/alignment | Temporal metadata từ mapping | Sequence Hit@5, ordered-event coverage tăng |
| 2 | Reranker nhẹ trên pool rộng hơn | Không re-embed | MRR@20/Hit@5 tăng ở cùng candidate set |
| 3 | OCR riêng, trước tiên trên giáo dục/trường học | OCR artifacts/index | Text-critical Hit@5 tăng, không chỉ cùng chủ đề |
| 4 | VLM rubric và clip verification | Không sinh lại caption toàn bộ | Tăng top đầu ở cùng budget, tỷ lệ unknown/error rõ ràng |
| 5 | Fusion theo loại query + event-aware MMR | Không re-embed | Giảm lỗi top-1/duplicates mà không mất event evidence |
| 6 | ASR/sampling/embedding khi cần | Có thể tốn offline compute | Chứng minh gain tương xứng chi phí |

Mục tiêu pilot có thể đặt trước khi chạy: tăng ít nhất 10 điểm phần trăm Hit@5
hoặc cải thiện MRR rõ ràng trên tập mục tiêu; đây là tiêu chí kỳ vọng, không phải
cam kết kết quả. Báo paired wins/losses, confidence interval khi cỡ mẫu cho phép,
regression theo nhóm và P95 latency. Baseline mạnh phải bao gồm những module
đã chứng minh có lợi, không so với pipeline cũ đã bị thay nhiều thành phần.

**Gói nên làm đầu tiên:** bước 0, sau đó event retrieval + temporal ranking và
reranker nhẹ; OCR được triển khai tiếp cho query bài giảng. Đây là hướng giải
quyết việc top đầu cùng chủ đề nhưng sai chi tiết/thứ tự mà không phải chạy lại
caption toàn bộ dữ liệu.
