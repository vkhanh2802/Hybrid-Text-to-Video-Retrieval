# Bước 1: Event retrieval + temporal alignment

> **Web cập nhật 14/11/2026:** `SearchService` dùng `require_full_results=True`,
> luôn tìm danh sách chính 100 frame khác nhau và trả chuỗi ở `temporal_results`
> riêng (`temporal.presentation="auxiliary"`). Qwen chấm top 20 frame chính,
> không chấm/xác minh cả chuỗi. Routing early-return mô tả dưới đây vẫn được giữ
> cho `solve_kis`/benchmark cũ khi không bật hợp đồng full-results.

## Đã triển khai

Nhánh tìm chuỗi sự kiện dùng **model/embeddings đang có**, không tạo caption,
không chạy image encoder trên corpus, không yêu cầu OCR/ASR/model video mới.

```text
Planner API hiện có: events + temporal_constraints
  -> EventPlan (chỉ nhận sequence rõ ràng, 2–6 bước)
  -> từng event: SigLIP/Milvus + BM25 + MiniLM
  -> union video theo round-robin event/modality, tối đa 12 video
  -> chấm lại frame trong từng video bằng visual/caption embeddings sẵn có
  -> DP theo timestamp thực, có abstention cho event yếu
  -> một evidence span/video + frame gợi ý từng bước
```

Query đơn hoặc không có thứ tự rõ ràng vẫn đi qua pipeline RRF/object/VLM cũ.
Nếu không có temporal path đủ ngưỡng ở bất kỳ video nào, trả lại kết quả frame
của pipeline cũ với `temporal.status="fallback"` và chẩn đoán lỗi.

## Thành phần

- `kis/query_plan.py`: lấy thứ tự từ constraint, không suy thứ tự từ vị trí event;
  từ chối partial order, event trùng, nhiều constraint hoặc vượt budget.
- `kis/event_retrieval.py`: seed retrieval, giới hạn video/frame, memory-map NPY,
  kiểm tra mapping, chấm local visual/semantic và tạo output.
- `kis/temporal.py`: cấu hình, support gates, DP O(events × frames).
- `kis/text.py::CaptionRetriever.score_frames`: cosine chính xác trên caption
  vectors đã load; frame thiếu caption có score unknown, không bị gán âm.
- `kis/solve_kis.py`: routing/fallback, không trộn score frame với score sequence.
- `kis/web_app.py`: API toggle, serialize ảnh từng event và danh sách chuỗi bổ
  sung; Qwen lazy-load để chấm top 20 frame chính, không chấm chuỗi temporal.
- `web/app.js`: mở chi tiết chuỗi, hiển thị bước thiếu bằng chứng và link ảnh.

Planner được hướng dẫn tạo **action tiếng Anh, tự đủ ngữ cảnh, trung thành với query**
ngay trong lần gọi API hiện có. Không thêm lần gọi LLM để tách event. Saved query
cũ có event tiếng Việt không được tự dịch ở local stage; cần planner/fixture mới
hoặc kiểm tra ngôn ngữ trước khi dùng encoder English-oriented.

## Cấu hình và A/B

`config/pipeline.json` chứa `temporal`:

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `enabled` | true | Chỉ kích hoạt nếu có explicit sequence |
| `max_events` | 6 | Không âm thầm cắt bớt chuỗi dài |
| `event_top_k` | 50 | Mỗi event, mỗi seed stream |
| `max_videos` | 12 | Số video được local rescore |
| `candidate_pooling` | round_robin | round_robin / event_rrf / balanced_rrf; mặc định giữ cách cũ |
| `candidate_rrf_k` | 60 | Hằng số RRF cho video candidate allocation, không phải event verification |
| `max_frames_per_video` | 1500 | Tối đa frame local; giữ seed rows trước, phần còn lại lấy đều theo row |
| `max_gap_seconds` | 60 | Khoảng cách tối đa giữa hai frame **có support** liên tiếp |
| `gap_penalty` | 0.001 | Phạt mỗi giây của evidence span |
| `min_visual_similarity` | 0.25 | Ngưỡng cosine visual cho support |
| `min_semantic_similarity` | 0.45 | Ngưỡng cosine caption cho support |

**Các ngưỡng là giá trị pilot, chưa được calibrated trên corpus.** Một frame
được support khi ít nhất một raw cosine vượt ngưỡng; không dùng RRF rank hoặc
min-max để biến top hit yếu thành bằng chứng mạnh. Local score lấy trung bình
cosine không âm theo trọng số visual/semantic `0.6/0.4`, chuẩn hóa lại trên các
kênh thực sự có dữ liệu. BM25 chỉ tạo seed; chưa tham gia local evidence scoring.

Giải thích selector mới, tác dụng và thí nghiệm cùng seed/budget:
[CANDIDATE_POOLING.md](CANDIDATE_POOLING.md).

Đã có pilot hiệu chỉnh support gates trên tập ảnh Drive v2 với positives và
near negatives, tách dev/holdout: [EVENT_SUPPORT_CALIBRATION.md](EVENT_SUPPORT_CALIBRATION.md).
Profile này chưa thay mặc định runtime; pair-level gain không đồng nghĩa gain
Hit@K hoặc đúng chuỗi end-to-end.

Bộ bằng chứng điều kiện trang phục opt-in (caption không phải pixel verification):
[EVENT_CONDITION_EVIDENCE.md](EVENT_CONDITION_EVIDENCE.md). Mặc định `off`, giữ
ngưỡng baseline và bảo toàn visual support khi thử `semantic_guard`.

DP ưu tiên số event được support, sau đó tổng score trừ gap penalty. Cho phép
bỏ event nhưng không tự điền frame cho bước không support. Hai event theo thứ tự
không được dùng cùng timestamp. Với penalty tuyến tính này, tổng penalty là
`lambda × (last_time - first_time)`, **không phải mô hình duration từng hành động**.
Score cuối đảm bảo nhiều bước có support xếp trước ít bước có support:

```text
quality = clip((sum(event_score) - lambda * span_seconds) / supported_count, 0, 1)
temporal_score = (supported_count + 0.5 * quality) / (event_count + 0.5)
```

Tắt toàn cục trong PowerShell rồi khởi động lại service:

```powershell
$env:KIS_DISABLE_TEMPORAL = "1"
```

Hoặc A/B qua API, không đổi server config:

```json
{"query": "Đổ trứng vào nồi, sau đó khuấy súp", "temporal": false}
```

`temporal=true` ép bật routing (vẫn cần explicit sequence), `false` chạy baseline,
không truyền thì dùng config. Khi benchmark cần **cố định cùng structured query**
để không gán thay đổi planner cho temporal ranking. Python:

```python
from kis.temporal import TemporalConfig
from kis.solve_kis import solve_kis

# Giữ nguyên structured_query, các retriever, encoder và retrieval policy.
baseline = solve_kis(..., temporal_config=TemporalConfig(enabled=False))
temporal = solve_kis(..., temporal_config=TemporalConfig(enabled=True), diagnostics={})
```

Đo retrieval-only với `vlm_reranker=None`, cùng ngân sách seed/candidate được báo
rõ. Nhánh temporal **không gọi VLM ảnh đơn**: một ảnh không chứng minh A → B → C.
Nếu fallback về baseline và VLM vẫn bật, baseline vẫn có thể dùng Qwen như trước;
đặt `KIS_DISABLE_VLM=1` trước khi khởi động để tránh mọi lần nạp Qwen.

## Output và ý nghĩa

Mỗi kết quả temporal dùng frame đầu có support làm ảnh đại diện và thêm:

- `temporal.status`: `retrieved_complete` hoặc `partial`.
- `verified=false`: **không phải xác nhận đã xảy ra mọi hành động**.
- `coverage`: số event support / số event yêu cầu.
- `start_seconds`, `end_seconds`: **evidence span**, không phải clip boundary.
- `events`: action, `retrieved/unknown`, frame ID, keyframe number, timestamp,
  local score, raw visual/semantic cosine và URL ảnh nếu có thể truy cập.
- `sampled`, `total_frames`, `scored_frames`, `seed_frames_scored`: độ phủ local pool.

API top-level `temporal` là diagnostics: trạng thái, seed streams từng event,
video đã chọn, cấu hình, lỗi artifact từng video, thời gian chạy. Khi applied,
`fusion_method="temporal_dp"`; không giả rằng đây là score RRF.

`stage_seconds.temporal` chỉ xuất hiện khi event retrieval thực sự chạy, kể cả
khi không có path đủ ngưỡng và `temporal.status="fallback"`. Query đơn, temporal
bị tắt hoặc planner không tạo được sequence không có khóa này (không ghi timing 0
giả). Các stage baseline sau fallback được đo riêng, không cộng trùng thời gian
temporal.

`retrieved_complete` chỉ nghĩa **mọi bước có frame vượt heuristic gate đúng thứ tự**.
Hành động gần giống, caption sai hoặc query drift vẫn có thể gây false positive.
UI ghi rõ trạng thái chưa xác minh, không gọi các bước này là đã được VLM kiểm chứng.

## Artifact, ngân sách và giới hạn

- Cần mapping CSV có `n, frame_idx, pts_time`. Không suy giây từ frame ID hoặc
  keyframe number. Sắp xếp timestamp sau khi join vector theo row gốc.
- Legacy mapping có thể lặp `frame_idx` giữa hai keyframe khác nhau. Các dòng
  chung frame ID đó được loại khỏi temporal scoring (không đoán timestamp/ảnh),
  giữ nguyên row join của các vector còn lại. Diagnostics ghi
  `ambiguous_frame_rows_excluded`; không sửa CSV/NPY gốc hoặc loại cả video.
- Cần NPY visual tương ứng mapping; count/dimension/finite vectors được kiểm tra.
  Nếu NPY thiếu có thể dùng caption vectors; caption thiếu không loại video visual-only.
  Nếu cả hai thiếu thì không có support và có thể fallback.
- Kiểm tra caption keyframe identity với local mapping. **Chưa có provenance
  manifest mới cho NPY**: hai file count bằng nhau vẫn có thể sai model/thứ tự;
  chỉ dùng features được tạo bởi pinned SigLIP và mapping tương ứng.
- Không tải ảnh để temporal-score: dùng dot products. 6 events tạo tối đa
  18 seed searches, đọc tối đa 18.000 local frame rows theo config mặc định.
  CPU vẫn chạy text encoders đang có; không cần Kaggle để tạo artifact mới ở bước này.
- NPY dùng mmap, cache tối đa 2 video; semantic query vector cache trong một request.
  Đổi corpus/mapping khi server đang chạy cần restart để tránh cache cũ.
- Sampling có thể bỏ evidence giữa seed frames; diagnostics phải được xem trước
  khi kết luận temporal model sai. Giới hạn video cũng có thể làm mất đáp án.
- Hiện chỉ trả **một path tốt nhất/video**, chưa multiple segments, chưa shot/story
  boundaries, chưa simultaneous events, duration model hoặc OCR/ASR độc lập.
- Gap 60 giây là pilot constraint, có thể loại chuỗi dài hợp lệ. Cùng video dài
  vẫn có thể chứa nhiều bản tin; gap bound chưa bảo đảm cùng câu chuyện.

## Kiểm thử và nghiệm thu

### Kiểm tra artifact toàn scope, không nạp model

```powershell
python -B -m tools.temporal_preflight
```

Lệnh kiểm tra mapping/timestamp, dimension/finite/zero vectors theo block nhỏ,
caption identity và nhóm thiếu mapping trong scope hiện tại. Báo cáo ghi vào
`benchmark_results/temporal/preflight.json` (gitignored); exit code khác 0 nếu
có video lỗi, nhóm thiếu mapping hoặc không tìm thấy video nào.

Đây là kiểm tra file, không gọi Docker/Milvus, không tải model, không sinh lại
artifact và không đo chất lượng tìm kiếm. Caption index/manifest và provenance
NPY chưa được xác minh bởi lệnh này. Danh sách video lấy từ mapping nên không
phát hiện được video mất cả mapping nếu không có inventory đối chiếu bên ngoài.

Unit tests cho công cụ: `python -B -m unittest tests.test_temporal_preflight`.

Lượt preflight đầu: phát hiện 535 video, 525 ready và 10 lỗi tại L21_V006,
L21_V007, L21_V012–L21_V016, L21_V022, L21_V023, L21_V029. Cả 10 có hai
dòng đầu cùng `frame_idx=0` tại timestamp 0 và 0,0333333 giây, không có keyframe
number trùng hoặc timestamp không hợp lệ. Runtime đã thêm cơ chế loại riêng
các dòng mơ hồ. Lượt kiểm tra lại tại
`benchmark_results/temporal/preflight_after_duplicate_handling.json` xác nhận:
**535/535 video ready, 0 lỗi, 138.796 mapping rows, không thiếu nhóm mapping**.
Có 29 video visual-only; 10 video cảnh báo với tổng 20 dòng mơ hồ bị loại khỏi
temporal scoring, còn 138.776 dòng không mơ hồ trước sampling. CSV/NPY gốc không
bị thay đổi. Đây là artifact readiness, không phải accuracy hoặc xác minh live index.
Test hồi quy sau xử lý legacy duplicates: **36/36 đạt** qua
`python -B -m unittest tests.test_temporal_preflight tests.test_temporal tests.test_solve_kis tests.test_runtime_config`.

Không chạy model thật trong unit tests; dùng numpy, FAISS nhỏ và mock encoder/API.

```powershell
python -B -m unittest tests.test_temporal tests.test_solve_kis tests.test_runtime_config
node --check web/app.js
```

Bao phủ: thứ tự explicit, đảo query, event thiếu, timestamp bằng nhau, gap quá xa,
mapping/feature mismatch, caption thiếu, sampling giữ seed, video quota, fallback,
toggle, serialize event images, lazy VLM. DP được đối chiếu exhaustive enumeration
trên 60 fixture ngẫu nhiên có skips/ties, không chỉ vài happy paths.

Lượt xác minh toàn suite: `python -B -m unittest discover -s tests` — **228 tests,
214 đạt, 14 bỏ qua** vì thiếu PyAV hoặc sample artifacts; không có failure/error.
`node --check web/app.js` và `git diff --check` cũng đạt (Git có cảnh báo line ending).

**Chưa có benchmark chất lượng trên corpus thật**: cần live Milvus/Docker và nhãn
video + frame/timestamp từng event. Các query regression hiện tại chủ yếu là ảnh
đơn nên không dùng để chứng minh sequence correctness. Khi có nhãn, đo baseline,
unordered aggregation, DP lambda=0, DP có penalty; thêm đảo thứ tự và event giả.
Hit@K đúng video không thay được event/path correctness.

### Pilot retrieval thật, cố định query và không dùng model mới

Pilot đã chạy thành công trên corpus thật. Xem kết quả và giới hạn tại
[TEMPORAL_PILOT_RESULTS.md](TEMPORAL_PILOT_RESULTS.md): module vận hành được,
nhưng chưa chứng minh tăng accuracy; phát hiện semantic false-positive risk và
visual threshold chưa được calibrated.

```powershell
python -B -m tools.temporal_pilot
```

Fixture: `benchmarks/temporal_pilot.json` gồm ba mô tả rút gọn từ ví dụ của người
dùng (súp/trứng/đậu hũ, học sinh/hành lang, bảng bài giảng tiếng Anh). **Không có
ground truth**: chưa tính Hit@K/MRR/path accuracy và không coi top result là đáp án.
Query tiếng Anh/events được cố định để không lẫn biến thiên planner API vào kết quả.

Runner chỉ dùng pinned SigLIP/MiniLM đã cache trên CPU; ép Hugging Face offline,
không tải weights, không gọi planner hoặc Qwen, không fetch ảnh, không ghi vào
Milvus/corpus. Thiếu cache thì dừng và ghi `failure.json`, không tự tải model.
Kết quả vào thư mục mới `benchmark_results/temporal_pilot_<timestamp>/`:

- `context.json`: cấu hình, model/source/query hashes, collection ID/row count,
  caption count và thời gian nạp model; không lưu API key.
- `results/<query_id>.json`: cùng structured query, kết quả baseline/temporal,
  timings, diagnostics, và controls trên **cùng local candidate pool**.
- `summary.json`: top video, coverage tự báo, latency và lỗi artifact (không phải accuracy).
- `retrieval.log`: tiến trình chi tiết. `run.lock` ngăn ghi đè cả run bị ngắt.

Controls: unordered max mỗi event, ordered DP, DP lambda=0, đảo thứ tự hàng score
trên cùng pool. Nếu original và reversed đều complete thì đó là dấu hiệu cần kiểm
tra model/evidence ambiguity, không chứng minh hệ thống đã hiểu thứ tự. Chưa có
event giả gán nhãn hoặc kiểm chứng bằng ảnh trong runner này.

**Không phải A/B đồng ngân sách**: baseline dùng 100 hit/stream/variant; temporal
dùng 50 hit/stream/event và local rescore. Hai nhánh trả đơn vị khác nhau (frame
so với evidence span). Luân phiên thứ tự chạy chỉ giảm thiên lệch warm cache,
không đủ để coi latency một lần là benchmark P50/P95. Lợi ích ordering chỉ được
tách tương đối bằng same-pool controls; cần nhãn độc lập để kết luận chất lượng.

Kiểm tra artifact-only trên hai video thật (không nạp encoder/model):

| Video | Mapping rows | NPY shape | Timestamp đầu/cuối (giây) |
|---|---:|---|---|
| L19_V001 | 243 | 243 × 768 | 0 / 976,1 |
| L21_V031 | 285 | 285 × 768 | 0 / 1096,6 |

Đây chỉ là kiểm tra khả năng đọc mapping/features, không xác minh nội dung,
model provenance hoặc coverage toàn corpus. UI đã được kiểm tra DOM trong
browser bằng response giả lập có một bước unknown; không gọi search/model thật.

Không có lệnh full caption/embedding/OCR/ASR nào được chạy cho bước triển khai này.
Nếu chẩn đoán cho thấy cần thêm frame, re-embed hoặc clip model nặng, **dừng và
chuyển yêu cầu artifact sang Kaggle theo quyết định của người dùng**.
