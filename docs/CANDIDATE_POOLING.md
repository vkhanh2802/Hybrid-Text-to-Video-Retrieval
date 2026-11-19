# Chọn video ứng viên trước temporal scoring

## 1. Vấn đề cần giải quyết

Temporal DP chỉ tìm đường đi trong **những video đã được chọn vào pool**. Nếu
video đúng bị loại trước đó, DP hay VLM mạnh hơn không thể tự khôi phục nó.
Tập holdout cũ đã cho thấy hiện tượng này; không dùng các query holdout đó để
chọn cấu hình mới.

Pipeline:

```text
Tách query thành events
  -> SigLIP/BM25/MiniLM tìm frame từng event
  -> Chọn tập video sẽ được đọc/chấm local  <-- bước nâng cấp này
  -> Chấm local frame của mỗi event
  -> DP ghép theo timestamp
  -> Kết quả đoạn + frame bằng chứng
```

**Tác dụng kỳ vọng:** phân bổ ngân sách local scoring hợp lý hơn, tăng cơ hội
video có bằng chứng cho nhiều event được xét. Không bảo đảm frame/event đúng;
đây là candidate allocation, không phải xác minh hành động.

## 2. Ba policy có thể bật/tắt độc lập

### `round_robin` — giữ baseline

Đi qua từng mức rank của các danh sách frame, luân phiên event/stream; lấy video
mới gặp đầu tiên tới khi đủ `max_videos`. Đây là hành vi cũ, được giữ nguyên để
đối chứng. Không tính tổng bằng chứng mà video nhận được từ các event.

### `event_rrf` — gộp bằng chứng ở cấp video

1. Với từng **event × stream**, loại lặp video, giữ vị trí xuất hiện đầu tiên.
   Một video có 20 frame gần giống nhau **không nhận 20 phiếu**.
2. Xếp hạng danh sách **video phân biệt**. Ví dụ frame hits `A,A,A,B` trở thành
   video ranks `A=1, B=2`, không đẩy B xuống rank 4 vì nhiều frame của A.
3. Gộp rank qua streams bằng weighted reciprocal rank fusion:

```text
S_event(video) = sum_stream(weight_stream * (k+1)/(k+video_rank))
weights: visual=0.5, BM25=0.2, semantic=0.3; k=60

S_video = 0.7 * mean(S_event) + 0.3 * max(S_event)
```

Mean ưu tiên video có evidence cho nhiều bước; max giữ cơ hội cho một event
đặc trưng dù các event khác truy hồi yếu. Event không có hit trong seed list
đóng góp 0 vào **pool score**, không bị diễn giải là hành động không xảy ra.

BM25 và semantic cùng dựa caption nên không coi là hai nguồn kiểm chứng độc
lập. Tổng trọng số caption family là 0,5. Công thức/trọng số là heuristic cố
định cho pilot, chưa được chứng minh tối ưu; không tune chúng trên holdout cũ.

### `balanced_rrf` — RRF với chỗ dự phòng cho từng event

- Tìm video mạnh nhất cho từng event bằng `S_event`.
- Giữ các video thắng event đó trước, rồi lấp phần budget còn lại bằng `S_video`.
- Nếu video thắng nhiều event thì chỉ chiếm một slot. Không lấy giao top-k cứng.
- Nếu số video thắng event vượt budget, chọn theo global score và video ID để
  tie-break, không lấy theo thứ tự event. Không thể bảo đảm mọi event có slot
  khi budget quá nhỏ.

**Ví dụ:** một video phổ thông có điểm vừa phải ở mọi event; video khác chỉ
có cảnh “cắt đậu hũ” rất đặc trưng. Policy balanced có thể giữ video thứ hai để
local scorer tìm tiếp các bước còn lại, thay vì để video phổ thông chiếm hết pool.

Đảo thứ tự event không làm thay đổi phép gộp điểm/reservation về mặt thiết kế;
**thứ tự chỉ được kiểm tra ở DP sau đó**. Không được dùng rank score để vượt
visual/semantic support gates hoặc tự gán `verified=true`.

## 3. Ngân sách và trade-off

| Pool budget | Trần frame chấm local với cap 1.500/video | Trade-off |
|---:|---:|---|
| 12 video | 18.000 | Rẻ hơn nhưng dễ loại video cần thiết |
| 24 video | 36.000 | Thêm cơ hội recall, thêm I/O/scoring và distractors |
| 48 video | 72.000 | Phủ rộng hơn; không mặc định tốt hơn hoặc tăng top-1 |

Thực tế video thường ít hơn 1.500 frame; xem `scored_frames` chứ không dùng
trần lý thuyết như số đo. Tăng budget chỉ giúp video đã có mặt trong **union
seed lists**; không cứu video vắng toàn bộ seeds. Muốn đánh giá seed depth cần
thí nghiệm riêng, không gán gain tăng `event_top_k` cho pooling.

Thêm distractors có thể làm top-1 kém đi do caption/event support còn nhiễu.
Không tăng pool lên 48 chỉ vì có RAM hoặc vì video đích có thể xuất hiện sâu hơn.

## 4. Code và cấu hình

- `kis/candidate_pool.py`: các policy thuần CPU/rank, không load model.
- `kis/event_retrieval.py`: gọi selector trước local scoring, giữ toàn seed
  frame IDs của video được chọn.
- `kis/temporal.py`: thêm `candidate_pooling`, `candidate_rrf_k`.
- Diagnostics `temporal.candidate_pool`: global rank, score từng event, vote
  từng stream, event reservation, selected flag. Có thể giải thích vì sao một
  video được chọn hoặc bị loại khỏi pool.

Mặc định vẫn **`round_robin`, 12 video** cho đến khi có cơ sở đổi. Ví dụ bật
thử qua Python:

```python
from dataclasses import replace

experiment = replace(runtime_config.temporal,
                     candidate_pooling="balanced_rrf", max_videos=24)
results = solve_kis(..., temporal_config=experiment)
```

Hoặc thêm vào object `temporal` của `config/pipeline.json`, rồi restart service:

```json
{"candidate_pooling": "balanced_rrf", "candidate_rrf_k": 60, "max_videos": 24}
```

Đoạn trên là **các field để sửa trong object hiện có**, không thay toàn file
pipeline. Không cần Kaggle, caption generation, index rebuild, OCR/ASR hoặc model mới.

## 5. Thiết kế kiểm tra không lẫn nhiều thay đổi

`tools/candidate_pool_ablation.py` lấy **cùng saved event hits** từ run có nhãn
Drive v1, chỉ mở record files của bốn query development. Không dùng outcomes
holdout cũ để chọn policy/budget. So 3 policies × 3 budgets = 9 cấu hình.

- Seed depth cố định 50, query/events cố định, giữ nguyên support gates/DP.
- Replay nhẹ: chỉ tính video được chọn, không load encoder.
- `--rescore`: CPU cached SigLIP/MiniLM, chấm local từng video trong union các
  cases một lần rồi chia lại cùng score table cho từng case.
- Báo target-video inclusion, top-video rank, known-path hits, frame workload,
  artifact failures. Không đổi nhãn cửa sổ để làm metric đẹp hơn.
- `scoring_seconds_sum` cộng thời gian local đã đo một lần theo video; do reuse
  và cache, **không phải latency request**, không phải P50/P95 hoặc tốc độ end-to-end.
- Source/label hashes và source-run context hash được lưu. Output version mới,
  không ghi đè run cũ; không ghi vào corpus/Milvus.

```powershell
python -B -m tools.candidate_pool_ablation --labels benchmark_results/drive_groundtruth_20261102_seed20261102/frozen_labels_v1.json --source-run benchmark_results/groundtruth_eval_20261102_002150
# Thêm --rescore để chấm local bằng model đã cache trên CPU.
```

### Kết quả replay và local rescoring

Run: `benchmark_results/candidate_pool_dev_rescore_20261102/`.
Không dùng holdout cũ để chọn cấu hình; các con số sau chỉ từ 4 development
sequence queries với nhãn Drive v1 đã đóng băng.

| Policy | Budget video | Target trong pool | Top video đúng video mẫu | Tổng frame chấm local, 4 queries |
|---|---:|---:|---:|---:|
| round_robin | 12 | 4/4 | 4/4 | 12.843 |
| round_robin | 24 | 4/4 | 4/4 | 25.256 |
| round_robin | 48 | 4/4 | 3/4 | 50.803 |
| event_rrf | 12 | 4/4 | 4/4 | 12.576 |
| event_rrf | 24 | 4/4 | 4/4 | 25.244 |
| event_rrf | 48 | 4/4 | 4/4 | 50.773 |
| balanced_rrf | 12 | 4/4 | 4/4 | 12.540 |
| balanced_rrf | 24 | 4/4 | 4/4 | 25.244 |
| balanced_rrf | 48 | 4/4 | 4/4 | 50.773 |

- **0 artifact failures** ở mọi cấu hình.
- Known-path Hit@20 theo frozen labels v1 vẫn **0/4** ở mọi cấu hình. Không đổi
  annotation để gán lợi ích cho pooling; xem phân biệt known paths và posthoc
  visual judgments trong [DRIVE_GROUNDTRUTH_RESULTS.md](DRIVE_GROUNDTRUTH_RESULTS.md).
- Với `round_robin_48`, query `drive_02_sequence` bị đẩy top từ target L16_V028
  sang L20_V014. Pool rộng hơn đưa thêm candidate có score DP cao vào cạnh tranh;
  chưa xem ảnh L20_V014 nên không khẳng định mọi nội dung của nó đều sai.
- `balanced_rrf_12` xử lý ít hơn 303 frame so baseline (khoảng 2,36%), nhưng mức
  giảm workload nhỏ và chưa đủ để kết luận latency tốt hơn. Time sums trong JSON
  có reuse/cache, không dùng làm speedup end-to-end.
- Target coverage đã ceiling 4/4 ngay ở baseline12, nên **chưa chứng minh tăng
  recall/top-1 ở budget mặc định**, cũng chưa có đánh giá độc lập trên holdout mới.

**Quyết định:** giữ mặc định `round_robin`, budget12; hai policy mới opt-in để
thử nghiệm. Không tăng mặc định lên48 vì vừa tăng workload gần4× vừa có regression
ở policy cũ. Muốn promote balanced_rrf cần dev khó hơn và holdout mới độc lập;
không chọn config dựa trên kết quả holdout cũ đã xem.

## 6. Điều kiện để triển khai rộng

Xác minh sau thay đổi: `python -B -m unittest discover -s tests` — **257 tests,
243 đạt, 14 skipped**, không failure/error. Các test build/ingest dùng mock và
thư mục tạm; log staging/publish của chúng không phải rebuild collection thật.
Kiểm thử riêng pooling/ablation: **10/10 đạt**; `git diff --check` đạt.

- Không làm giảm candidate coverage/top-video trên dev, và có lợi ích đo được
  so với baseline dưới cùng ngân sách.
- Không đánh đổi bằng việc chọn nhiều false full paths hoặc tăng chi phí quá mức.
- Nếu dev đã ceiling và các cấu hình hòa: giữ default rẻ/cũ, cần thêm dev cases
  khó hơn và một holdout mới để xác minh. Không tuyên bố gain ngoài mẫu thử.
- Giải quyết candidate coverage không thay thế calibration event support hoặc
  xác minh chi tiết bằng ảnh/OCR; các vấn đề này vẫn tách riêng.
