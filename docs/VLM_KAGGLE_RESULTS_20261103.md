# Kết quả Qwen VLM từ Kaggle — 03/11/2026

> Bước tiếp theo đã hoàn tất: [thử nghiệm multi-path trên CPU](TEMPORAL_MULTIPATH_RESULTS_20261103.md).
> Không tăng known-path coverage; giữ production và không chạy VLM mới.

## Kết luận

**Đã kiểm tra và replay thành công file `dist/vlm-results.zip` người dùng gửi.**
Qwen2.5-VL-3B mang lại cải thiện nhỏ về ranking ở vài query, nhưng không phục
hồi được candidate/path chưa được retrieval/local DP tìm thấy. Chưa có cơ sở
đổi phương pháp retrieval hoặc bật joint-path VLM thành mặc định production.

**Không chạy lại VLM, không gọi lại retrieval/Drive/API, không retune ngưỡng,
không sửa nhãn hoặc bản dịch trong lượt phân tích này.** LLM planner chưa được
đánh giá; đây vẫn là fixed-plan control, không phải toàn bộ pipeline từ LLM.

## 1. Tính toàn vẹn và coverage

ZIP kết quả SHA-256:

```text
07452d7b50e49a798e02272b6b52f625df9024e5508c828836ee875bbe9ad480
```

- ZIP chỉ gồm4 file được cho phép: `scores.jsonl`, `run_context.json`,
  `hardware.json`, `summary.json`; kiểm tra CRC/size/path trước giải nén.
- Đủ **2.376 task IDs**, không trùng/ngoại lai; status thiếu ảnh khớp manifest.
- Model/revision: `Qwen/Qwen2.5-VL-3B-Instruct` @
  `66285546d2b821cf421d4f5eb2576359d3770cd3`.
- Prompt `kis_blind_image_and_ordered_path_v1`,4-bit NF4, giới hạn ảnh256²–512²,
  code hashes khớp packet và sources của CPU ablation vẫn khớp bản đã khóa.
- Máy chạy theo metadata trả về: **Tesla T4**,15.636.037.632 bytes VRAM,
  torch2.11.0+cu128,CUDA12.8,transformers5.15.0,bitsandbytes0.50.2.

| Trạng thái | Tác vụ |
|---|---:|
| Score hợp lệ | **2.128** |
| Không trả score hợp lệ | 64 |
| Thiếu ảnh, không chấm | 184 |
| Tổng đã ghi nhận | **2.376** |

Trong2.296 image tasks:2.052 scored,62 invalid,182 missing. Trong80 path tasks:
76 scored,2 invalid,2 missing. Mỗi caption mode có1.064 scored,32 invalid,
92 missing. Không đổi invalid/missing thành0 hay negative.

**Giới hạn log:** ZIP không có raw model responses hoặc traceback từng task.
`invalid_score` có thể là parse failure, hoặc reranker trả `None` khi OOM;
không thể tách các nguyên nhân này từ bốn file đã nhận. Không suy64 task đó
đều OOM hoặc đều lỗi parser. Không cần chạy lại GPU chỉ để chọn score đẹp hơn.

## 2. Thiết kế đối chiếu được giữ nguyên

- 51 query units ×3 retrieval arms =153 cases CPU; mỗi case có2 caption modes,
  thành306 comparisons VLM.
- **Development:**11 image queries support-v2 và4 sequence queries dev-v1,
  nhãn AI quan sát ảnh, chưa được người dùng xác nhận toàn bộ.
- **Human-v3:**26 image queries và10 sequence queries; nhãn/bản dịch đã được
  người dùng duyệt. Bộ này đã quan sát trước đây, chỉ là **regression**, không
  còn là independent new holdout.
- Arms: Full100 hits/stream,Full deep200 hits/stream,Full + compact100 hits
  mỗi variant cho ảnh đơn; chuỗi dùng50/100/50×2 tương ứng. Giữ pool12 video.
- Ảnh đơn chấm20 ứng viên đầu, joint-path VLM chấm tối đa5 complete paths.
- VLM weight0,5 và đúng `finalize_vlm_ranking` production. Không threshold
  selection, không verified-first mới; invalid/missing giữ retrieval score.
- Mọi query vẫn vào denominator retrieval, kể cả không có complete path để
  VLM chấm. Không chỉ báo metric trên các query có score thuận lợi.

## 3. Ảnh đơn: so với đúng baseline của từng arm

Các mức Hit@K trong hai caption modes trùng nhau ở bảng dưới; MRR được báo
riêng để không nói toàn bộ ranking giống nhau.

### Development,11 query

| Retrieval arm | CPU Hit@1 | +VLM Hit@1 | CPU Hit@5 | +VLM Hit@5 | CPU Hit@20 | +VLM Hit@20 |
|---|---:|---:|---:|---:|---:|---:|
| Full | 2/11 | **3/11** | 3/11 | **4/11** | 5/11 | 5/11 |
| Full deep | 3/11 | 3/11 | 3/11 | **4/11** | 5/11 | **6/11** |
| Full + compact | 4/11 | 4/11 | 4/11 | **5/11** | 6/11 | 6/11 |

### Human-v3 regression,26 query

| Retrieval arm | CPU Hit@1 | +VLM Hit@1 | CPU Hit@5 | +VLM Hit@5 | CPU Hit@20 | +VLM Hit@20 |
|---|---:|---:|---:|---:|---:|---:|
| Full | 3/26 | 3/26 | 4/26 | 4/26 | 6/26 | **7/26** |
| Full deep | 3/26 | 3/26 | 5/26 | 5/26 | 7/26 | **8/26** |
| Full + compact | 3/26 | 3/26 | 4/26 | 4/26 | 6/26 | **7/26** |

Full deep + VLM đạt8/26 (30,77%) ở lượt regression, so với Full không VLM6/26
(23,08%). **Không quy toàn bộ gain cho VLM**: Full deep CPU đã đạt7/26; phần
gain thêm của VLM trong chính arm này chỉ là1/26. Đây là mô tả thử nghiệm nhỏ,
không đủ để chọn default hoặc tuyên bố cải thiện khái quát.

### MRR@20

| Tập / arm | CPU | VLM có caption | VLM chỉ ảnh |
|---|---:|---:|---:|
| Dev / Full | 0,24242 | 0,30303 | 0,31313 |
| Dev / Full deep | 0,28687 | 0,30844 | 0,31710 |
| Dev / Full + compact | 0,37942 | 0,39394 | 0,40404 |
| Human / Full | 0,13138 | 0,14318 | 0,14464 |
| Human / Full deep | 0,13629 | 0,15403 | 0,15477 |
| Human / Full + compact | 0,12850 | 0,13747 | 0,13792 |

Image-only có MRR nhỉnh hơn trong cả6 nhóm, nhưng số query ít, correlated frames
và nhiều score lỗi/thiếu. Chưa đủ để bỏ caption ở production chỉ vì chênh lệch này.

### Query nào thực sự thay đổi?

- Dev Full:1 query known-frame top1 tốt lên,0 mất top1,10 không đổi.
- Human từng arm:1 query Hit@20 tốt lên,0 mất Hit@20,25 không đổi; **không query
  nào đổi trạng thái known-frame Hit@1**.
- Nhiều arms dùng chung cùng query/ảnh, không cộng các gain này thành các ca
  độc lập hoặc tuyên bố nhiều lần xác nhận cùng một cải thiện.

Các gain Hit@20 đáng chú ý là do **hạ điểm distractor**:

| Query / arm | Hạng CPU của known-positive | Hạng sau VLM image-only | VLM trực tiếp chấm ảnh đó? |
|---|---:|---:|---|
| `human_drive_05_event_1` / Full | 28 | 17 | **Không**, nằm ngoài top20 |
| `human_drive_05_event_2` / Full deep | 21 | 19 | **Không**, nằm ngoài top20 |
| `dev_support_v2_07` / Full deep | 29 | 14 | **Không**, nằm ngoài top20 |

Đây vẫn là gain retrieval ranking hợp lệ theo công thức đã khóa, nhưng không
phải bằng chứng VLM đã nhìn và xác nhận những ảnh đáp án mới vào top20.

## 4. Chuỗi: xếp lại path, chưa tăng số path được tìm thấy

| Tập / metric | CPU, cả3 arms | +VLM, cả3 arms và2 modes |
|---|---:|---:|
| Dev known-path Hit@1 | 0/4 | 0/4 |
| Dev known-path Hit@20 | 0/4 | 0/4 |
| Dev target video top1 | 4/4 | 4/4 |
| Human known-path Hit@1 | 1/10 | **2/10** |
| Human known-path Hit@5/20 | 2/10 | 2/10 |
| Human target video top1 | 6/10 | **7/10** |

Human `drive_12_sequence` là ca known-path được nâng hạng2→1, không có ca
known-path top1 tốt thành kém trong mẫu này. Path đã có sẵn từ CPU gồm
L09_V022/keyframes172→175; Qwen chấm8/10 ở cả hai mode. Full retrieval score
0,85149 được kết hợp thành0,89604, sau đó vượt distractor.

**VLM không tìm thêm path mới:** known-path Hit@20 vẫn2/10. Nó không thể sửa
video ngoài pool hoặc DP đã chọn frame khác. Tất cả result vẫn
`sequence_verified=false`/`temporal.verified=false`; không khẳng định hành động
liên tục hay quan hệ nhân quả chỉ từ các ảnh keyframe thưa.

Coverage path không đồng đều: Human Full/Deep mỗi mode có27/27 request scores
hợp lệ trên8/10 queries có path đủ điều kiện;2 queries không có complete path
để chấm vẫn giữ trong denominator. Compact24/25 hợp lệ,8/10 queries có score.
Dev mỗi arm chỉ2/4 queries có ít nhất1 valid path score.

## 5. Pair diagnostics: không nên dùng ngưỡng VLM làm hard gate lúc này

Ngưỡng8/10 được đặt trước, không chọn lại sau kết quả. Dùng cùng tập pair có
score hợp lệ ở **cả hai mode** để so caption công bằng; unknown labels bị bỏ
khỏi confusion matrix. Không so trực tiếp bảng này với metric support baseline
trên toàn bộ pair cũ vì subset coverage khác nhau.

| Tập / mode | Pair đủ score cả2 modes | Macro TPR | Macro FPR | Macro balanced accuracy |
|---|---:|---:|---:|---:|
| Dev / có caption | 37/39 | 45,45% | 0% | 72,73% |
| Dev / chỉ ảnh | 37/39 | 36,36% | 0% | 68,18% |
| Human / có caption | 127/130 | 27,56% | 0% | 63,78% |
| Human / chỉ ảnh | 127/130 | 27,24% | 0% | 63,62% |

Human paired-valid có40 positive,85 negative,2 unknown. Có caption nhận15/40
positive, chỉ ảnh12/40; cả hai chấp nhận0/85 negative ở ngưỡng8. Số này là micro
count, **không phải macro TPR** trong bảng. FPR0 trong85 negatives không chứng
minh FPR0 trên corpus. VLM bỏ sót nhiều positive ở ngưỡng này; không dùng điểm
>=8 làm điều kiện bắt buộc cho tất cả event hoặc sửa gate production theo mẫu.

Caption có ích cho một số positive (pair-level) nhưng image-only MRR nhỉnh hơn
(ranking-level). Hai kết quả không mâu thuẫn: binary acceptance và relative
ranking là nhiệm vụ khác nhau. Không chọn một mode làm “tốt hơn mọi mặt”.

## 6. Chi phí ghi nhận, không phải latency end-to-end

- Tổng recorded task durations:2.296,05s, khoảng**38,27 phút**; đã bao gồm
  thời gian các task invalid/missing nhưng không bao gồm mọi overhead notebook.
- Model load/download theo metadata:57,16s, ghi riêng.
- Median của task hợp lệ trên T4: ảnh có caption1,031s, ảnh-only0,981s;
  path có caption1,709s,path-only1,603s.
- Requests được dedup giữa các arms nên không cộng task seconds theo mỗi case
  để suy runtime toàn bộ lượt chạy. CPU/GPU ở hai máy, retrieval/index warmup,
  fetch ảnh và API planner không được tính thành production end-to-end latency.
- Chưa có per-task raw response nên không tách rõ chi phí invalid-score theo
  parse/OOM. Không đo lại phần này bằng thay cấu hình sau khi nhìn scores.

## 7. Quyết định kỹ thuật và bước tiếp theo

1. **Giữ nguyên production config, nhãn và ngưỡng hiện tại.** Không auto-promote
   compact/deep/image-only/joint-path VLM từ regression đã quan sát.
2. Qwen có tín hiệu hữu ích cho reranking khi evidence đã nằm trong danh sách,
   nhưng mức gain nhỏ và không chữa candidate/path recall. Ưu tiên cải thiện
   candidate coverage/đa dạng path trên development trước khi tăng ngân sách VLM.
3. Nếu thử tiếp, lập protocol development mới cho per-event top-K evidence và
   nhiều temporal path thay vì chỉ một path/video; chốt budget và đo cả recall,
   ranking,coverage,chi phí. Đây là hướng đề xuất, **chưa triển khai trong lượt này**.
4. Khi chọn được phương án trên development, cần holdout mới hoặc human
   adjudication độc lập để kiểm tra khái quát; không relabel các output sai theo
   model hoặc tái gọi human-v3 là untouched holdout.
5. Để đo cả LLM planner, người dùng cần cấu hình API trong ứng dụng rồi báo lại,
   **không gửi API key trong chat**. Lượt này chưa gọi planner; kế hoạch event đã
   duyệt và literal object entities không phải đầu ra LLM production.
6. **Không cần chạy Kaggle lại cho lượt đã hoàn tất này.** File kết quả đủ task,
   việc thiếu/invalid đã được xử lý fallback minh bạch.

## 8. Artifact, tái lập và kiểm tra

- Kết quả nhập: `benchmark_results/kaggle_vlm_results_20261103/`, có
  `import_audit.json` và nguyên bốn file từ ZIP.
- Replay306 cases/modes: `benchmark_results/full_pipeline_vlm_replay_20261103/`.
- `summary.json`: metric đủ query và paired-valid pair diagnostic.
- `analysis_v1.json`: transitions query tốt/kém/không đổi,coverage,timing,
  score distribution,pair rows và source hashes.
- Tool mới `tools/vlm_result_report.py` chỉ đọc artifacts, không load model.

```powershell
python -B -m tools.vlm_benchmark_packet replay --source-run benchmark_results/full_pipeline_ablation_20261102 --packet-root benchmark_results/full_pipeline_vlm_packet_20261102 --result-dir benchmark_results/kaggle_vlm_results_20261103 --output-dir benchmark_results/vlm_replay_NEW
python -B -m tools.vlm_result_report --source-run benchmark_results/full_pipeline_ablation_20261102 --packet-root benchmark_results/full_pipeline_vlm_packet_20261102 --result-dir benchmark_results/kaggle_vlm_results_20261103 --replay-dir benchmark_results/vlm_replay_NEW --output benchmark_results/vlm_replay_NEW/analysis_v1.json
```

Output cũ không bị ghi đè. Replay kiểm tra score range,ID set,code/model/prompt
fingerprints,private binding hash và original CPU case hashes. Không đọc ảnh
để tự adjudicate thêm vào nhãn đã freeze. Không commit/push/rebuild Milvus.

Kiểm thử cuối: **331 tests,317 đạt,14 skipped**, không failure/error. Lượt
targeted report/Kaggle scorer/packet **13/13 đạt**. Log publish/rebuild trong
unit tests là mock/thư mục tạm, không thay collection thật. Các score giả lập
trong unit tests không được trộn vào score Kaggle dùng cho báo cáo này.
