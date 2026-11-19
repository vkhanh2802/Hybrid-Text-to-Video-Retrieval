# Kết quả cross-encoder CPU — 04/11/2026

## Kết luận

**Đã hoàn tất download, smoke thật và153 cached CPU cases** (45 development,
108 observed human regression). Runtime hoạt động, scores hợp lệ và fallback
được bảo toàn, nhưng gain ranking nhỏ và có loss rõ ràng. **Giữ text reranker
off ở production; không chạy VLM mới, không retune theo regression.**

Human Full giữ Hit@1=3/26, Hit@5 tăng4→5/26 nhưng có2 query được top1 và2 query
mất top1. Deep mất1 known-frame Hit@20. Compact cải thiện một path có sẵn trên
regression nhưng đã giảm mạnh trên development, nên không được chọn theo kết
quả human. Không có independent holdout mới để chứng minh khái quát.

## Phạm vi

Đã được người dùng cho phép tải checkpoint và chạy inference CPU. Model:
`cross-encoder/ms-marco-MiniLM-L6-v2`, revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`.

Không chạy GPU/VLM, không gọi retrieval/LLM/Drive, không sinh lại caption hoặc
rebuild index. Production giữ `text_reranker.mode=off` trong lượt đo.
Protocol: [TEXT_RERANKER_CPU_PROTOCOL_20261104.md](TEXT_RERANKER_CPU_PROTOCOL_20261104.md).

## Smoke thật: đạt

- Model nạp safetensors từ cache, CPU FP32; CUDA không được khởi tạo.
- 22.713.601 parameters;90.854.404 bytes dữ liệu parameter, không phải RSS/peak RAM.
- Ba synthetic query/caption pairs trả logits9,15745 /-11,11899 /9,06569.
  Chấm lại cùng input cho cùng kết quả; batch16 toàn bộ finite.
- Đây là raw logits, không phải thang0–10 hoặc xác suất; không áp ngưỡng8/10
  của phép thử Qwen VLM trước đây cho cross-encoder này.
- Thiếu caption giữ slot/score; lỗi injected giữ fallback. Đây là kiểm tra giả
  lập lỗi có chủ đích, không có lỗi checkpoint thật trong smoke.
- Cold model load + batch3 đầu:26,170s; warm batch3:0,084s; warm batch16:0,293s.
  Torch runtime đã khởi tạo trước cold timer; không bao gồm download/startup đầy đủ.
- CPU4 threads/interop1, Python3.11.15, torch2.7.1+cu118,
  transformers5.15.0 trên Windows. Torch build có CUDA không có nghĩa đã dùng GPU.
- Synthetic scores chỉ xác minh runtime, **không phải metric chất lượng retrieval**.
  Smoke có thời điểm chạy song song unit tests; không suy latency production từ nó.

Artifacts: `benchmark_results/text_reranker_smoke_20261104/`.

## Thiết kế development đã khóa

15 query units (11 ảnh,4 chuỗi) ×3 retrieval arms =45 cases, dùng candidate/plan
đã lưu từ `full_pipeline_ablation_20261102`. Cấu hình cố định pool100,
batch16, max_pairs1200, max_chunks4, max_length512; không đổi sau khi xem scores.

Baseline được tính lại và kiểm tra khớp; qrels/case fingerprints, model files,
source và caption inputs được kiểm tra trước/sau. Caption inline giữ nguyên,
temporal lookup đóng băng từ artifact có hash lịch sử khớp. Đọc historical human
context chỉ lấy fingerprint caption, không dùng regression outcomes để chọn cấu hình.

Không memoize scores giữa arms. Case timing loại caption preparation/retrieval/
planner; case đầu gồm model load. Đây không phải latency toàn bộ ứng dụng.

Dev dùng qrels AI quan sát ảnh, cỡ mẫu nhỏ và đã quan sát trước đây. Ba arms
dùng chung query units, không coi45 cases là45 query độc lập. Qrels thưa nên
không tự gán kết quả ngoài nhãn thành false positives. Caption score không xác
nhận pixels hoặc chuỗi hành động.

## Development: có gain cục bộ nhưng không ổn định

### Ảnh đơn — 11 query/arm

| Arm | Hit@1 trước → sau | Hit@5 trước → sau | Hit@20 trước → sau | MRR@20 trước → sau |
|---|---:|---:|---:|---:|
| Full | 2/11 → **3/11** | 3/11 → 3/11 | 5/11 → 5/11 | 0,24242 → **0,28788** |
| Full deep | 3/11 → 3/11 | 3/11 → 3/11 | 5/11 → 5/11 | 0,28687 → 0,28717 |
| Full + compact | 4/11 → **2/11** | 4/11 → **3/11** | 6/11 → **5/11** | 0,37942 → **0,24272** |

Không chỉ nhìn mean: Full có1 query tăng Hit@1,0 mất,10 không đổi; deep có1
tăng/1 mất/9 không đổi; compact0 tăng/2 mất/9 không đổi. MRR Full có1 query tốt
lên và2 kém đi: một gain lớn có thể che nhiều loss nhỏ hơn.

Những thay đổi first-known-positive rank:

| Query | Full | Full deep | Full + compact |
|---|---:|---:|---:|
| `dev_support_v2_01` | 15→64 | 18→72 | 16→66 |
| `dev_support_v2_06` | 2→15 | 1→17 | 1→17 |
| `dev_support_v2_07` | **88→1** | **29→1** | 1→2 |

Gain query07 Full là ảnh được cross-encoder trực tiếp chấm trong pool100,
không phải ảnh unscored tăng hạng do distractor bị hạ. Tuy nhiên model chỉ nhìn
caption; không nói cross-encoder đã nhìn pixels và xác nhận ảnh. Các loss01/06
không được relabel thành negative để phù hợp với model.

### Chuỗi — 4 query/arm

Mọi arm: known-path Hit@1/5/20 vẫn **0/4**, target-video top1 vẫn **4/4**.
Chỉ17 complete candidate paths được chấm text (Full6,deep6,compact5);121 partial
paths giữ nguyên slot. Không có path đúng sẵn để text rerank tạo thêm recall.

### Coverage và chi phí dev

- 45 cases chạy xong, không thiếu query; baseline metrics/fingerprints khớp.
- 3.041/3.041 pair score hợp lệ,0 batch errors. Đây là **pair references**, không
  phải3.041 frame độc lập: cùng query/caption có thể lặp giữa các arms.
- Ảnh:3.001/3.300 top-pool candidate references có caption/score;299 missing text
  giữ slot/score;6.269 candidate references ngoài pool100 không bị text xếp lại.
- Temporal:40 event/caption pair references cho17 complete paths; không đổi gate,
  frame/path, coverage hoặc `verified=false`.
- Tổng recorded rerank time≈75,89s cho45 cases. Case đầu Full có model load,
  mất18,064s. Không memoize scores giữa arms.
- 32 image cases còn lại: median≈1,769s/case, P95 nearest-rank≈2,189s/case.
  Các con số này chỉ là CPU text stage trên input đã chuẩn bị, không phải API
  end-to-end hoặc latency app mặc định (benchmark giới hạn4 CPU threads).

Artifacts: `benchmark_results/text_reranker_dev_20261104/`; protocol SHA-256
`737ba6a198e67aaf0bc0db485ff00e1729cdce0dc699e72004e818426abb49c7`.

### Quyết định sau dev, trước regression

Gain Full đủ để kiểm tra **observed regression** bằng code/settings không đổi,
nhưng compact regression rõ, deep gần như hòa. Không chọn lại một arm/budget,
không bật production, không chạy VLM. Cả3 arms đều giữ nguyên khi kiểm tra human-v3;
không lấy một nhánh thắng dev rồi tuyên bố đã được validation độc lập.

## Human-v3: observed regression, không dùng để chọn tham số

### Ảnh đơn — 26 query/arm

| Arm | Hit@1 trước → sau | Hit@5 trước → sau | Hit@20 trước → sau | MRR@20 trước → sau |
|---|---:|---:|---:|---:|
| Full | 3/26 → 3/26 | 4/26 → **5/26** | 6/26 → 6/26 | 0,13138 → 0,14712 |
| Full deep | 3/26 → 3/26 | 5/26 → 5/26 | 7/26 → **6/26** | 0,13629 → 0,14712 |
| Full + compact | 3/26 → 3/26 | 4/26 → **5/26** | 6/26 → 6/26 | 0,12850 → 0,14712 |

Ở mỗi arm có2 query tăng top1 /2 mất top1 /22 không đổi, không phải “không có
query nào bị ảnh hưởng” chỉ vì tỷ lệ top1 trung bình bằng nhau. MRR mỗi arm:
2 tốt lên /4 kém đi /20 không đổi; các gain lớn bù các loss nhưng chưa ổn định.

Ví dụ first-known-positive rank của **Full**:

| Query | Hạng trước → sau | Diễn giải |
|---|---:|---|
| `human_drive_05_event_1` | **28→1** | Gain lớn, được text chấm trực tiếp |
| `human_drive_05_event_2` | **56→1** | Gain lớn, được text chấm trực tiếp |
| `human_drive_01_event_3` | **1→8** | Mất top1 và top5 |
| `human_drive_03_event_3` | **1→2** | Mất top1 |
| `human_drive_08_event_2` | **11→56** | Mất known-frame Hit@20 |
| `human_drive_09_event_3` | **8→57** | Mất known-frame Hit@20 |

Hai gain top1 thuộc cùng sample/video `drive_05`, không phải hai xác nhận độc
lập trên corpus. Không xem các frames ngoài qrels là false positives chắc chắn,
không sửa human judgment hoặc bản dịch theo điểm model.

### Chuỗi — 10 query/arm

| Arm | Known-path Hit@1 trước → sau | Hit@20 trước → sau | Target-video top1 trước → sau |
|---|---:|---:|---:|
| Full | 1/10 → 1/10 | 2/10 → 2/10 | 6/10 → **5/10** |
| Full deep | 1/10 → 1/10 | 2/10 → 2/10 | 6/10 → **5/10** |
| Full + compact | 1/10 → **2/10** | 2/10 → 2/10 | 6/10 → 6/10 |

Compact nâng known path đã có của `human_drive_12_sequence` từ2→1; đồng thời
cả3 arms mất target-video top1 ở `human_drive_09_sequence`. Ca09 vốn chưa có
known path đúng trong candidate set, nên video-level loss không phải mất một
known-path hit. Không thêm path mới và không xác nhận chuyển động/thứ tự từ text.
Mọi frame/path/coverage/timestamp và `verified=false` giữ nguyên.

Tất cả bảng trên là **baseline CPU vs CPU + text**, không phải so với kết quả
CPU + Qwen VLM của lượt trước. Không reuse GPU scores cho shortlist mới.

### Coverage, chi phí và audit regression

- 108/108 cases,36 query units;7.479/7.479 pair references hợp lệ,0 batch errors.
- 7.246 image candidate references scored,554 missing text giữ nguyên;
  104 complete paths scored bằng233 event/caption pairs;239 partial paths không chấm.
- Tổng CPU text-stage time≈192,40s, gồm first-case cold≈20,074s.
- 77 warm image cases: median≈1,941s, P95 nearest-rank≈3,841s; không phải latency
  retrieval/API/planner hay measured production SLA.
- Audit23.154 candidate references: score retrieval và temporal evidence không
  đổi;15.804 unscored references giữ đúng slot/score. Không lọc case thiếu caption
  hoặc partial path để nâng tỷ lệ thành công.

Artifacts: `benchmark_results/text_reranker_regression_20261104/`; protocol hash
`dcc3329b421ef802fffd3f6ce673a2d24d1910d86738460f642a09ad629e3855`.

## Quyết định và bước sau

1. **Giữ mode off**, không promote model/policy hiện tại từ gain trung bình nhỏ.
   Module hoạt động về runtime/integration, nhưng kết quả chất lượng mixed.
2. Không chỉnh pool/weight/aggregation để cứu những query đã thấy, không lấy
   kết quả human để chọn riêng compact cho chuỗi rồi gọi là validation.
3. Không chạy VLM mới trong lượt này. Không có bằng chứng text reranker đã cải
   thiện shortlist đủ nhất quán để ưu tiên thêm ngân sách GPU.
4. Nếu tiếp tục outline, ưu tiên evidence recall theo event và OCR độc lập cho
   query chữ/bảng/nguyên liệu; không bắt mọi query dùng OCR/ASR, không hạ ngưỡng
   global theo nhãn regression. Phương án ranking mới cần protocol dev riêng và
   kiểm chứng độc lập mới trước khi bật mặc định.
5. Người dùng **không cần chạy Kaggle hoặc gán nhãn thêm cho lượt đã hoàn tất**.
   Checkpoint CPU đã sẵn trong cache cho những phép thử được cho phép tiếp theo.

## Tái lập và bảo toàn artifacts

Tool mới `tools/text_reranker_benchmark.py` có ba lệnh: smoke, prepare, run.
Prepare lưu code snapshot/model/input hashes trước inference; run từ chối
input/code đổi hoặc output đã tồn tại. Không sửa fingerprint run cũ cho qua.

Ví dụ tái lập, **không cần chạy lại cho lượt đã hoàn tất**:

```powershell
python -B -m tools.text_reranker_benchmark smoke --output-dir benchmark_results/text_smoke_NEW
python -B -m tools.text_reranker_benchmark prepare --smoke-dir benchmark_results/text_smoke_NEW --output-dir benchmark_results/text_dev_NEW
python -B -m tools.text_reranker_benchmark run --output-dir benchmark_results/text_dev_NEW
# Chỉ sau khi đọc dev, nếu thực hiện observed regression với cấu hình không đổi:
python -B -m tools.text_reranker_benchmark prepare --role observed_regression --development-run benchmark_results/text_dev_NEW --smoke-dir benchmark_results/text_smoke_NEW --output-dir benchmark_results/text_regression_NEW
python -B -m tools.text_reranker_benchmark run --output-dir benchmark_results/text_regression_NEW
```

Mỗi benchmark run có protocol/source_snapshot/prepared_inputs/cases/summary.
`prepared_inputs.json` chứa nhãn và caption nội bộ cho replay cục bộ, **không
đưa file này vào gói scoring mù nhãn/Kaggle**. Model chỉ nhận query/caption,
không nhận judgments, target video hoặc event_relevant_keyframes làm input.

Audit dev trên toàn9.707 candidate references: score retrieval gốc và temporal
evidence không đổi;6.689 unscored references giữ đúng slot và score. Caption
thiếu/partial path không biến thành negative hoặc score0.

Hai benchmark tổng cộng10.520 pair references hợp lệ/10.520 request,0 batch
errors; recorded text-stage time≈268,29s (4,47 phút), không tính smoke, download,
preparation hay toàn bộ quá trình làm việc. Audit từng run lưu trong `audit_v1.json`
kèm hashes các result cases. Quyết định/hash tổng hợp:
`benchmark_results/text_reranker_decision_20261104.json`.

Checksum cuối xác nhận human annotations, frozen human labels và
`dist/vlm-results.zip` giữ nguyên. Runtime hiện vẫn mode off, ngưỡng temporal
0,25/0,45 không đổi. Không commit/push, không sinh lại ZIP đã công bố.

Kiểm thử hiện tại: **367 tests,353 đạt,14 skipped**, không failure/error;
targeted text module/benchmark21/21 đạt. Unit tests không nạp weights thật;
inference thật nằm riêng trong smoke và benchmark artifacts. Test logs
publish/rebuild là mocks/thư mục tạm, không ghi collection Milvus thật.
