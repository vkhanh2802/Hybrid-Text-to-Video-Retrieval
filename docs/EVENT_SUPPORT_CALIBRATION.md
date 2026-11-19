# Hiệu chỉnh event support từ ảnh Drive — bộ v2

## 1. Đã thực hiện

- Lấy mẫu mới với seed `2026110202`, loại **40 video đã có trong các tập nhãn/
  manifest benchmark cũ** trước khi sampling.
- Chọn ngẫu nhiên 24 video, random anchor/video, lấy +/-2 keyframe lân cận;
  split theo video trước khi xem ảnh: 12 dev, 12 holdout.
- Truy xuất Drive read-only: **110/120 ảnh**. Hai video L14_V020 (dev) và
  L14_V007 (holdout) không truy xuất được cả 5 ảnh. Không thay mẫu hoặc tạo nhãn.
- Đã xem 24 contact sheets, trong đó hai sheet chỉ chứa fetch failures.
- Tạo **22 câu event**, mỗi câu gắn positive/negative/unknown đã nhìn trực tiếp:

| Split | Query event | Positive pairs | Negative pairs | Unknown pairs |
|---|---:|---:|---:|---:|
| Development | 11 | 16 | 22 | 1 |
| Holdout mới | 11 | 13 | 22 | 1 |

Nhãn là **AI visual judgments, chưa human verified**, không lấy caption hoặc
retrieval score để viết query/nhãn. Các cặp negative có những điều kiện quan
trọng không khớp: sai áo/cà vạt, sai hình trên màn hình, sai góc cận/toàn, cầm
cốc ở miệng thay vì đặt trên bàn, screenshot bản đồ thay vì tàu thật, v.v.
Độ khó không đồng nhất; có cả near negatives và negative khác cảnh rõ ràng.

Đây là **event-to-frame classification benchmark**, không phải bộ 22 query
temporal nhiều bước hay bài đo KIS end-to-end. Một câu mô tả nhiều chi tiết vẫn
là một điều kiện ảnh có thể quan sát; không suy chuyển động liên tục từ ảnh.

## 2. Cơ chế hiệu chỉnh

Support gate runtime hiện tại là:

```text
support(event, frame) =
    visual_cosine >= visual_threshold
    OR caption_semantic_cosine >= semantic_threshold
```

Lỗi có thể xảy ra theo hai hướng:

- Visual threshold quá cao: ảnh đúng bị bỏ qua dù query và image embeddings
  cùng model. Bốn ảnh dev cũ đã cho thấy cosine thấp nhưng vector tính lại khớp.
- Semantic threshold quá thấp: caption cùng chủ đề nhưng sai chi tiết cũng có
  thể làm frame được coi là có support.

Không đổi model, công thức local score, candidate policy hay DP trong thí nghiệm
này. Chỉ chọn hai **ngưỡng của OR gate**, trên cặp **câu event cụ thể / frame đã
gán nhãn**, không dùng toàn query nhiều bước so với một ảnh làm nhãn event.

Grid đã cố định trong `kis/support_calibration.py` trước khi chấm:

- Visual: 0,000 tới 0,300, bước 0,025.
- Semantic: 0,25 tới 0,80, bước 0,05.
- Chỉ chọn trên dev: macro FPR <= 0,10 và macro TPR >= 0,50, rồi tối đa
  macro balanced accuracy. Nếu không có ứng viên đạt thì báo `no_safe_candidate`,
  không chọn ngưỡng làm mọi frame bị từ chối chỉ để đạt FPR thấp.
- Tie-break: FPR thấp hơn, gần baseline hơn, cuối cùng threshold để quyết định
  deterministic. Không thay selection objective sau khi xem holdout.
- Chỉ query có cả positive và negative mới vào macro averages. Unknown được bỏ
  khỏi metric, không bị đổi thành negative. Query nhiều frame lặp lại không có
  trọng số cao hơn chỉ vì nhiều pair.

## 3. Freeze nhãn và profile trước holdout

- Observation source: `benchmarks/event_support_v2_observations_20261102.json`.
- Manifest/sheets/frozen labels:
  `benchmark_results/event_support_v2_seed2026110202/`.
- Frozen label SHA-256:
  `2f808990bdcc04cd3fe1f72b1b1da4a615b68376e77ff1f00a8cc3b9133b454e`.
- Development calibration:
  `benchmark_results/event_support_v2_dev_calibration_20261102/calibration.json`.
- Profile SHA-256 được holdout lưu để đối chiếu:
  `7d586acdf840743dcdb2ba4623e587517393bca402150ca0c5fcba92d9615d93`.
- Holdout evaluation:
  `benchmark_results/event_support_v2_holdout_evaluation_20261102/calibration.json`.

Training chỉ gọi encoder cho dev queries, không đọc/chọn theo holdout scores.
Holdout phải nhận profile đã ghi ra từ dev, kiểm tra label hash và split trước
khi chạy. Source hashes được lưu; thư mục output/nhãn không ghi đè run cũ.

Profile được chọn và giữ nguyên khi test holdout:

```text
Baseline:     visual >= 0.25  OR semantic >= 0.45
Experimental: visual >= 0.075 OR semantic >= 0.60
```

## 4. Kết quả pair-level

Các tỷ lệ bên dưới là **macro average theo 11 query của từng split**, không
phải accuracy toàn corpus hoặc Hit@K. TPR là tỷ lệ nhận positive; FPR là tỷ lệ
chấp nhận nhầm negative. Balanced accuracy = (TPR + 1 - FPR) / 2.

| Split | Profile | Macro TPR ↑ | Macro FPR ↓ | Macro balanced accuracy ↑ |
|---|---|---:|---:|---:|
| Dev | Baseline | 63,64% | 22,73% | 70,45% |
| Dev | Experimental | 68,18% | 9,09% | 79,55% |
| Holdout mới | Baseline | 72,73% | 22,73% | 75,00% |
| Holdout mới | Experimental | 81,82% | 13,64% | 84,09% |

Số pair để kiểm tra không nhầm macro với micro:

| Split/profile | TP | TN | FP | FN |
|---|---:|---:|---:|---:|
| Dev baseline | 10 | 17 | 5 | 6 |
| Dev experimental | 12 | 20 | 2 | 4 |
| Holdout baseline | 10 | 17 | 5 | 3 |
| Holdout experimental | 11 | 19 | 3 | 2 |

Mỗi split có 1 unknown pair không tính vào các số trên. Không báo significance
hoặc CI từ các pair như thể chúng độc lập: nhiều ảnh cùng video và một số ảnh
negative được dùng cho nhiều query.

### Trade-offs còn tồn tại trên holdout

- Query cleanroom nhận ra positive mới nhưng cũng nhận nhầm worker mặc đồ trắng
  ở dây chuyền thực phẩm: màu/đồ bảo hộ giống nhau chưa đủ xác nhận bối cảnh.
- Query hai presenter vẫn có thể nhận nhầm một presenter đơn.
- Một ảnh tàu từ trên cao được baseline nhận nhưng profile mới bỏ sót.
- FPR holdout **13,64% vẫn vượt mục tiêu 10% dùng để sàng lọc trên development**.
  Không tự tune lại trên holdout hoặc coi ngưỡng đã bảo đảm FPR <= 10% khi deploy.

## 5. Quyết định áp dụng

**Profile mới vẫn là experimental, chưa đổi `config/pipeline.json` hoặc default
`TemporalConfig`.** Pair-level cải thiện là tín hiệu tốt, chưa đủ để khẳng định
ranking/temporal path tốt hơn.

Trong local pool có thể có hàng nghìn frame chưa gán nhãn; FPR pair-level 13,6%
không thể chuyển thẳng thành expected false-full-path rate. Hạ visual gate có
thể cho thêm distractors vượt gate và tăng số chuỗi “đầy đủ” sai.

Đã chạy regression trên **4 chuỗi dev cũ**, giữ đúng saved pool/seed frames,
query, scoring weights và DP; chỉ đổi support thresholds. Hai profile dùng chung
raw visual/semantic score table. Đây là regression diagnostic, **không phải new
holdout** và không dùng để sửa lại threshold đã chọn.

Output: `benchmark_results/event_support_v2_sequence_regression_20261102/`.

| Regression diagnostic | Baseline | Experimental |
|---|---:|---:|
| Top video đúng video mẫu | 4/4 | **3/4** |
| Tổng full-support paths trong các pool | 6 | 3 |
| Known-path hits@20 theo nhãn v1 cũ | 0/4 | 0/4 |

Baseline tái lập đúng top của source run ở 4/4 query; không có lỗi artifact.
Query `drive_02_sequence` (người trong cây cỏ tối → presenter hồng) đổi top từ
L16_V028 sang L10_V019. Chưa xem ảnh mới của L10_V019 trong regression này nên
không coi mọi nội dung của video đó là đã bị adjudicate sai; nhưng đây là
**regression tìm lại target video đã biết**.

Giảm tổng số full-support paths từ6 xuống3 không tự chứng minh false-positive
path giảm: không phải mọi path được gán nhãn đầy đủ. Known-path v1 vẫn là metric
bảo thủ vì cửa sổ nhãn hẹp; không đổi nhãn hoặc tune lại trên holdout để sửa kết quả.

**Quyết định cuối: không promote profile mới làm mặc định.** Pair-level holdout
cải thiện nhưng chưa bảo toàn ranking chuỗi, và FPR holdout còn cao. Cần support
phụ thuộc loại event/điều kiện cụ thể hoặc evidence verifier tốt hơn; không áp
một cặp ngưỡng chung chỉ vì balanced accuracy trên cặp ảnh tăng.

Bước tiếp theo đã có bộ điều kiện trang phục hẹp, opt-in, giữ nguyên baseline:
[EVENT_CONDITION_EVIDENCE.md](EVENT_CONDITION_EVIDENCE.md). Kết quả mới chỉ là
development/regression, không dùng lại holdout ở báo cáo này để chọn quy tắc và
chưa bật mặc định.

## 6. Cách tái lập

```powershell
python -B -m tools.event_support_calibration freeze --manifest benchmark_results/event_support_v2_seed2026110202/manifest.json --observations benchmarks/event_support_v2_observations_20261102.json --output benchmark_results/event_support_v2_seed2026110202/frozen_support_labels_v1.json
python -B -u -m tools.event_support_calibration train --labels benchmark_results/event_support_v2_seed2026110202/frozen_support_labels_v1.json --output-dir benchmark_results/event_support_dev_new
python -B -u -m tools.event_support_calibration holdout --labels benchmark_results/event_support_v2_seed2026110202/frozen_support_labels_v1.json --calibration benchmark_results/event_support_dev_new/calibration.json --output-dir benchmark_results/event_support_holdout_new
```

Lệnh freeze sẽ từ chối ghi đè nhãn đã tồn tại; chọn output version mới khi thực
sự cần re-annotation, không sửa qrels sau khi nhìn scores. Các file raw scores
được lưu riêng theo split. Không tự động áp dụng threshold cho web service.

## 7. Tài nguyên và kiểm thử

- Dùng existing Drive reader read-only và pinned SigLIP/MiniLM cache trên CPU.
- Không Qwen/VLM, không tải model mới, không sinh caption/OCR hoặc re-embed corpus.
- Không ghi vào Milvus/Drive, không sửa mapping, features hoặc index gốc.
- `tests/test_support_calibration.py`: 8 tests bao phủ unknown, macro weighting,
  no-safe-candidate, cấm holdout trong selection, cấm cross-split frame pairs và
  đối chiếu calibration gate với support mask runtime (kể cả missing/NaN/inf).
- Full regression tại thời điểm thêm calibration: **265 tests, 251 đạt,
  14 skipped, không failure/error**. Log publish/staging trong unit tests là
  mock và thư mục tạm, không phải rebuild corpus thật.
- `tools/support_profile_replay.py` bổ sung sau lượt full suite đó. Lượt xác minh
  cuối chạy **33 tests liên quan, tất cả đạt**, gồm calibration, replay guard,
  sampler và temporal. Replay guard từ chối profile không phải development hoặc
  không có selected thresholds trước khi load model. Kết quả chạy thật của
  replay được báo riêng, không suy unit tests thành bằng chứng chất lượng.
- Hash nhãn/profile được kiểm tra lại sau holdout và regression, vẫn khớp các
  giá trị đã freeze. `config/pipeline.json` giữ nguyên visual0,25/semantic0,45.

## 8. Giới hạn cần giữ khi đọc kết quả

Đây là tập nhỏ, AI-authored, chưa human adjudicated; nhiều query là các thuộc
tính cảnh/đồ vật chứ không kiểm tra thao tác nấu ăn hay temporal reasoning đầy
đủ. Split theo video không loại mọi near-duplicate, cùng người dẫn hoặc cùng
nội dung phát lại. Hai video không đọc được ảnh gây availability bias. Ground
truth không có phạm vi relevance toàn corpus. Không coi kết quả này là bảo đảm
cho cuộc thi hoặc tác dụng của một model mới.
