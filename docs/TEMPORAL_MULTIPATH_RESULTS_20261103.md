# Kết quả thử nghiệm nhiều temporal path — 03/11/2026

## Kết luận

**Đã triển khai và chạy xong multi-path thử nghiệm. Chưa thấy tăng known-path
coverage trên 4 development sequences hoặc 10 human regression sequences.**
Giữ nguyên production; **không cần chạy Kaggle/VLM mới cho phương án này**.

Trên human-v3, 6/10 chuỗi thiếu hỗ trợ ở frame đúng, 2/10 mất video ở pool,
2/10 đã có path đúng. Không có ca known-supported path bị single-path loại bỏ
trong cached pool của bộ regression này. Đây là kết luận giới hạn trong nhãn
đã biết, không chứng minh mọi path ngoài nhãn đều sai.

Protocol chốt trước replay: [TEMPORAL_MULTIPATH_PROTOCOL_20261103.md](TEMPORAL_MULTIPATH_PROTOCOL_20261103.md).
Development được chạy/đọc trước regression; không thay settings sau kết quả.

## 1. Đã thêm gì và cơ chế

- `tools/temporal_multipath.py`: bộ căn chỉnh k-best DAG thử nghiệm, **không
  được production import**. Mỗi event giữ top32 frame được gate hỗ trợ, cộng
  frame của path production nếu cần; xét top32 path, trả tối đa4 path/video.
- Ghim nguyên path production đầu tiên mỗi video; giữ timestamp tăng nghiêm
  ngặt, gap <=60s, gap penalty0,001, abstention và support-count-first objective.
- Hai cách chọn: top4, hoặc greedy diverse4 ưu tiên <=50% cặp event/frame trùng
  nhau trong nhóm support count tối đa, rồi điền theo objective.
- Giữ nguyên score, pool <=12 video và raw frame matrices <=1.500/video.
- Hai ngân sách đầu ra: <=12 path/query để đối chiếu cùng trần với baseline,
  và <=48 để đo lợi ích khi tăng ngân sách. Cùng trần không đồng nghĩa cùng
  CPU, cùng số path thực tế hoặc cùng số unique videos.
- `tools/temporal_multipath_experiment.py`: prepare/freeze rồi replay; lưu
  hashes, source snapshot, attribution, ranking, costs. Từ chối input/code đổi
  sau freeze hoặc baseline khác kết quả đã lưu.

Không đọc ảnh/caption mới, không tải model, không gọi retrieval/API, không
đổi nhãn, config, Milvus hoặc bộ packet VLM đã công bố. Đây là fixed-plan
CPU replay, không phải phép đo LLM planner hay end-to-end pipeline mới.

## 2. Kết quả ranking và candidate coverage

**Tất cả số dưới đây là CPU, chưa cộng VLM.** Không so Hit@1=1/10 ở đây với
2/10 của lượt **+VLM trước** rồi gọi là regression do code mới.

| Tập / phương án | Known-path Hit@1 | Hit@5 | Hit@20 | Có known-path trong toàn candidate set | Có trong 5 complete paths đầu cho VLM |
|---|---:|---:|---:|---:|---:|
| Dev / single | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| Dev / top4 hoặc diverse4, trần12 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| Dev / top4 hoặc diverse4, trần48 | 0/4 | 0/4 | 0/4 | 0/4 | 0/4 |
| Human / single | 1/10 | 2/10 | 2/10 | 2/10 | 2/10 |
| Human / top4 hoặc diverse4, trần12 | 1/10 | 2/10 | 2/10 | 2/10 | 2/10 |
| Human / top4 hoặc diverse4, trần48 | 1/10 | 2/10 | 2/10 | 2/10 | 2/10 |

Top4 và diverse4 chọn một số path khác nhau nhưng cùng metric trên mẫu này.
Mỗi experimental arm: dev 0 tốt lên / 0 kém đi / 4 không đổi về Hit@20;
human 0 / 0 / 10. Target-video top1 vẫn4/4 dev và6/10 human ở mọi arm.
**Đúng video không đồng nghĩa đúng frame cho từng event.**

## 3. Lỗi nằm ở đâu?

| Attribution trong fixed cache | Dev | Human regression |
|---|---:|---:|
| Video đích ngoài pool | 0 | 2 |
| Known ordered path có frame nhưng bị support gate chặn | 3 | 6 |
| Known supported path khả thi nhưng single alignment không giữ | 1 | 0 |
| Known path đã được single alignment giữ | 0 | 2 |
| Scoring failure | 0 | 0 |

Feasibility dùng qrels **sau khi tạo candidate mù nhãn** để chẩn đoán, không
đưa frame đúng vào output. Trên các target có cache, known ordered path đều
có sẵn trong raw frame table; các ca gate failure không phải lỗi thứ tự/gap
hay thiếu frame trong table. Không suy được hỗ trợ cho2 video ngoài pool.

### Development: một ca thật sự bị mất do chọn một path

`drive_03_sequence`, video `L09_V011`:

- Known-positive sets: `[118,119] → [120,121] → [122]`.
- Single alignment chọn `114 → 121 → 124`, không khớp qrels đã biết.
- Known path vẫn khả thi sau gate và sau top32-frame cap.
- Chẩn đoán hậu nghiệm riêng trên dev: 9×14×7=882 tổ hợp supported frames,
  trong đó54 complete paths đúng thứ tự/gap. Known path tốt nhất
  `118 → 121 → 122` ở **hạng25** theo objective; path `119 → 121 → 122` hạng35.
- Path hạng25 nằm trong top32 search nhưng không được selector top4/diverse4
  giữ. **Đa dạng đơn thuần chưa đưa bằng chứng đúng lên đủ cao** trong budget4.

Lưu tại `benchmark_results/temporal_multipath_dev_20261103/dev03_posthoc_complete_rank_v1.json`.
Đây là diagnostic dùng nhãn, không phải retrieval gain hoặc VLM verification.
Không tăng K lên25 sau khi thấy con số này rồi tuyên bố phương pháp đã được
validation; không dùng kết quả này để chỉnh human regression.

### Human regression: không có thêm path đúng khả thi để multi-path cứu

- Ngoài pool: `drive_04_sequence`, `drive_07_sequence`.
- Bị gate: `drive_01`, `drive_03`, `drive_06`, `drive_08`, `drive_09`, `drive_10`.
- Đã giữ path đúng: `drive_05`, `drive_12`.

Vì gate/pool không đổi, tăng số path không thể tạo known-supported path cho
8 ca đầu. Điều này giải thích vì sao candidate coverage đứng yên ở2/10.
Không kết luận những path khác mà người dùng chưa gán nhãn là false positive.

## 4. Chi phí và trade-off

| Tập / phương án | Tổng output paths | Tổng complete paths | Unique videos trung bình/query |
|---|---:|---:|---:|
| Dev / single | 46 | 6 | 11,50 |
| Dev / top4 hoặc diverse4, trần12 | 48 | 12 | **4,75** |
| Dev / top4 hoặc diverse4, trần48 | 174 | 12 | 11,50 |
| Human / single | 114 | 35 | 11,40 |
| Human / top4 hoặc diverse4, trần12 | 120 | 67 | **4,70** |
| Human / top4 hoặc diverse4, trần48 | 422 | 108 | 11,40 |

**Cùng trần12 path làm giảm mạnh độ phủ video**: nhiều path điểm cao của một
video chiếm các vị trí của video khác. Mẫu này chưa có loss known-path Hit@K,
nhưng không thể coi việc giảm unique videos là miễn phí với qrels thưa.

Expanded48 giữ video coverage nhưng số output human tăng114→422, complete
paths35→108, **không thêm known-path đúng**. Complete chỉ có nghĩa đủ event
vượt heuristic gate, không chứng minh nội dung mọi điều kiện là đúng.

Nếu vẫn chấm tối đa5 complete paths/query bằng VLM, số request/path trước
dedup tăng dev6→10, human27→36, nhưng query có known-path trong danh sách
đủ điều kiện vẫn0/4 và2/10. Đây là **ước tính request từ candidate list**, chưa
chạy GPU, không phải chi phí đo được hay kết quả VLM mới.

Phần aligner trong replay rất nhỏ vì gate hiện tại thưa: tổng measured CPU
time top4 khoảng0,016s dev /0,048s human; diverse4 khoảng0,014s /0,042s.
Đây là một lần micro-timing, loại file I/O, evidence reconstruction và pipeline
startup; không so tốc độ hai selector từ chênh lệch này, không gọi là request
latency. Trường hợp support dày có thể tốn nhiều hơn đáng kể.

## 5. Quyết định và hướng tiếp theo

1. **Không bật multi-path production hoặc tăng ngân sách VLM từ lượt này.**
   Code để riêng cho thí nghiệm; các mặc định hiện hành không đổi.
2. Ưu tiên phát triển **per-event evidence recall trước gate/DP** và candidate
   video coverage. Chẩn đoán gate failure không phải đề nghị hạ ngưỡng toàn
   cục: phương án hạ ngưỡng trước đây đã có regression, vẫn không được promote.
3. Nếu thử cơ chế đưa evidence yếu vào một nhánh candidate riêng, phải tách
   candidate chưa xác nhận khỏi event được coi là supported, lập protocol dev
   mới và đo trade-off; không gắn nhãn positive theo caption/model.
4. Path diversity có thể nghiên cứu tiếp trên development có nhiều known
   supported alternatives, nhưng chưa có cơ sở tăng K vô hạn hoặc chuyển ngay
   sang heavy VLM để bù recall chưa đạt.
5. Human-v3 đã quan sát, chỉ regression. Muốn chọn và xác nhận một phương án
   mới phải có nhãn/holdout độc lập mới; không gọi lại bộ hiện tại là untouched.

**Người dùng chưa cần chạy Kaggle hay gán nhãn thêm cho lượt đã hoàn tất này.**

## 6. Artifacts và tái lập

- Development: `benchmark_results/temporal_multipath_dev_20261103/`.
- Regression: `benchmark_results/temporal_multipath_regression_20261103/`.
- Mỗi run có `protocol.json`, `source_snapshot.json`, `cases/`, `summary.json`.
- Protocol hashes: dev `07cb2a287a5563d1e3d9e5e100ef680928da3744a83e1860f34145916c6e82d6`;
  regression `605a2d6273ab3b0cd76bb9014daf6210d19af75de55e2a314f253dea18578202`.

```powershell
python -B -m tools.temporal_multipath_experiment prepare --role development --output-dir benchmark_results/multipath_dev_NEW
python -B -m tools.temporal_multipath_experiment run --output-dir benchmark_results/multipath_dev_NEW
python -B -m tools.temporal_multipath_experiment prepare --role observed_regression --development-run benchmark_results/multipath_dev_NEW --output-dir benchmark_results/multipath_regression_NEW
python -B -m tools.temporal_multipath_experiment run --output-dir benchmark_results/multipath_regression_NEW
```

Run từ chối ghi đè output; protocol chứa đường dẫn workspace hiện tại, cần giữ
nguyên file hoặc tạo protocol mới có hash đúng khi chuyển máy. Không thay
code/input giữa prepare và run; protocol không phải chữ ký mật mã chống sửa
đồng thời mọi artifact, mà là kiểm tra provenance/reproducibility cục bộ.

## 7. Kiểm thử và bảo toàn dữ liệu

- **340 tests: 326 đạt,14 skipped**, không failure/error.
- Targeted temporal/multipath/replay:32/32 đạt;9 tests mới gồm exhaustive tiny
  DAG (60 ma trận ngẫu nhiên), exact pinned top1/ties, abstention, equal timestamps,
  max gap, dedup, budget/validation, alternative path, attribution, hash/source
  mismatch và kiểm tra đổi qrels không đổi candidate paths.
- Baseline khớp kết quả cũ ở cả14 queries; không scoring failure.
- Kiểm tra lại frozen source hashes của full-pipeline benchmark trước, raw
  inputs và nhãn/annotations human: **không thay đổi**.
- Mọi temporal result vẫn `verified=false`; không dùng complete support để
  khẳng định chuyển động liên tục/quan hệ nhân quả.
- Unit test logs publish/rebuild/model là mock/temp, không chạy model nặng hoặc
  ghi Milvus thật; không commit/push/rebuild/re-embed trong lượt này.
