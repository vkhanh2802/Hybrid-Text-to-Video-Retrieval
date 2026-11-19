# Bằng chứng điều kiện từng event trên keyframe (thử nghiệm)

## Mục tiêu và phạm vi

Similarity có thể nhận đúng chủ đề nhưng sai chi tiết. Ví dụ: query yêu cầu
áo xanh trơn nhưng caption của frame ghi rõ áo trắng kẻ ô. Thay vì đổi một
cặp ngưỡng chung cho mọi event, module mới phát hiện **mâu thuẫn tường minh
ở một số thuộc tính cụ thể** và giới hạn khả năng caption tự xác nhận event.

Module `kis/condition_evidence.py` chỉ xử lý các action tiếng Anh đã có trong
planner. Không đổi schema planner, không gọi LLM, VLM hoặc video encoder. Đơn vị
bằng chứng vẫn là **keyframe**; `video_id` và timestamp chỉ phục vụ ghép chuỗi.

Phiên bản hiện tại: `garment_attributes_v2`.

- Hỗ trợ thuộc tính nằm ngay trước danh từ: màu cơ bản, họa tiết trơn/kẻ ô/
  sọc/chấm trên `shirt`, `tie`/`necktie`, `dress`, `jacket`, `coat`.
- Chuẩn hóa một số từ đồng nghĩa như `gray/grey`, `checked/checkered`,
  `solid/plain`. Sáng/tối cùng màu cơ bản không bị kết luận mâu thuẫn.
- Không suy ra họa tiết trơn chỉ vì caption không nhắc họa tiết.
- Chưa hỗ trợ trạng thái cốc, hành động nấu ăn, góc cận/toàn, số presenter,
  OCR hoặc nội dung màn hình. Các điều kiện này vẫn là `unknown` với module này.
- Chỉ đọc trường `caption` gốc; không lấy `retrieval_text` ghép keywords/OCR
  để suy màu hoặc thuộc tính vật thể.

## Ba trạng thái không phải nhãn đúng/sai của ảnh

| Trạng thái | Ý nghĩa |
|---|---|
| `consistent` | Các thuộc tính mà bộ quy tắc hỗ trợ và trích ra được đều khớp caption |
| `contradicted` | Có ít nhất một thuộc tính cùng loại, cùng trang phục được mô tả rõ nhưng khác yêu cầu |
| `unknown` | Không đủ chi tiết hoặc không gán thuộc tính an toàn |

`consistent` **không có nghĩa toàn bộ event đúng**, và `contradicted` không có
nghĩa ảnh chắc chắn sai: caption là văn bản sinh tự động, có thể bỏ sót hoặc
hallucinate. Output ghi rõ `source=generated_caption`, `pixel_verified=false`;
kết quả temporal vẫn luôn `verified=false`.

Caption thiếu, không có thuộc tính cần thiết, mô tả nhiều người, lặp nhiều
trang phục cùng loại, nhiều màu, phủ định hoặc cách diễn đạt mơ hồ sẽ được
giữ `unknown`. Đây là bộ lọc heuristic, **không phải giải quyết đầy đủ subject
binding bằng NLP**. Ví dụ màu áo trong ảnh trên một màn hình có thể thuộc người
khác; gặp ngữ cảnh đó trong phạm vi câu liên quan thì bộ quy tắc abstain.

Scope giữ toàn bộ ngữ cảnh trước đó và câu cuối có nhắc trang phục; đoạn OCR/
bình luận tin tức ở câu sau không tự xóa một thuộc tính đã mô tả rõ. Chặn phủ
định, nhiều người và một số từ chỉ nhóm vẫn áp dụng trên toàn caption. Nếu cùng
một người được nhắc bằng nhiều danh từ, quy tắc có thể quá thận trọng và abstain.

## Cơ chế tác động lên support

Giữ nguyên ngưỡng production `visual=0.25`, `semantic=0.45`, trọng số điểm local
`0.6/0.4`, seed hits, candidate pool và thuật toán DP.

```text
baseline_support = visual >= 0.25 OR semantic >= 0.45

guarded_support = baseline_support AND (
    condition_status != contradicted OR visual >= 0.25
)
```

Vì vậy:

- Mâu thuẫn chỉ có thể loại **support dựa riêng vào caption**.
- Support đã có từ visual vượt ngưỡng baseline được giữ nguyên, không bị caption
  phủ quyết. “Visual vượt ngưỡng” vẫn là heuristic, không phải pixel verification.
- `unknown` không làm frame mất support; `consistent` không tạo thêm support.
- Không thay raw cosine, điểm kết hợp, trọng số hoặc DP. Nếu visual vượt ngưỡng,
  điểm kết hợp vẫn có thành phần semantic cũ dù caption có mâu thuẫn. Đây là
  giới hạn chủ đích của thử nghiệm, chưa phải mô hình hợp nhất evidence hoàn chỉnh.

Ba chế độ trong `TemporalConfig.condition_evidence`:

| Giá trị | Tác dụng |
|---|---|
| `off` (mặc định) | Không đọc caption thêm cho module; giữ hành vi support cũ |
| `diagnostics` | Chỉ ghi trạng thái và số support *sẽ* bị loại; không đổi ranking |
| `semantic_guard` | Áp dụng công thức trên, ghi số support thực sự bị loại |

Có thể tạo cấu hình thử trong Python bằng
`TemporalConfig(condition_evidence="semantic_guard")`, hoặc thêm trường
`"condition_evidence": "diagnostics"` vào phần `temporal` của một bản config thử.
Không có web toggle mới và không tự bật trên service đang chạy.

`LocalEventScorer.score()` giữ nguyên tuple trả về để tương thích công cụ cũ.
Thống kê tổng hợp nằm ở `stats.condition_evidence`; mỗi event được chọn có thêm
`condition_evidence` khi bật chế độ. Custom scorer vẫn tự chịu trách nhiệm về
support mask của nó: phần xuất chẩn đoán **không tự áp lại guard** lên mask đó.

## Đánh giá development, không dùng lại holdout để chọn quy tắc

`tools/condition_evidence_evaluation.py`:

1. Từ chối score rows không phải development, đối chiếu đủ identity, event text,
   judgment và hash ảnh với nhãn đã đóng băng.
2. Dùng raw scores development cũ, đọc caption JSON hiện có; phần pair không
   cần FAISS, encoder, ảnh Drive hoặc kết nối Milvus.
3. Có tùy chọn replay 4 chuỗi dev cũ với pool/seed cố định, raw evidence dùng
   chung giữa baseline và guard. Lần đầu chỉ dùng encoder CPU đã cache offline;
   lưu score tables để các lần replay sau không load model nữa.
4. Output version mới, không ghi đè nhãn/run cũ; lưu source hashes, source
   snapshot (từ lượt v2), caption hashes và provenance của evidence cache.

Đã có một vòng sửa trên development: bản `garment_attributes_v1` trả `unknown`
cho cả 39 cặp vì chặn từ như `screen`/`appears` trên toàn đoạn caption. Bản v2
thu hẹp scope các từ này về ngữ cảnh trang phục như mô tả ở trên. **Đây là
development iteration, không phải đánh giá độc lập hoặc preregistered holdout.**

Các nhãn vẫn là AI quan sát ảnh Drive, **chưa human verified**. Các holdout cũ
đã được xem ở thí nghiệm trước nên không được coi là untouched và không được
dùng để chọn quy tắc lần này. Tại thời điểm lượt development này chưa có
holdout mới; lượt human holdout tiếp theo đã hoàn tất và **không cho gain**,
xem [HUMAN_HOLDOUT_V3_RESULTS.md](HUMAN_HOLDOUT_V3_RESULTS.md).

Artifacts:

- Lượt đầu: `benchmark_results/condition_evidence_v1_dev_20261102/`.
- Lượt scope v2: `benchmark_results/condition_evidence_v2_dev_20261102/`.
- `dev_pairs.json`: raw score, trạng thái điều kiện, prediction của hai chế độ.
- `report.json`, `sequence_summary.json`, `drive_*_sequence.json`: metric và
  ranking; file `*_evidence.json` là bảng evidence development dùng lại được.

### Kết quả lượt v2

11 event queries development, **16 positive + 22 negative + 1 unknown pair**:

| Metric (macro theo query) | Baseline | Semantic guard |
|---|---:|---:|
| TPR: nhận positive | 63,64% | 63,64% |
| FPR: chấp nhận nhầm negative | 22,73% | **18,18%** |
| Balanced accuracy | 70,45% | **72,73%** |
| TP / TN / FP / FN (số pair) | 10 / 17 / 5 / 6 | 10 / 18 / 4 / 6 |

Unknown judgment không được tính vào confusion matrix. Trạng thái điều kiện
trên 39 pair: **1 consistent, 2 contradicted, 36 unknown**. Do đó coverage
còn rất hẹp; không suy việc giảm 1 FP thành cải thiện ổn định hoặc có ý nghĩa
thống kê trên toàn corpus.

Pair thay đổi là `support_v2_04`, L10_V002/keyframe40: query yêu cầu áo xanh
**trơn**, caption ghi áo **kẻ ô**; visual0,02981 không vượt baseline, semantic
0,59664 vượt baseline nên trước đó caption tự tạo support. Guard bỏ support
này; nhãn nhìn ảnh development đã đóng băng là negative. Không cần suy màu áo
khi caption chưa ghi màu, và không coi caption là ground truth để sửa nhãn.

Regression trên 4 chuỗi dev cũ, 12 candidate videos/query, dùng lại raw tables
để không load model ở lượt v2:

| Metric | Baseline | Semantic guard |
|---|---:|---:|
| Top video đúng video mẫu | 4/4 | 4/4 |
| Full-support paths trong các pool | 6 | 6 |
| Known-path hits@20 theo nhãn v1 | 0/4 | 0/4 |

Baseline tái lập đúng source top4/4; không lỗi artifact. Có **7 event–frame
support bị loại** trong các pool; chưa có nhãn cho tất cả chúng nên không gọi
đó là “7 false positives đã sửa”. Top path không đổi ở cả 4 query. Kết quả ở
các hạng thấp hơn có thay đổi với `drive_03_sequence`; không nói toàn bộ ranking
giữ nguyên. Chưa có bằng chứng chọn đúng keyframe/path tốt hơn.

### Kiểm thử và an toàn dữ liệu

- Full suite: **285 tests, 271 đạt, 14 skipped**, không failure/error.
- Kiểm thử thêm: thuộc tính thiếu, nhiều người, nhiều màu, phủ định, caption
  nhiều câu, không lấy màu nền/cà vạt làm màu áo, không suy trạng thái/góc ảnh,
  không loại visual support vượt ngưỡng, không sửa score matrices.
- Kiểm tra cả `off`/`diagnostics`/`semantic_guard`, caption/keyframe identity,
  tương thích custom scorer và JSON không chứa NaN.
- Replay cache có test cấm gọi bước load model; evaluator từ chối holdout rows
  trước khi đọc nhãn/caption và phát hiện score rows không khớp nhãn đóng băng.
- Source hashes của lượt v2 khớp source hiện tại tại thời điểm xác minh; hash
  hai bộ nhãn đóng băng vẫn nguyên. `git diff --check` đạt (chỉ cảnh báo line
  endings trên các file có sẵn trong working tree).
- Không tải model, sinh caption, re-embed corpus hoặc ghi Drive/Milvus. Log
  staging/publish trong full unit tests là mock/thư mục tạm, không rebuild index
  thật. Không commit/push và không hoàn nguyên các thay đổi có trước.

### Tái lập không load model, không tải dữ liệu

Chọn thư mục output mới vì công cụ từ chối ghi đè:

```powershell
python -B -u -m tools.condition_evidence_evaluation --dev-scores benchmark_results/event_support_v2_dev_calibration_20261102/scores.json --labels benchmark_results/event_support_v2_seed2026110202/frozen_support_labels_v1.json --output-dir benchmark_results/condition_evidence_replay_new --sequence-labels benchmark_results/drive_groundtruth_20261102_seed20261102/frozen_labels_v1.json --source-run benchmark_results/groundtruth_eval_20261102_002150 --evidence-run benchmark_results/condition_evidence_v1_dev_20261102
```

Bỏ cả ba tùy chọn cuối nếu chỉ chấm pair. Bỏ riêng `--evidence-run` sẽ tính
score chuỗi bằng encoder CPU offline; không dùng lệnh đó nếu máy không có cache.

## Quyết định áp dụng

Mặc định tiếp tục **`off`**. Bảo toàn top video trong một regression nhỏ chưa
chứng minh keyframe/path đúng hơn. Cần kiểm tra được tác dụng ở cấp frame/path
và có một holdout thực sự mới trước khi đề nghị bật guard mặc định. Nếu công
việc tiếp theo cần model nặng hoặc sinh lại caption, phải dừng và báo người dùng
để chạy trên Kaggle; module này không làm các bước đó.

Đã hoàn tất bộ mới do người dùng gán nhãn, khóa bản dịch/nhãn trước scoring:
[HUMAN_HOLDOUT_V3_RESULTS.md](HUMAN_HOLDOUT_V3_RESULTS.md). Đây là runner human
holdout riêng, không bỏ guard development-only của công cụ cũ. Metric chính
không đổi,128/130 pair ở trạng thái condition unknown; tiếp tục mặc định `off`.
