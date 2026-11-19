# Ground truth từ keyframe Drive và kiểm thử lại

## Phạm vi và nguồn nhãn

Đã lấy mẫu với seed **20261102**: chọn ngẫu nhiên 8 video từ scope mapping hiện
tại, rồi chọn ngẫu nhiên một anchor row/video; lấy thêm hai keyframe trước/sau.
Split được ấn định **theo video trước khi xem ảnh**: 4 development, 4 holdout.
Không chọn mẫu dựa trên kết quả retrieval hoặc caption hiện có.

Ảnh được truy xuất qua adapter `kis/local_images.py` sẵn có, sử dụng quyền
Google Drive **read-only**. Không thay đổi dữ liệu Drive. Bytes ảnh gốc chỉ ở RAM;
lưu contact sheets thu nhỏ và SHA-256 trong `benchmark_results/` (gitignored).
Không ghi credentials hoặc Drive file/folder IDs vào nhãn hay report.

Kết quả fetch: **35/40 ảnh**, 7/8 video. Cả 5 ảnh mẫu L18_V007 không truy xuất
được (`FileNotFoundError`). Không thay mẫu âm thầm; mẫu đó bị loại với lý do rõ
ràng. Vì vậy tập thực đánh giá có **4 dev videos + 3 holdout videos** và có
availability bias. Không suy rằng toàn corpus hoặc toàn Drive đều được phủ.

## Quan sát ảnh trước khi chạy retrieval

Đã đọc cả 8 contact sheets (một sheet ghi fetch failures). Nhãn từ ảnh thành công:

| Sample | Split | Video | Nội dung ảnh anchor |
|---|---|---|---|
| drive_01 | dev | L03_V006 | Boong thuyền chở cát, lốp xe, người áo phao cam, sông nước nâu |
| drive_02 | dev | L16_V028 | Người mũ đỏ/áo phản quang vàng trong cây cỏ tối |
| drive_03 | dev | L09_V011 | Nam dẫn chương trình áo tím/cà vạt sọc, nền skyline cam |
| drive_04 | dev | L08_V002 | Tay đeo găng cầm túi mì bạc xanh cạnh vắt mì vàng |
| drive_05 | holdout | L18_V007 | Không đọc được ảnh — không gán nhãn |
| drive_06 | holdout | L02_V015 | Nam dẫn chương trình đeo kính, áo hồng, bàn trắng |
| drive_07 | holdout | L10_V026 | Nữ dẫn chương trình váy đỏ rượu vang, màn hình vàng cam |
| drive_08 | holdout | L17_V026 | Hai người áo sáng màu cạnh lửa, bụi cây xanh và khói |

Tạo **14 query**: 7 ảnh đơn + 7 chuỗi shot, mỗi chuỗi từ các keyframe lân cận đã
xem. Không bịa chuyển động liên tục hoặc quan hệ nhân quả: ví dụ ảnh phóng sự
chuyển về người dẫn bản tin chỉ được gán là thứ tự chuyển cảnh. Nhãn không dùng
tên người hoặc nội dung chỉ suy ra từ caption, không dùng timestamp ticker làm
query shortcut.

**Đây là AI visual judgments, chưa được người dùng/human adjudicator xác nhận.**
Không coi là gold ground truth độc lập. Mẫu nhỏ, nhiều presenter frames lặp lại
trong corpus; split khác video không bảo đảm khác nội dung hoặc khác người dẫn.

## Freeze và reproducibility

- Observation source: `benchmarks/drive_observations_20261102.json`.
- Manifest/sheets: `benchmark_results/drive_groundtruth_20261102_seed20261102/`.
- Frozen labels: `frozen_labels_v1.json` trong cùng thư mục.
- SHA-256 nhãn trước evaluation:
  `ef339b14718e5d1a526aa2f2dd3b09c32b102fe99bd819ac3b2e0b96ab4dda7a`.
- Query, positive keyframe lists và split đã freeze trước retrieval; không sửa
  query/nhãn sau khi xem kết quả. Muốn sửa annotation cần version mới, ghi lý do.

## Lệnh tái lập

Không ghi đè thư mục/sample/nhãn cũ; chọn output version mới khi cần:

```powershell
python -B -m tools.drive_groundtruth --output-dir benchmark_results/drive_sample_new --seed 20261102
python -B -m tools.build_visual_labels --manifest benchmark_results/drive_sample_new/manifest.json --output benchmark_results/drive_sample_new/frozen_labels_v1.json
python -B -u -m tools.groundtruth_evaluation --labels benchmark_results/drive_sample_new/frozen_labels_v1.json
```

Runner chỉ dùng cache SigLIP/MiniLM trên CPU/offline; không VLM, planner API,
caption generation, feature extraction hoặc rebuild Milvus. Thiếu model cache
thì dừng. Thư mục output mới có `context.json`, từng query JSON, `summary.json`
và `retrieval.log`. Source/label hashes và live collection ID/row count được lưu.

## Metric và điều không được suy diễn

- Ảnh đơn: **known-target Hit@1/5/20, MRR@20** trên những keyframe đã nhìn và
  đánh dấu positive. Đây là kiểm thử tìm lại item đã biết, không phải exhaustive
  relevance. Một frame ngoài nhãn có thể vẫn đúng hoặc gần như giống hệt.
- Chuỗi: baseline/temporal **first-event known-target metrics** không đủ chứng
  minh đúng chuỗi. Metric riêng `known_path_hit_at_1/5/20` chỉ đạt khi đúng video
  và frame được chọn cho **từng event** nằm trong positive list tương ứng đã xem.
  Dùng đúng video hoặc `coverage=1` không tự động được tính đúng path.
- Nếu một path hợp lệ dùng frame chưa được xem/gán nhãn, known-path metric có
  thể bỏ sót; không gọi đó là false positive đã được xác minh.
- Tách ảnh đơn/chuỗi, dev/holdout. Hai task từ cùng video không là hai video
  độc lập. Không dùng mọi query như 14 quan sát độc lập để khẳng định ý nghĩa thống kê.
- Lưu raw visual/semantic similarity ở anchor để chẩn đoán ngưỡng. Với chuỗi,
  raw similarity này dùng **query toàn chuỗi so với frame đầu**, không được dùng
  như event-level positive calibration.
- **Chưa thay đổi ngưỡng runtime.** Tập nhỏ này chưa có tập negatives được gán
  nhãn đủ tin cậy để tune specificity/FPR; không chọn threshold từ holdout.

## Trạng thái thực thi

Sampling, đọc ảnh, freeze 14 query và lượt đánh giá corpus thật đã hoàn tất.
Kết quả/giới hạn: [DRIVE_GROUNDTRUTH_RESULTS.md](DRIVE_GROUNDTRUTH_RESULTS.md).
Chưa thay đổi ngưỡng hoặc chạy model nặng trên Kaggle.
