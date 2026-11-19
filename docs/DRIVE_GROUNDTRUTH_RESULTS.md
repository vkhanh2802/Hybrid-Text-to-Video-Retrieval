# Kết quả kiểm thử từ ảnh Drive — nhãn AI provisional

Run: `benchmark_results/groundtruth_eval_20261102_002150/`.
Nhãn: `benchmark_results/drive_groundtruth_20261102_seed20261102/frozen_labels_v1.json`.
Sampling/annotation protocol: [DRIVE_GROUNDTRUTH_EVALUATION.md](DRIVE_GROUNDTRUTH_EVALUATION.md).

## Kết luận ban đầu

**Chưa đạt mục tiêu tìm đúng frame/chuỗi một cách tin cậy.** Hệ thống đã được
kiểm tra bằng ảnh Drive thực, không chỉ bằng caption hoặc query tự đặt không
có nhãn. Nhưng mẫu nhỏ và do AI gán nhãn, chưa được human adjudicator xác nhận.
Không dùng kết quả này như accuracy đại diện toàn cuộc thi/corpus.

Không thay đổi runtime threshold, không sửa frozen labels sau retrieval, không
re-embed hoặc rebuild corpus. Chỉ dùng cached encoders CPU, không Qwen.

## 1. Tìm lại ảnh đã được xem

| Split | Query ảnh đơn | Known-target Hit@1 | Hit@5 | Hit@20 | MRR@20 |
|---|---:|---:|---:|---:|---:|
| Development | 4 | 0/4 | 1/4 | 1/4 | 0,1250 |
| Holdout | 3 | 1/3 | 1/3 | 1/3 | 0,3333 |
| Tổng | 7 | 1/7 | 2/7 | 2/7 | 0,2143 |

Rank đầu tiên của keyframe đã gán nhãn:

| Query ảnh | Target video | Split | Rank |
|---|---|---|---:|
| Boong thuyền chở cát | L03_V006 | dev | Không trong 300 kết quả lưu |
| Mũ đỏ/áo phản quang trong cây cỏ | L16_V028 | dev | 2 |
| Nam dẫn chương trình áo tím | L09_V011 | dev | Không trong 300 kết quả lưu |
| Tay đeo găng và túi mì | L08_V002 | dev | 81 |
| Nam dẫn chương trình áo hồng/đeo kính | L02_V015 | holdout | 1 |
| Nữ dẫn chương trình váy đỏ rượu vang | L10_V026 | holdout | Không trong 300 kết quả lưu |
| Hai người cạnh lửa và bụi cây | L17_V026 | holdout | 26 |

Đây là conservative **known-target** metrics. Một frame chưa gán nhãn có thể
đúng hoặc gần như giống anchor; không gọi mọi frame ngoài danh sách là sai.
Các frame presenter thường gần giống nhau ở nhiều thời điểm/video.

## 2. Chuỗi: video đúng, frame đúng và path đúng là ba việc khác nhau

| Phép đo | Development (4 chuỗi) | Holdout (3 chuỗi) |
|---|---:|---:|
| Temporal: target video có trong local pool 12 video | 4/4 | 0/3 |
| Temporal: top video đúng video mẫu | 4/4 | 0/3 |
| Baseline: first-event known-frame Hit@20 | 3/4 | 0/3 |
| Temporal: first-event known-frame Hit@20 | 1/4 | 0/3 |
| Temporal: known-path Hit@20 (tất cả event đúng frame đã gán nhãn) | 0/4 | 0/3 |

**0 known paths không chứng minh 0 path đúng về mặt nội dung.** Phạm vi nhãn
chỉ là các frame đã xem trong cửa sổ lấy mẫu. Temporal có thể chọn frame khác
cùng cảnh ở ngoài cửa sổ; những frame đó cần được adjudicate riêng, không được
tự động coi đúng chỉ vì caption giống hoặc sửa nhãn gốc để tăng metric.

Ví dụ dev:

- Thuyền/cát: prediction n132 → n134 → n136; nhãn ban đầu n133 → n134 → n135.
  n132 là góc nhìn bên ngoài thuyền, chưa hỗ trợ đầy đủ yêu cầu nhìn dọc boong
  chứa cát. n136 chưa được xem trong nhãn v1.
- Cây cỏ → nữ presenter hồng: prediction n305 → n318; nhãn n316 → n317/318.
  Cảnh n305 nằm ngoài cửa sổ nhãn v1, cần xem ảnh để phân biệt cùng cảnh hay sai.
- Xe bạc hỏng → presenter tím → đường ngập: prediction n114 → n121 → n124;
  nhãn n118/119 → n120/121 → n122. n114/n124 cần xem trực tiếp.
- Đóng gói mì: prediction đúng n48, n49, n52 nhưng event tay cầm túi mì n50
  là `unknown`; coverage 3/4. Không coi tìm đúng video là giải được chuỗi.

Không so trực tiếp score RRF với temporal score; ngân sách và đơn vị output
khác nhau. Hai query/task từ cùng video không là các mẫu độc lập.

## 3. Chẩn đoán từ development, không tune trên holdout

Raw visual cosine tại **4 anchor ảnh đơn dev đã được nhìn và gán nhãn**:
0,03461 / 0,17183 / 0,08722 / 0,06303 — cả 4 dưới visual gate thử nghiệm 0,25.
Raw semantic cosine tương ứng: 0,52818 / 0,35646 / 0,50559 / 0,46178.

Như vậy gate hiện tại bỏ qua visual support ở cả 4 positives này. Nhưng chỉ có
positives và chưa có negatives được adjudicate đầy đủ thì **chưa đủ chọn ngưỡng
mới hoặc báo FPR/F1**. Với sequence, cosine toàn query so với frame đầu không
được dùng làm event-level calibration.

Đã kiểm tra hai vấn đề tách biệt:

1. **Dev posthoc image adjudication:** lấy thêm n136, n305, n114, n124 từ Drive
   (và xem lại n132),
   không xem thêm holdout để tune và không sửa nhãn v1. Output:
   `benchmark_results/dev_path_review_20261102/`.
2. **Visual consistency audit:** dùng chính cached SigLIP, tính tạm image vector
   cho 4 dev anchor trên CPU, đối chiếu với NPY đang dùng. Không ghi vector mới
   vào corpus. Output: `benchmark_results/visual_consistency_dev_20261102/`.

### 3.1. Kết quả kiểm tra vector từ ảnh thật

| Dev image | Cosine image tính lại / stored | Cosine text / stored image |
|---|---:|---:|
| Thuyền/cát | 0,999930 | 0,034608 |
| Mũ đỏ/áo phản quang | 0,999710 | 0,171827 |
| Presenter tím | 0,999787 | 0,087219 |
| Tay/túi mì | 0,999743 | 0,063033 |

Hashes ảnh Drive khớp ảnh dùng gán nhãn. Các mẫu không cho thấy sai feature
model/row identity nghiêm trọng; text/image cosine thấp vẫn xảy ra khi tính lại
image vector từ bytes gốc. **Không có cơ sở re-embed toàn corpus từ bốn mẫu này.**
Không suy từ bốn mẫu rằng provenance/compatibility mọi vector đều đã được xác minh.

### 3.2. Xem trực tiếp top path development

AI review **posthoc, không blind**, chỉ top path 4 development queries:

| Dev query | Nhận xét sau xem ảnh | Toàn path phù hợp mô tả? |
|---|---|---|
| Thuyền/cát | n132 sai góc nhìn yêu cầu; n136 không thể hiện cận cảnh vũng nước | Không |
| Mũ đỏ/cây cỏ → presenter hồng | n305 đúng nội dung event dù nằm ngoài cửa sổ nhãn; n318 đúng | Có |
| Xe bạc hỏng → presenter tím → ngập đường | n114/n121/n124 đều đúng nội dung và thứ tự timestamp | Có |
| Đóng gói mì | n48/n49/n52 phù hợp; bước tay/túi mì vẫn unknown | Không đầy đủ |

Như vậy **2/4 top dev paths được AI review là phù hợp**, nhưng metric frozen
labels v1 vẫn giữ nguyên **0/4 known paths**. Đây là chẩn đoán thiếu nhãn trong
cửa sổ hẹp, không phải tăng accuracy nhờ đổi thuật toán; không gộp kết quả review
dev vào holdout hoặc tự cập nhật qrels v1. Judgments/hash ảnh được lưu tại
`benchmarks/drive_dev_path_adjudication_20261102.json`.

## 4. Hướng xử lý phụ thuộc kết quả

- Các vector ảnh tính lại khớp sát ở bốn mẫu: chưa cần re-embed; nếu phát hiện
  mismatch ở mẫu khác, xác minh model/preprocessing/row identity trước và trao
  đổi để chuyển việc nặng sang Kaggle, không tự chạy toàn corpus.
- Cần tập dev positives + negatives được xác nhận để calibrate
  visual/semantic support; không hạ threshold để tạo full paths bằng mọi giá.
- Holdout target videos đều bị loại trước local scoring: cần ablation candidate
  video pooling/budget, không thể kỳ vọng tune DP giải quyết video không có trong pool.
- Các mục đã xem trong holdout chỉ là đánh giá một lần; nếu dùng chúng để sửa
  query/heuristic thì phải coi là development và thu thập holdout mới.
