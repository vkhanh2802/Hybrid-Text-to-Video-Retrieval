# Thử nghiệm multi-path — protocol chốt trước replay

## Mục tiêu và ranh giới

Kiểm tra path đúng có đang bị bộ căn chỉnh một-path/video loại bỏ trước VLM
hay không. Không thay production `align_events`, gate, pool, điểm, index, nhãn,
caption hay model. Không tải ảnh, gọi retrieval/API hoặc chạy encoder/VLM.
Development trước; human-v3 chỉ làm observed regression sau development,
không dùng để chọn K/budget/overlap. Không có independent holdout mới.

## Cấu hình cố định

- Giữ raw matrices và tối đa 1.500 keyframes/video, 12 video/query hiện có.
- Gate visual 0,25 / semantic 0,45; score weights 0,6 / 0,4, semantic guard off.
- Mỗi event lấy tối đa **32 frame hỗ trợ có điểm cao nhất**, tie theo index,
  cộng tối đa một frame của path production nếu nằm ngoài top32. Không dùng
  qrels để đưa known-positive vào danh sách ứng viên.
- DAG dùng node (event, frame). Cạnh chỉ tiến event và timestamp, gap >0 và
  <=60s giữa hai frame hỗ trợ liên tiếp; có thể skip event. Gap penalty0,001.
- Giữ top32 prefix/node và top32 path toàn DAG, objective support count trước,
  tổng score trừ gap penalty sau. Đây là k-best chính xác **trong DAG đã cắt**,
  không khẳng định top32 của mọi frame trong video/corpus.
- Luôn ghim nguyên path production ở vị trí đầu mỗi video, kể cả tie. Các
  path khác được dedup theo cặp event/frame, không đổi công thức score/coverage.
- Trả tối đa **4 path/video**: top4 theo objective, hoặc greedy diversity chọn
  path có <=50% cặp event/frame trùng với mọi path đã giữ, ưu tiên cùng support
  count tối đa, sau đó điền phần còn thiếu theo objective. Diversity chỉ nằm
  trong search set cố định, không bảo đảm đa dạng toàn cục.

## Các arms được chốt

| Arm | Path/video | Tổng path/query | Mục đích |
|---|---:|---:|---|
| single_12 | 1 production | <=12 | Baseline phải khớp kết quả đã lưu |
| top4_equal12 | <=4 | <=12 | Cùng trần số path đầu ra |
| diverse4_equal12 | <=4 | <=12 | Cùng trần, có diversity |
| top4_expanded48 | <=4 | <=48 | Coverage khi tăng ngân sách path |
| diverse4_expanded48 | <=4 | <=48 | Coverage tăng ngân sách + diversity |

Mọi arm giữ nguyên pool video và frame đã chấm. Xếp toàn cục theo score cũ,
video ID, path index. Với equal12, cắt sau khi xếp; nhiều path của một video có
thể đẩy video khác khỏi danh sách. Đây là trade-off cần đo, không che giấu bằng
cách ép video đích vào pool. Số path thực tế có thể nhỏ hơn trần; cùng trần
không có nghĩa cùng CPU hay số unique videos. Expanded48 không phải equal-budget.

## Phép đo và attribution

- Known-path Hit@1/5/20 và có path đúng trong toàn bộ candidate set (<=48).
- Có path đúng trong 5 complete paths đầu đủ điều kiện cho VLM tương lai;
  **chưa chạy VLM**, không reuse score của bộ ảnh/path khác từ packet trước.
- Target video top1 chỉ là metric phụ, không thay known-path đúng từng event.
- Số path/complete paths, unique videos và CPU replay time/work counts. Timing
  không phải latency end-to-end; ghim path gọi lại DP baseline ở mỗi selector.
- Attribution tuần tự: ngoài pool → scoring failure → không có known ordered
  path trong cached frames (thiếu frame/gap/thứ tự) → bị gate → đã được giữ →
  known supported path bị mất do một-path alignment. Nếu known path khả thi
  trong full cache nhưng không trong DAG cắt, báo riêng qua feasibility.
- Feasibility dùng qrels **sau** bước tạo path mù nhãn, chỉ để chẩn đoán upper
  bound; không dùng oracle này làm kết quả retrieval. Qrels thưa: kết quả ngoài
  nhãn vẫn unknown, không tự gọi là false positive.

## Reproducibility và quyết định

`prepare` lưu protocol, snapshot code, hash input/labels/raw matrices trước khi
`run` chấm. `run` từ chối code/input thay đổi hoặc baseline không khớp bản cũ;
không ghi đè output. Regression phải tham chiếu development đã hoàn tất với
đúng code/settings. Không retune sau khi nhìn kết quả regression.

**Quyết định chốt trước: giữ production nguyên trạng**, kể cả mẫu nhỏ tăng.
Nếu development không cho thấy known-supported path bị mất ở single alignment,
multi-path chưa giải quyết nút thắt trên mẫu này; không chuyển ngay sang GPU.
Nếu cần VLM trên path mới, dừng và thông báo trước khi chuẩn bị lượt Kaggle mới.

Development: 4 dev sequence cũ, nhãn AI quan sát ảnh, không phải toàn bộ human
verified. Human-v3: 10 sequence đã quan sát, nhãn human đã khóa; không được gọi
là validation độc lập hoặc bằng chứng ý nghĩa thống kê.
