# Kết quả pilot bước 1 — corpus thật

Ngày tổng hợp: **02/11/2026**. Run:
`benchmark_results/temporal_pilot_20261101_235958/`.

## 1. Kết luận

**Temporal pipeline đã chạy end-to-end qua Milvus và local rescoring, nhưng
chưa có bằng chứng tăng chất lượng tìm kiếm.** Không coi top result hoặc
`retrieved_complete` là đáp án đúng. Các query chưa có ground truth; không báo
Hit@K, MRR hay accuracy và không hiệu chỉnh ngưỡng theo chính ba ví dụ này.

Hai vấn đề nổi bật: support gate visual pilot cao so với phân bố cosine quan
sát được; semantic caption có thể khớp chủ đề nhưng không đủ chứng minh các
chi tiết/hành động. DP đảm bảo thứ tự các frame được chọn, không biến semantic
similarity thành bằng chứng hành động đã xảy ra.

## 2. Điều kiện chạy

- Milvus: 138.796 vectors, collection ID `470054560685313683`, metadata model
  khớp cấu hình. Caption index: 130.996 vectors.
- SigLIP/MiniLM pinned cache, **CPU**, offline; không gọi LLM planner, Qwen,
  không fetch ảnh và không chạy caption/feature generation hoặc index rebuild.
- Nạp caption/model/runtime: **56,13 giây** trong lượt thành công.
- Queries/events cố định từ `benchmarks/temporal_pilot.json`, là mô tả rút gọn
  dựa trên ví dụ người dùng, không phải bộ query đã được gán nhãn video/frame.
- Baseline: 100 hits/stream/variant, RRF, không VLM; temporal: 50 hits/stream/event,
  12 candidate videos rồi local rescore. **Không đồng ngân sách/đơn vị output**.
- Lượt đầu `temporal_pilot_20261101_235722` bị timeout 120 giây khi nạp model,
  không có query results; không gộp vào bảng thời gian bên dưới.

## 3. Kết quả vận hành, không phải accuracy

| Query | Top video baseline | Top video temporal | Bước có support ở top temporal | Baseline | Temporal |
|---|---|---|---|---:|---:|
| Trứng → cắt đậu hũ → khuấy súp | L03_V026 | L18_V008 | 1/3, partial | 8,91 s | 7,07 s |
| Học sinh/bảng lớp → hành lang/cầu thang | L18_V007 | L17_V024 | 2/2, chưa xác minh | 4,27 s | 4,77 s |
| Ví dụ → giải thích → công thức tiếng Anh | L12_V004 | L12_V004 | 2/3, partial | 4,31 s | 8,26 s |

Ba query đều `temporal.status=applied`, không có local artifact failure. Mỗi query
rescore 12 video; lần lượt 3.180 / 2.785 / 2.999 frame rows. Không video nào bị
subsample bởi giới hạn 1.500 frame/video trong run này. Candidate video cap vẫn
có thể bỏ đáp án; không có ground truth để đo candidate recall.

Thời gian chỉ đo một lần, luân phiên thứ tự chạy. Không kết luận temporal nhanh
hơn/chậm hơn nói chung từ ba số này; chưa có P50/P95 hoặc đo cold/warm lặp lại.

## 4. Phân tích theo query

### 4.1. Nấu súp

- Top temporal: L18_V008, frame 11839, **473,56 giây**, chỉ support bước khuấy.
- Bước đổ trứng và cắt đậu hũ đều `unknown`; đây là **một gợi ý ảnh**, chưa phải
  chuỗi nấu ăn đúng yêu cầu.
- Cùng video top, unordered coverage=2/3 nhưng hai event cùng chọn frame 11839.
  Ordered DP chỉ giữ được 1/3, không dùng cùng frame để hoàn thiện chuỗi.
- Đảo thứ tự cho coverage 2/3 trong video này; không coi kết quả đảo là đáp án.
- Không candidate video nào đạt 3/3 theo gate hiện tại. Chưa biết video đúng có
  trong pool, thiếu keyframe hay caption/embedding không mô tả được thao tác.

### 4.2. Học sinh/hành lang

- Top temporal: L17_V024, **642,10 → 649,20 giây**, frame 19263 → 19476.
- Caption frame đầu mô tả học sinh **ngồi trong lớp**, bảng tên `MÀI - HẢI`, chưa
  hỗ trợ yêu cầu “đứng xếp hàng với bảng số lớp”. Đây là **dấu hiệu false positive
  từ semantic evidence**, chưa phải phán quyết sau kiểm tra pixels.
- Trong 12 candidate videos: 7 video có full support ở order gốc, 5 video vẫn
  full support khi đảo thứ tự; top video thuộc cả hai nhóm.
- Không thể coi coverage=1 là đã tìm đúng sequence. Cần đọc ảnh/bảng và gán nhãn
  điều kiện đứng/xếp hàng/chữ trên bảng, thay vì chỉ dựa vào caption similarity.

### 4.3. Bài giảng tiếng Anh

- Top temporal giữ L12_V004, **212,08 → 216,52 giây**, frame 5302 → 5413.
- Support bước 1 và 3, bước giải thích `unknown`.
- Unordered chọn cùng frame 5413 cho cả ba event; ordered DP không chấp nhận
  việc dùng một timestamp để mô tả cả quá trình thay đổi bảng.
- Caption frame 5413 mô tả **phương trình toán/vector/ma trận**, chưa chứng minh
  nội dung là ví dụ/công thức ngữ pháp tiếng Anh. Chưa đọc ảnh/OCR trong pilot.
- Không candidate video nào đạt 3/3 theo gate hiện tại.

## 5. Visual gate và kiểm tra nhất quán vector

Ngưỡng visual pilot: **0,25**. Cosine cao nhất trong các visual seed lists:

| Query | Max theo từng event |
|---|---|
| Nấu súp | 0,09265 / 0,09875 / 0,08452 |
| Học sinh | 0,12534 / 0,10990 |
| Tiếng Anh | 0,12035 / 0,12005 / 0,10235 |

Toàn bộ maxima này dưới 0,25. Các matched event ở **top result của cả ba query**
đều qua ngưỡng semantic 0,45 nhưng không qua visual 0,25. Vì vậy nhận định
“full support” hiện có thể chỉ do caption semantic; thiếu caption ở video
visual-only cũng là một rủi ro khi dùng ngưỡng chưa calibrated này.

Đã đọc trực tiếp embedding Milvus và đối chiếu NPY ở visual seed đầu mỗi query:

| Video/frame | Local row | Cosine live/local |
|---|---:|---:|
| L21_V017 / 1560 | 16 | ≈1,000000 |
| L12_V028 / 31642 | 262 | ≈1,000000 |
| L15_V002 / 8670 | 71 | ≈1,000000 |

Ba mẫu này **không cho thấy lệch giữa vector local và live Milvus**, nhưng không
chứng minh provenance/model correctness toàn corpus, cũng không giải thích hết
chất lượng image-text similarity. Chưa đổi model hoặc tự hạ threshold.

## 6. Quyết định tiếp theo

1. Giữ module ở mức **pilot**, không tuyên bố cải thiện top-1 hoặc dùng full
   support như verified confidence.
2. Gán nhãn một tập video/frame đúng và negative gần giống, tách development/
   holdout; kiểm tra event evidence bằng ảnh/video, không lấy caption làm ground truth.
3. Đo phân bố cosine visual/semantic trên positives/negatives để calibrate gate.
   Không hạ ngưỡng chỉ để tăng số full paths hoặc tune trên ba query đang báo cáo.
4. Đo candidate coverage trước/sau video cap; nếu video đúng không vào pool,
   giải candidate generation trước khi tune DP.
5. Với nội dung bảng lớp/tiếng Anh, OCR độc lập là hướng phù hợp để kiểm tra chữ;
   **chưa triển khai/chạy OCR trong bước 1**. Với thao tác nấu ăn cần xem local
   frames để phân biệt thiếu evidence hay model không nhận ra.

Hiện chưa có yêu cầu sinh lại artifact/model nặng. Nếu kết quả gán nhãn xác định
cần thêm keyframes, re-embedding hoặc model video/clip verifier, dừng và trao đổi
với người dùng để chạy phần đó trên Kaggle.
