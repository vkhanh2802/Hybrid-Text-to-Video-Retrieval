# OCR retrieval và Qwen evidence có điều kiện

## Cập nhật triển khai đã hoàn tất

Sau khi kết nối lại mạng trường, đã triển khai worker96666 tại8018/local18008,
chuyển active config và kiểm handshake. Snapshot mới chứa L01–L05 (36992record),
trong scope retrieval hiện tại nạp L01/L02/L03 (22074record). Đã chạy15lượt Qwen
với5query và3nhánh: `benchmark_results/user_five_selective_ocr_20261116_run01/`.
OCRretrieval tăng Hit@1 từ1/5 lên2/5 trong lượt này; prompt chọn lọc không tăng
metric và vẫn có outlier60–82giây. Chưa restartweb hoặc gọiDRES. Phần timeout
phía dưới là lịch sử attempt đầu, không phải trạng thái hiện tại.

## Thiết kế đã tích hợp trong source

1. Planner production giữ nguyên raw query. `literal_queries` dùng literal entity,
   OCR text nhập tay, hoặc span gốc sau cue như `dòng chữ`, `sign reads`.
   Cue rõ ràng vẫn hoạt động nếu planner quên entity/needs_ocr.
2. Với diễn đạt gián tiếp `dòng chữ liên quan...`, chỉ lấy tên riêng/acronym gốc
   trong span làm lexical keyword. Không dịch hay sửa lỗi gõ.
3. Exact match trước. Nếu không hit, public OCR thử tối đa `max_queries` cụm con
   liên tiếp 4token ở đầu/cuối literal dài. Không dùng OR toàn bộ từ, không bỏ
   dấu hoặc sửa OCR. `fallback_queries`/`match_kind` được ghi diagnostics.
4. Trên school snapshot, một thread OCR chạy chồng lấp với visual/BM25/semantic.
   Worker giữ request lock, không có hai query cùng truy cập FTS. Trường hợp
   temporal cần OCR seeds sẽ join trước event retrieval, không đổi temporal gate.
   Executor được join cả khi retrieval gặp lỗi.
5. Fusion trước top20, vẫn giữ bounded positive bonus0.1 theo private runtime,
   không thay corpus, caption, model hoặc trọng số visual/caption.
6. Worker vẫn validate evidence và mapping trên pool; chỉ gửi full OCR payload
   của top20 để giảm response. Ranking evidence/score và mọi frame vẫn giữ.
7. `rerank_candidates_with_vlm` nhận structured query để giữ manual OCR literal.
   Chỉ truyền OCR statusok có literal/cụm con liên quan, tối đa384ký tự quanh
   match. Không truyền OCRmissing/empty/error hoặc OCR không liên quan.
   Full OCR trên candidate không bị sửa; `vlm_ocr_used` thể hiện sử dụng prompt.
8. Qwen vẫn top20 gốc, batch12 theo startup hiện có; không backfill ảnh thiếu.
   Batch hỗn hợp dùng None ở candidate không có evidence liên quan.

CPU stage timings có thể overlap; `retrieval_wall_seconds` dùng tính RPC overhead
thay vì cộng tất cả stage duration. `join_wait_seconds` ghi thời gian chờ OCR.

## Trạng thái triển khai ngày16/11/2026

- Source local đã tích hợp; chưa commit/push.
- Bộ tests OCR/routing/pipeline/worker/web/batch/video/result/config đã chạy;
  kết quả kiểm cuối lưu trong báo cáo integration.
- Deployment attempt: `benchmark_results/ocr_selective_integration_20261116_run01`.
  SSH<school-host>:22 timeout ngay bước preflight; chưa upload hoặc tạo worker.
- Local18007 và web8000 không lắng nghe ở thời điểm kiểm. Active private config
  chưa được thay; không restart web hoặc gọiDRES.
- Chưa có phép đo latency/accuracy thực với code mới. Các metric benchmark
  A/B/C trước đó là của code cũ; không dùng để khẳng định code mới tốt hơn.

## Tiếp tục sau khi kết nối lại CPU trường

Deploy bằng `tools.deploy_school_worker` vào port8018/local18008, settings mới
`config/school-ocr-selective.local.json`, output **run02** (run01 giữ failure receipt).
Chỉ dùng snapshot hiện tại:

```
~/kis-ocr-l01-l02-20261115T172101Z/updated-groups-snapshot
0fdb5b786abe23f23a843cd352e7497aba9f076e83d89cd0c5464dad0acc81c6
```

Sau deployment, mở SSH tunnel18008→8018 rồi chạy:

```powershell
& "$env:USERPROFILE\miniconda3\envs\focusguard_ai\python.exe" -m tools.verify_selective_ocr_integration --output benchmark_results/ocr_selective_integration_20261116_run02
```

Smoke dùng5structured queries đã đóng băng, không gọi planner mới; retrieve thật
và một lượt GPUq5. Đây là smoke integration, không phải ablation mới. Kiểm100
distinct/mapped, top20, batch12, prompt≤384ký tự và Qwen thực sự chạy.

Chỉ sau smoke pass mới chuyển active private config bằng atomic replace. Web
đang chạy cần restart riêng sau xác nhận người dùng để nạp source mới; không
coi việc đổi private JSON là đã hot-reload process.
