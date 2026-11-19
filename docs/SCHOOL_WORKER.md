# Worker CPU trường + Qwen GPU laptop

> **Cập nhật OCR:** sau xác nhận restart của người dùng, private school runtime
> hiện chọn worker OCR90716, remote8015 qua SSH local18005. Worker baseline89224
> (8014/18004) và config trước OCR được giữ, không sửa dataset gốc. Port cũ trong
> mô tả triển khai ban đầu bên dưới là lịch sử. Xem
> [SCHOOL_OCR.md](SCHOOL_OCR.md) về snapshot, coverage, giới hạn và kiểm tra thật.

## Kiến trúc

```text
Laptop: DRES + Groq planner + kiểm tra task/hints + Qwen GPU + web
                         │ HTTP loopback có token riêng
                         │ SSH tunnel qua VPN
CPU trường: SigLIP + BM25 + MiniLM/FAISS + object evidence + temporal
            dataset ảnh trên ổ đĩa; model/index nạp một lần
```

`kis/school_worker.py` là worker resident, chỉ bind `127.0.0.1:8014` trên server.
`kis/remote_worker.py` gọi qua `127.0.0.1:18004` tại laptop. Không gửi khóa Groq,
Drive hoặc DRES sang trường. Token worker riêng là file 0600 trên server và
config local bị Git ignore trên laptop; không in token vào log/report.

## Phạm vi và tính tương đương

- Giữ nguyên scope trong `config/pipeline.json` và model revisions. Handshake
  kiểm model manifest + groups + retrieval profile + temporal/OCR/text-reranker
  config, mismatch thì dừng thay vì fallback.
- Visual worker dùng **exact cosine trên NPY**, không dùng Milvus laptop qua VPN.
  Không rebuild hoặc đổi bytes vector. Đây là khác biệt backend phải ghi rõ
  trong báo cáo tốc độ/relevance, không coi là IVF_FLAT hoàn toàn giống exact.
- Loại frame identity mơ hồ theo cùng quy tắc không join duplicate frame_idx;
  giữ nguyên CSV và báo số identity bị loại trong worker status.
- Text query được Groq xử lý trên laptop rồi gửi toàn bộ structured query sang
  worker. Solver đầy đủ chạy trên server (VLM off tại đó); **toàn bộ pre-Qwen
  candidate pool** được trả về, không cắt 100 sớm làm sai final ranking.
- Laptop chấm tối đa 20 ảnh bằng Qwen, xếp hạng cuối và yêu cầu đúng 100 identity
  khác nhau. Temporal results vẫn là auxiliary, không thay danh sách chính.
- Video được decoder giới hạn trên laptop xử lý thành tối đa sáu ảnh mẫu; gửi
  PNG lossless để SigLIP trên server encode. Weighted RRF text/video vẫn giữ
  70/30, full text path chạy trước fusion, Qwen chỉ chạy một lần sau fusion.
- Không gửi video gốc lên server. Decoder/kiểm tra input và điều phối cuối vẫn
  trên laptop; không có model retrieval corpus lớn được nạp ở laptop khi remote on.
- Ảnh chỉ phục vụ nếu đúng scope và có số ảnh trong authoritative CSV. Path
  traversal/outside-root bị từ chối; không cache ảnh trên đĩa laptop.
- Server còn thiếu image groups L18/L21 so với production; trả 404 thật, không
  loại hai nhóm khỏi retrieval, không fallback Drive hoặc tạo ảnh thay thế.
- DRES hint accumulation, Search-triggered refresh, stale checks, xác nhận nộp
  một lần và không tự nộp vẫn nằm nguyên trên laptop.

## Triển khai và chạy

Tái sử dụng venv/cache trường; không cài package hoặc tải model mới:

```powershell
& "$env:USERPROFILE\miniconda3\envs\focusguard_ai\python.exe" -u -m tools.deploy_school_worker --target <user>@<school-host> --dataset ~/kis-normalized-20261114T160546Z/dataset --output dist/school-worker-deploy-20261115-01
```

Công cụ tạo snapshot root riêng `~/kis-worker-<UTC>/`, giữ source ZIP/log/receipt;
không ghi đè repo, artifact hoặc các snapshot trước. Startup dùng pipe readiness
để đợi load xong, không poll log/file và không khởi chạy model lại mỗi request.
Receipt không có token; config private là `config/school-runtime.local.json`.
Không chạy deploy lại nếu config đã có; không xóa config để thử lại mù quáng.

Sau khi có VPN, khởi động web hybrid bằng:

```powershell
.\start_school.ps1
```

Launcher tạo tunnel nếu port chưa được dùng, truyền worker URL/token qua process
environment, rồi chạy `start_kis.ps1`. Chỉ dừng tunnel do chính invocation đó tạo.
Listener có sẵn vẫn phải vượt handshake/token; không tin process chỉ vì đúng port.
Web vẫn bind localhost:8000; không mở public API. Không in config private.

`start_kis.ps1` bình thường không tự chuyển sang school runtime file. Có thể
chọn remote bằng `KIS_WORKER_URL` + `KIS_WORKER_TOKEN`, hoặc file opt-in riêng
`config/worker.local.json`. Không đồng thời chạy hai web process giữ Qwen GPU.

## Đo end-to-end

Đặt hai biến worker từ config private **trong process benchmark**, rồi chạy:

```powershell
& "$env:USERPROFILE\miniconda3\envs\focusguard_ai\python.exe" -u -m tools.benchmark_full_pipeline --output benchmark_results/school_hybrid_timing_20261115_run01/e2e
```

Runner ghi backend và nguồn ảnh, Groq thật, stage server, transport overhead,
image reads qua VPN, Qwen GPU load/score, cold/warm, số score và missing thật.
Không cộng trùng RPC wall với server stage hoặc VLM stage với image/Qwen times.
So sánh với `full_pipeline_timing_20261115_run01` cần nêu rõ backend exact vs
Milvus và số ảnh chấm khác nhau do nguồn ảnh thay đổi. Không gọi đó là isolated
CPU hardware speedup. Chưa tính browser rendering sau HTTP response.

## Trạng thái triển khai/đo

Code đã có unit/integration tests cho auth, loopback URL, scope/model mismatch,
pool 100 distinct, giữ full query/pre-Qwen pool và budget 20. Full regression
suite: **945 tests — 929 pass, 16 skip**, không failure/error (149,413 giây).
Worker thực đã ready tại `~/kis-worker-20261114T192739Z`, PID 89224,
tunnel PID 22016. RPC text/image/video-embedding smoke đạt. Có 130.996 caption
vectors và 138.776 visual vectors an toàn; 20 raw rows L21 có frame identity mơ
hồ bị loại, không sửa bytes dữ liệu. Đây là khác biệt với 138.796 rows Milvus
baseline cần công bố. Timing end-to-end đã hoàn tất: warm median **41,46 giây**
so với baseline **76,62 giây**, giảm **45,89%** trong ba lượt warm, chưa gồm browser
rendering. Có 10 lượt đọc ảnh thiếu và một Qwen invalid score; không phải PASS
sạch hoặc kiểm tương đương relevance. Chi tiết khác biệt exact/IVF và số candidate
được chấm nằm trong report, không gọi đó là hardware-only speedup.

Báo cáo: [school_hybrid_timing_20261115_run01](../benchmark_results/school_hybrid_timing_20261115_run01/SUMMARY.md).

### Cập nhật Qwen batching

Qwen đã chuyển từ scalar sang independent-row batch, cấu hình school runtime
hiện chọn microbatch12 sau thử1/4/8/12/20. Cùng20 ảnh: median16,49s thay27,51s;
size20 gần như hòa nhưng dùng VRAM cao hơn. Có score drift giữa các size, xem
[QWEN_BATCH.md](QWEN_BATCH.md), không diễn giải là quality-equivalent. Một warm
end-to-end sample mới22,46s, chưa phải median/P95 mới. Full suite956 tests đạt
(940 pass,16 skip). Worker/tunnel giữ, web đã restart và cần login DRES lại.
