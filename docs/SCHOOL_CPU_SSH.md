# Thiết kế tăng tốc bằng CPU Linux của trường qua OpenVPN + SSH

## Trạng thái và mục tiêu

**Cập nhật 15/11/2026:** worker/RPC và nguồn ảnh server đã được triển khai, Qwen
vẫn ở GPU laptop. Warm median end-to-end đo được 41,46 giây so với baseline
76,62 giây trong ba lượt warm; có missing ảnh/invalid score, không phải PASS sạch.
Hướng dẫn vận hành hiện tại: [SCHOOL_WORKER.md](SCHOOL_WORKER.md). Phần lịch sử
pilot/đề xuất bên dưới không thay thế trạng thái đã triển khai này.

Người dùng xác nhận có **SSH vào máy Linux** sau khi kết nối OpenVPN GUI.
Cập nhật 14/11/2026: đã xác thực SSH bằng tài khoản `<user>` tại
`<school-host>`, Ubuntu 22.04, Python 3.10.12. Máy ảo VMware hiển thị 12 vCPU
Xeon E5-2690, khoảng 12 GiB RAM (9,7 GiB khả dụng tại lần đo của người dùng),
AVX nhưng không thấy AVX2. Kiểm tra filesystem còn khoảng 259 GiB trống; quota
tài khoản riêng chưa xác định. Không coi topology 3 socket ảo là 3 CPU vật lý
được dành riêng. Không có cam kết nhanh hơn bao nhiêu lần trước benchmark.

Người dùng yêu cầu **sao chép dữ liệu trước, chưa chép keyframe**, giữ nguyên
nguồn và dữ liệu server. Đã chuyển và xác minh **349.100 file / 3,44 GB** vào
`~/kis-data-20261114T145713Z`, không đụng `~/KISBranche` đang có.
Receipt `TRANSFER_VERIFIED.json` xác nhận hash archive và từng file khớp; giữ
nguyên nguồn, bản nén và bản bung. Xem [SCHOOL_DATA_TRANSFER.md](SCHOOL_DATA_TRANSFER.md).
Tại thời điểm chỉ chuyển artifact này, chưa triển khai worker/RPC. Sau đó đã
triển khai theo kiến trúc hybrid; không cài lại package hoặc model cache cũ.

Sau đó, theo yêu cầu thử L01, đã chuyển ZIP L01 và chạy pilot CPU riêng tại
`~/kis-l01-pilot-20261114T152802Z`. Có 7.339 ảnh/31 video và 6 lượt
trả 100 frame, đã kiểm tra hiển thị ảnh qua browser. Pilot tái sử dụng venv/cache
có sẵn, không cài package và không chạy Groq/Qwen/Milvus. Xem
[SCHOOL_L01_PILOT.md](SCHOOL_L01_PILOT.md). Đây chưa phải worker tích hợp với
backend DRES local hoặc triển khai production toàn corpus.

OpenVPN chỉ tạo đường mạng. Nó không biến CPU máy trường thành core của laptop;
phải thực sự chạy process trên máy Linux để tận dụng CPU đó.

## Kiến trúc khuyến nghị: giữ DRES và Qwen trên laptop

```text
LAPTOP WINDOWS                         LINUX TRƯỜNG (CPU)
Browser + backend DRES                 Worker retrieval chạy lâu dài
  - session DRES trong RAM               - SigLIP text/image embedding
  - Groq planner/key                     - MiniLM + FAISS
  - ghép toàn bộ câu text                - BM25 + object/OCR theo cấu hình
  - kiểm tra task/hint                   - temporal + fusion + pool đủ 100
               │                                   │
               └──── OpenVPN + SSH tunnel ──────────┘
                       request query/ảnh mẫu
                       response IDs/scores/metadata
               │
               ▼
Laptop GPU RTX 3050: Qwen chấm tối đa 20 frame
Browser hiển thị 100 frame + chuỗi temporal riêng
Nộp DRES chỉ sau xác nhận tại laptop
```

- Không chuyển password/session DRES, Groq key hay Drive credentials sang worker.
- Không gửi raw video nếu có thể gửi tối đa 6 ảnh mẫu đã resize; cần quyền của
  trường/ban tổ chức để xử lý nội dung đề trên máy này.
- Corpus caption, vector, mapping và ảnh ở ổ đĩa máy trường. Worker hiện dùng
  exact NPY cosine, không cần Milvus server; không stream corpus qua VPN mỗi query.
- Cố định model revisions, scope và hash/version artifacts giống laptop. Dùng
  lại artifacts hợp lệ, không rebuild caption/vector chỉ để chuyển máy.
- Worker chỉ bind `127.0.0.1`, truy cập bằng SSH forwarding. Không mở public
  port FastAPI/Milvus hoặc tắt firewall/TLS/SSH host-key checking.
- Khi worker/VPN lỗi: báo lỗi rõ, không bịa kết quả, không gửi lại bài nộp.

**Đã triển khai:** `kis/school_worker.py`, `kis/remote_worker.py` và adapter trong
web; `start_school.ps1` bật đường remote tường minh. SigLIP/BM25/MiniLM thực sự
chạy trong process server resident, không chỉ đổi IP Milvus. Worker auth/scope/
model handshake và lỗi VPN không fallback âm thầm. Xem hướng dẫn mới để chạy.

## Vì sao không chuyển toàn bộ Qwen sang CPU ngay?

Qwen2.5-VL-3B hiện yêu cầu CUDA mặc định. CPU nhiều core chưa chắc nhanh hơn GPU
laptop cho VLM; vẫn bị giới hạn memory bandwidth, token/image processing và RAM.
Chuyển retrieval sang CPU trường chỉ giảm phần retrieval, không tự giảm thời
gian tải Drive, Groq hoặc Qwen. Phương án ban đầu giữ Qwen trên GPU laptop.

Nếu cần CPU-only hoàn toàn: phải benchmark backend VLM CPU/quantization riêng,
kiểm chứng output score và bộ nhớ, không tự bật `allow_cpu=True` rồi mặc định nhanh.
Một chế độ không VLM có thể nhanh hơn nhưng là thay đổi chức năng, cần người dùng
chọn rõ; vẫn phải chạy đầy đủ pipeline text và trả đủ 100 frame.

## Thông tin cần lấy (chỉ đọc, trong phiên SSH của bạn)

Kết nối OpenVPN bằng profile trường đã cấp. Xác thực SSH fingerprint với IT trước
khi kết nối. Dùng host/user/port do trường cấp, không quét subnet tìm máy.

```powershell
ssh -p <ssh-port> <user>@<school-host>
```

Trên Linux, chạy từng lệnh:

```bash
lscpu
nproc
free -h
df -h "$HOME"
python3 --version
```

Nếu là container/cgroup v2, các lệnh sau (nếu file tồn tại) cho biết quota thật:

```bash
cat /sys/fs/cgroup/cpu.max
cat /sys/fs/cgroup/memory.max
```

Nếu trường dùng scheduler, hỏi IT về số CPU/RAM/thời gian được cấp và cổng/service
có được chạy hay không; không chạy job nặng trên login node. Cần biết thêm: có
Docker/rootless container không, quota đĩa và Internet để dùng model cache.

Không cần gửi mật khẩu SSH, private key, OTP hay nội dung `.ovpn`. Chỉ chia sẻ
thông số tài nguyên và cách truy cập không bí mật; địa chỉ host nếu cho phép.

## SSH tunnel của worker hiện tại

Worker bind `127.0.0.1:8014` trên máy trường, laptop forward `18004`:

```powershell
ssh -N -o BatchMode=yes -o StrictHostKeyChecking=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:18004:127.0.0.1:8014 <user>@<school-host>
```

Laptop adapter gọi `http://127.0.0.1:18004` với token worker riêng do deploy tạo.
Launcher có thể tạo tunnel khi chưa có. Không chép credentials vào script/Git và
không bật `StrictHostKeyChecking=no`.

## Thứ tự tối ưu và đo

1. **Đo baseline warm** trên laptop: cùng 10–20 query/clip, cùng corpus/scope,
   tách cold load, Groq, retrieval, temporal, VLM và tải ảnh. Dùng median/P95,
   không kết luận từ một query.
2. **Nạp worker/model/index một lần**, chạy lâu dài trong allocation hợp lệ.
   Tránh mở process/nạp index qua SSH mỗi lượt tìm. Bộ nhớ phải đủ giữ corpus.
3. **Batch encode và cache embedding truy vấn trùng** trong RAM có giới hạn,
   khóa cache chứa model revision và nội dung text. Cache kết quả còn cần corpus
   version + config + video hash. Câu mới phải làm đổi cache key; không cache chỉ
   bằng task ID. Không lưu secret vào cache.
4. **Giới hạn threads theo quota**: thử 4/8/16 hoặc các mức không vượt CPU được
   cấp. Cân chỉnh PyTorch, OpenMP/MKL và FAISS; không để mỗi thư viện tự dùng toàn
   core đồng thời gây oversubscription. `nproc` có thể lớn hơn quota thực.
5. Chỉ song song BM25/semantic/visual khi đo thấy có lợi; không fork process chứa
   toàn bộ model/index làm nhân RAM. Tránh nhiều Uvicorn workers tự nhân model.
6. Đo riêng VPN RTT và payload. Gộp RPC thành một request đủ query variants/ảnh
   mẫu, trả top candidates và evidence cần thiết, không một RPC cho mỗi frame.
7. Sau retrieval, đo lại Qwen/Drive. Nếu chúng vẫn chiếm phần lớn thời gian, bước
   tiếp theo là prefetch ảnh có giới hạn, batch VLM nhỏ và trả retrieval trước
   VLM (cần thiết kế request/revision và stale-task guard riêng).

**Đã cải thiện nhỏ trong bản hiện tại:** mở rộng pool video tái dùng vector clip,
không encode lại clip; mở rộng pool text tái dùng SigLIP vector trong request.
`stage_seconds` có thêm `runtime_load` và `query_planning`; video có
`text_stage_seconds` để tách thời gian nhánh text.

## Tiêu chí nghiệm thu trước chuyển máy

- Mỗi response thành công: 100 `(video_id, frame_id)` khác nhau, không padding.
- Câu text mới dùng toàn bộ nội dung đã phát, không reset, cache không trả câu cũ.
- Text/video 70/30, cùng model/config và mapping; kiểm tra top-k/score sai lệch
  khi đổi CPU backend/precision, không chỉ đo thời gian.
- VLM chỉ top 20 frame; temporal không bị coi là đã xác minh bởi điểm ảnh đơn.
- Mất VPN, worker lỗi, task/hint đổi: không trả kết quả cũ như kết quả hiện tại.
- Không tự submit; credentials DRES/Drive/Groq vẫn ở laptop.
