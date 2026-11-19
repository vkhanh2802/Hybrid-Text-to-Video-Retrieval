# Thử nghiệm L01 trên CPU trường với ảnh thật

## Mục đích và phạm vi

Người dùng yêu cầu chuyển ZIP L01 và thử xem CPU trường truy xuất được frame/ảnh
không. Đây là **pilot L01 riêng**, không đổi backend production, scope hiện tại,
phiên DRES, quyền Drive hoặc dữ liệu đã có ở `~/KISBranche`.

ZIP local: `dataset/data/L01.zip`, khoảng 1,18 GB, **7.339 ảnh / 31 video**.
ZIP có cấu trúc lồng `L01/L01-.../L01/L01_V001/001.jpg`; extractor đưa về
`images/L01/L01_V001/001.jpg`, giữ nguyên tên video và số ảnh, không renumber.
Giữ ZIP gốc, ZIP server và ảnh đã giải nén; không xóa dữ liệu sau thử.

## Môi trường

Đã kiểm tra interpreter có sẵn `~/KISBranche/.venv/bin/python`:

- Python 3.11.17; NumPy 2.4.4; PyTorch 2.7.1+cpu; FAISS 1.12.0.
- Pillow 11.1.0; open-clip 3.3.0; sentence-transformers 5.7.0.
- `torch.cuda.is_available() = false` trong process test.
- Cache có đúng revisions MiniLM và SigLIP theo `config/models.json`.

Chỉ **tái sử dụng interpreter/package/cache**, không cài đè môi trường cũ.
Source snapshot riêng không có local config, Drive reader/credentials, `.env`
hoặc Git history. Mọi request model đặt offline; không dùng token để tải model.

## Thao tác

Từ PowerShell tại project, khi VPN và SSH đã hoạt động:

```powershell
& "$env:USERPROFILE\miniconda3\envs\focusguard_ai\python.exe" -u -m tools.school_l01_pilot --zip dataset/data/L01.zip --artifact-root ~/kis-data-20261114T145713Z/dataset --output-dir dist/school-l01-pilot-20261114-01 --target <user>@<school-host>
```

Tên output phải mới; công cụ không xóa/ghi đè thư mục có sẵn. Mỗi lần tạo thư
mục server mới `~/kis-l01-pilot-<UTC timestamp>` và ghi `PILOT_STATE.json` local
trước SFTP để biết chính xác đích/hash. Nếu lỗi, giữ dữ liệu để kiểm tra, không
tự retry toàn workflow và tạo bản sao lặp không kiểm soát.

1. Hash SHA-256 ZIP L01 và source ZIP nhỏ; SFTP với known-host checking.
2. Server đối chiếu hash trước khi giải nén.
3. Từ chối zip-slip, symlink, file mã hóa, identity ảnh trùng, ảnh quá lớn hoặc
   không phải numbered L01 image. Mở đích bằng exclusive mode, không overwrite.
4. Đọc hết từng ảnh từ ZIP để kiểm CRC; lưu SHA-256/size từng ảnh trong manifest.
5. Đối chiếu toàn bộ CSV mapping L01 với ảnh và NPY 768 chiều; thiếu/mơ hồ thì
   dừng, không lấp bằng frame giả hoặc tự suy lại mapping.
6. Dùng SigLIP và MiniLM CPU đúng model/revision hiện có, 4 thread. Caption index
   vẫn phải qua manifest/hash/schema validation của pipeline thật.
7. Chạy solver dùng chung: SigLIP text-to-image + BM25 + semantic caption + RRF,
   bắt buộc mỗi response 100 identity frame khác nhau.
8. Ba query tiếng Anh cố định, mỗi query hai lượt; nạp model/index một lần.
   Lượt đầu mỗi query đọc/giải mã 100 ảnh thật và tạo thumbnail để kiểm hiển thị.
9. Ghi CPU_REPORT.json đầy đủ và gallery HTML riêng; source ZIP/config/code cũ
   và artifact đã chuyển trước đều được giữ nguyên.

## Khác biệt với production — không được bỏ qua khi diễn giải tốc độ

- **Không gọi Groq planner**: query/variants cố định để kiểm chức năng retrieval.
- **Không chạy Qwen/VLM**, không xác minh relevance bằng LLM.
- **Không dùng Milvus server**: visual search là cosine/IP chính xác trên NPY
  L01, bằng NumPy. Đây là adapter pilot, không thay Milvus production.
- Chỉ L01/31 video, không phải toàn corpus các nhóm đang dùng trên laptop.
- Temporal và OCR/text cross-encoder tắt cho ba query đơn của pilot.
- Nhanh/đúng chức năng không đồng nghĩa accuracy tăng hoặc full DRES pipeline
  có latency tương tự. Không nộp DRES trong thử nghiệm này.

## Xem ảnh

Gallery nằm tại `<pilot-root>/results/gallery/`, có 100 card/query và liên kết
ảnh gốc local. Chỉ phục vụ qua HTTP bind **127.0.0.1** và SSH tunnel khi cần xem;
không mở public port. Không cần Drive để đọc ảnh đã bung trên server.

Pilot này không tự mở dịch vụ lâu dài. Khi kiểm tra bằng browser, ghi lại PID
server/tunnel và chỉ dừng đúng process do lượt test tạo; không đụng backend cũ.

## Kết quả lượt 14/11/2026

Đã chạy thành công tại `~/kis-l01-pilot-20261114T152802Z`:
7.339 ảnh/mapping/vector/caption, không thiếu ảnh; 6 lượt × 100 frame; retrieval
0,249–0,778 giây sau nạp model. Đây là pilot không Groq/Qwen/Milvus, không phải
cam kết tốc độ toàn pipeline. Xem báo cáo
`benchmark_results/school_l01_cpu_20261114/SUMMARY.md` và ảnh preview.

HTTP/tunnel kiểm tra đã dừng sau xác minh; ZIP, ảnh, gallery và report vẫn còn.
Để tự xem lại gallery kết quả đã lưu, trong SSH Ubuntu chạy:

```bash
cd ~/kis-l01-pilot-20261114T152802Z/results/gallery
python3 -m http.server 8012 --bind 127.0.0.1
```

Mở PowerShell **khác** trên laptop, giữ VPN rồi chạy:

```powershell
ssh -N -L 127.0.0.1:18002:127.0.0.1:8012 <user>@<school-host>
```

Mở `http://127.0.0.1:18002/`. Đây là gallery tĩnh ba query đã thử, chưa phải
giao diện nhập query/DRES đang chạy. Ctrl+C ở hai terminal để dừng server/tunnel;
không xóa dữ liệu. Nếu cổng có process khác sử dụng thì chọn cổng riêng, không
kill process không xác định.
