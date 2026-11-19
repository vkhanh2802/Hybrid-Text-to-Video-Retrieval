# Sao chép dữ liệu KIS không gồm keyframes lên máy trường

## Phạm vi được người dùng chọn ngày 14/11/2026

- Chuyển trước toàn bộ artifact hiện có trong `dataset/caption_generator`,
  `caption_mapping`, `clip-features-32`, `index`, `mapping`, `objects`, `ocr`,
  `preparation`.
- Kèm **chỉ** ba cấu hình không bí mật: `config/models.json`, `pipeline.json`,
  `object_labels.json`. Không đổi scope hoặc model revision.
- Chưa chuyển `dataset/data`, ảnh JPEG/PNG, ZIP chứa ảnh, video gốc hoặc dữ liệu
  trên ổ Google Drive. Người dùng sẽ tải đầy đủ keyframe rồi yêu cầu chuyển sau.
- Không chuyển `config/*.local.json`, `.env`, SSH/VPN keys, Drive credentials,
  session DRES, browser storage, model cache hoặc Git history.
- **Chỉ sao chép. Không xóa nguồn, không xóa bản sao sau khi truyền, không đụng
  hoặc ghi đè `~/KISBranche` đã có trên server.**

Đây là chuyển file artifact, không phải triển khai backend. Các `.npy` là
embedding nguồn; không đồng nghĩa collection/index Milvus trên Docker laptop
đã được chuyển. Chưa thể chỉ chạy code cũ trên server rồi coi như pipeline mới
đã hoạt động; cần kiểm tra environment, artifact schema và cấu hình retrieval.

## Công cụ

`tools/transfer_school_data.py` chỉ dùng Python stdlib và OpenSSH đã có, không
nhận password. SSH bắt buộc `BatchMode=yes`, `StrictHostKeyChecking=yes`, dùng
agent/key/config hợp lệ; không tự chấp nhận host key mới hoặc tắt bảo mật.

Ví dụ thao tác chủ động từ PowerShell tại project (tên output phải mới):

```powershell
& "$env:USERPROFILE\miniconda3\envs\focusguard_ai\python.exe" -u -m tools.transfer_school_data --output-dir dist/school-transfer-new --target <user>@<school-host> --transfer
```

Bỏ `--transfer` nếu chỉ muốn kiểm kê, không đóng gói hoặc SSH. Không đưa mật
khẩu vào command, biến môi trường, JSON, Git hoặc script.

## Quy trình và file kiểm chứng

1. Kiểm kê bằng allowlist, bỏ ảnh/archives/secret-like paths, không theo symlink.
2. Ghi `PLAN.json`: danh sách file, size/mtime, số lượng theo component.
3. Đóng gói `artifacts.tar.gz`, hash SHA-256 từng file ngay khi đọc. Source thay
   đổi sau inventory hoặc trong lúc đóng gói thì dừng, không dùng bản chụp lẫn.
4. Ghi `PACKAGE.json`: SHA-256 archive và dung lượng file nguồn/nén.
5. Kiểm tra dung lượng máy trường cho cả archive, dữ liệu bung và 2 GiB dự phòng.
6. Tạo thư mục mới riêng `~/kis-data-<UTC timestamp>`, mode 0700. Nếu đã có thì
   dừng, không ghi đè hay gộp vào dữ liệu người dùng. Ghi `UPLOAD_PACKAGE.json`
   để khóa danh tính package và tạo file `.partial` rỗng bằng exclusive mode.
7. SFTP `put -a` bản nén vào `artifacts.tar.gz.partial`; xác minh hash archive
   trên server. File đích phải tồn tại để OpenSSH SFTP resume có thể stat được,
   kể cả lần đầu khi size=0.
8. Bung file thường, từ chối đường dẫn tuyệt đối, `..`, symlink và entry trùng;
   mọi file được tạo bằng exclusive mode, không ghi đè.
9. So size/SHA-256 từng file bung với `TRANSFER_MANIFEST.json` trong archive.
10. Chỉ sau khi mọi file khớp mới ghi `TRANSFER_VERIFIED.json` ở server và
    `RECEIPT.json` ở thư mục output local. Giữ nguyên archive ở cả hai nơi.

Thư mục output nằm trong `dist/` đã được Git ignore. Receipt/manifest là dữ liệu
kiểm chứng, không chứa password. Không coi thư mục hoặc archive có mặt là bằng
chứng truyền hoàn tất; phải có receipt `status="verified"`.

Nếu mất VPN, đầy quota hoặc checksum không khớp, công cụ dừng và giữ file đã tạo
để kiểm tra. Không tự xóa hoặc tự retry vô hạn. Với partial upload có thể resume
**chủ động** mà không đóng gói lại:

```powershell
& "$env:USERPROFILE\miniconda3\envs\focusguard_ai\python.exe" -u -m tools.transfer_school_data --output-dir dist/school-transfer-20261114-no-keyframes-01 --resume-package dist/school-transfer-20261114-no-keyframes-01/PACKAGE.json --remote-name kis-data-20261114T145713Z --target <user>@<school-host> --transfer
```

Chỉ dùng đúng package và thư mục của lần truyền đó. Resume kiểm tra hash local,
marker/hash/size remote, từ chối symlink, partial không rõ nguồn hoặc thư mục đã
có dữ liệu khác. Thư mục rỗng do lần lỗi trước có thể được khởi tạo lại có kiểm
soát, không xóa thư mục. CLI từ chối lặp lại khi đã có `RECEIPT.json`. Chưa tự
resume việc bung/verify nếu lần trước đã rename archive và bung dở; trường hợp
đó cần kiểm tra riêng, không ghi đè hoặc xóa dữ liệu để thử lại.

## Sau chuyển

- Giữ bản dữ liệu mới riêng với repo cũ; chưa cài package hoặc chạy job nặng.
- Cần kiểm tra wheel PyTorch/FAISS phù hợp guest CPU không thấy AVX2, và quota
  RAM/CPU thực. Không mặc định 12 vCPU là core độc quyền.
- Khi có keyframes đầy đủ, kiểm kê dung lượng và kiểm tra mapping trước khi
  chuyển vào thư mục dữ liệu ảnh riêng. Không dùng Drive credentials để tải ảnh
  thay người dùng trong lượt chuyển artifact này.
- Không có thao tác DRES search/submission hoặc sửa VPN từ công cụ chuyển file.
