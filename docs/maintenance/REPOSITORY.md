# Cấu trúc repo và bảo trì

## Phân chia trách nhiệm

| Thư mục | Trách nhiệm |
| --- | --- |
| `kis/` | Logic ứng dụng, model adapter, truy hồi, web và DRES API |
| `web/` | Frontend không có build step |
| `config/` | Manifest dùng chung và các file local gitignored |
| `tools/` | CLI chuẩn bị dataset, caption, embedding và Milvus |
| `scripts/` | Khởi động ứng dụng |
| `tools/legacy/`, `scripts/legacy/` | Công cụ triển khai cũ, tách khỏi demo local |
| `docs/data/` | Tài liệu dữ liệu |
| `docs/maintenance/` | Tài liệu vận hành và bố cục mã nguồn |
| `dataset/` | Nguồn dữ liệu và artifact, giữ nguyên đường dẫn runtime |
| `dist/` | Archive/build sinh ra, không commit |

Không đổi tên package `kis` hoặc đường dẫn dataset chỉ để làm cây thư mục ngắn
hơn: các module, config và manifest đang tham chiếu những đường dẫn này.

## Điểm vào

Launcher được duy trì ở `scripts/start_kis.ps1`, chạy với working directory là
gốc dự án dù được gọi từ thư mục khác. `start_kis.ps1` ở gốc chỉ là shim chuyển
tham số, giữ tương thích với shortcut/lệnh trước đây.

Hai tiện ích triển khai cũ đã chuyển module:

- `tools.deploy_school_worker` → `tools.legacy.deploy_school_worker`
- `tools.transfer_school_data` → `tools.legacy.transfer_school_data`
- `start_school.ps1` → `scripts/legacy/start_school.ps1`

Không xóa các module runtime worker trong `kis/`: backend hiện vẫn import chúng.
Di chuyển tiện ích không thay config local, không triển khai lại worker, không
restart web và không ngắt phiên DRES đang chạy.

Khi chạy demo **local**, không truyền `KIS_WORKER_URL` / `KIS_WORKER_TOKEN` từ
phiên triển khai cũ. `config/worker.local.json`, nếu có, cũng có thể chọn remote
backend; kiểm tra cấu hình này thay vì tự xóa. Cấu hình model/dữ liệu không được
tự chuyển chỉ vì tái cấu trúc thư mục.

## Chính sách dọn dẹp

- Có thể xóa `.pytest_cache/`, `__pycache__/`, `.pyc` được Python tạo lại.
- `structured_query.json` ở gốc là đầu ra CLI; có thể sinh lại, không phải dataset.
- Gói `kis-kaggle-code.zip` ở gốc được chuyển vào `dist/`, không xóa nội dung.
- Giữ nguyên báo cáo, notebook, benchmark, schema và tài liệu nghiên cứu local
  khi chưa xác định được chúng đã hết giá trị; không coi file gitignored là rác.
- Không dọn `dataset/`, `volumes/`, model cache, archive nguồn hay cấu hình private.
- Nếu xóa module Python, phải kiểm tra import và chạy regression test trước.
