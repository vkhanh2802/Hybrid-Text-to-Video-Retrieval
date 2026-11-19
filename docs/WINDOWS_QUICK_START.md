# Chạy KIS trên Windows từ terminal mới

## Cấu hình hiện tại

- Groq dùng model `openai/gpt-oss-20b`.
- `GROQ_API_KEY` đã lưu tại User Environment Variables của tài khoản Windows
  trên máy này. Không có key trong tài liệu hoặc mã nguồn.
- `start_kis.ps1` đọc key từ terminal, rồi User Environment Variables nếu terminal
  chưa có key; chỉ hỏi bằng ô nhập ẩn khi cả hai đều chưa được cấu hình.
- Python được chọn tự động từ môi trường `focusguard_ai` có sẵn. Không cần
  `conda activate` trước khi dùng script.
- Địa chỉ DRES/Run ID đã nằm trong `config/dres.local.json` trên máy này.

## Các bước chạy

### 1. Mở Docker Desktop

Mở **Start → Docker Desktop**, chờ Docker Engine sẵn sàng. Nếu Docker đang chạy
thì không cần đóng hoặc khởi động lại nó.

### 2. Mở PowerShell

Nhấn **Windows + R**, nhập `powershell`, nhấn **Enter**. Không cần chạy quyền
Administrator cho thao tác khởi động thông thường này.

### 3. Vào project

```powershell
Set-Location 'C:\Users\Khanh\Desktop\KISBranche'
```

### 4. Bật các dịch vụ Milvus

```powershell
docker start milvus-etcd milvus-minio milvus-standalone
```

Ba container này đã có sẵn trên máy. Nếu báo không kết nối được Docker Engine,
quay lại bước 1 và chờ Docker Desktop sẵn sàng.

### 5. Chạy backend và web

```powershell
.\start_kis.ps1
```

Không cần nhập lại Groq key. Chờ xuất hiện dòng tương tự:

```text
Uvicorn running on http://127.0.0.1:8000
```

Giữ cửa sổ này mở khi dùng ứng dụng. Không chạy thêm một server khác trên cổng
8000. Nếu báo cổng đã được sử dụng, dùng cửa sổ đang chạy server hoặc Ctrl+C
trong cửa sổ cũ trước khi chạy bản mới; không kết thúc process không xác định.

### 6. Mở web và kết nối DRES

Mở **http://127.0.0.1:8000**. Nếu đã mở trang trước khi cập nhật code, nhấn
**Ctrl+F5** để nạp giao diện mới.

1. Kiểm tra Run ID đã được điền từ cấu hình.
2. Chọn **Username / password**, nhập tài khoản DRES của bạn, bấm **Kết nối DRES**.
3. Sau xác thực, khối kết nối ẩn và phần **Đề hiện tại** hiển thị đề text/video
   chính thức. Không nhập query thay đề hoặc tải video bằng tay. Toàn bộ câu text
   DRES đã phát được hiển thị; không tự cập nhật theo chu kỳ.
4. Khi DRES thêm câu mới, có thể bấm **Tìm** ngay: ứng dụng đọc lại toàn bộ câu
   đã phát trước khi chạy model. Không cần reset hay kết nối lại. **Làm mới đề**
   chỉ cập nhật hiển thị, không chạy model. Không tự polling.
5. Với đề video, có thể nhập **Text bổ sung cho query video** (không bắt buộc),
   ví dụ “Tập trung vào người mặc áo đỏ”. Video vẫn tự lấy từ DRES. Text chạy như
   đề text (Groq cấu trúc → SigLIP + BM25 + semantic caption, các bước tùy cấu
   hình), rồi hợp nhất danh sách kết quả bằng weighted RRF: 70% text + 30% video.
   Vẫn cần Groq key cho nhánh text dù tắt VLM chấm ảnh. Tổng text DRES + text
   bổ sung tối đa 8000 ký tự; riêng text bổ sung tối đa 2000.
6. Khi DRES đang phát đề và task RUNNING, bấm **Tìm theo đề DRES**.
7. Kiểm tra đủ 100 frame khác nhau. Chuỗi temporal nằm riêng bên dưới, không thay
   thế frame. Thiếu dữ liệu sẽ báo lỗi rõ, không lặp frame để bù. Nhánh text,
   hoặc video có text kèm theo, chỉ xét
   tối đa 20 ứng viên đầu bằng LLM. Video không có text chỉ dùng SigLIP.
   Mỗi card ghi đúng điểm/trạng thái; nếu tắt LLM hoặc thiếu ảnh thì không giả điểm.
8. Chọn frame → kiểm tra preview → chỉ bấm xác nhận nếu thực sự muốn nộp bài.

Lần tìm đầu tiên có thể lâu vì phải nạp model/index. Nếu DRES đã chuyển task/hint
trong lúc xử lý, ứng dụng bỏ kết quả cũ; hãy tìm lại theo đề mới.

Khi có câu mới trong lúc model đang chạy, lỗi 409 sẽ cập nhật lại đề trên trang;
bạn chỉ bấm Tìm lại, không cần reset. Kế hoạch dùng CPU trường qua VPN/SSH:
[SCHOOL_CPU_SSH.md](SCHOOL_CPU_SSH.md).

## Dừng và mở lại

- **Dừng:** Ctrl+C trong terminal đang chạy KIS.
- **Mở lại trong cùng terminal:** chạy lại `.\start_kis.ps1`.
- **Đổi terminal:** lặp lại bước 2–6; key đã lưu vẫn được đọc tự động.
- **Đóng/mở lại server:** model phải nạp lại vào RAM và phải kết nối lại DRES.
  Dataset/index, URL DRES/Run ID và cấu hình User Groq key không bị xóa.
- **Cập nhật code:** Ctrl+C rồi chạy lại script, sau đó Ctrl+F5 trên browser.

## Lưu ý về key

Biến User được lưu lâu dài trong tài khoản Windows, **không phải kho bí mật được
mã hóa**. Không dùng lệnh in key, đưa key vào mã nguồn/Git hoặc gửi key qua chat.
Nếu key đã lộ, thu hồi/tạo key mới trên Groq và cập nhật cấu hình User; terminal
đang chạy có thể vẫn dùng key cũ nên cần mở lại server.

Muốn thử một key khác chỉ trong phiên hiện tại bằng ô nhập ẩn:

```powershell
.\start_kis.ps1 -PromptForKey
```

Lệnh trên không ghi đè key User đã lưu. Cấu hình trên máy Windows này không tự
đồng bộ sang tài khoản hoặc server CPU của trường.
