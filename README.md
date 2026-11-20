# KIS Framefinder

Hệ thống **Known-Item Search (KIS)**: tìm keyframe trong video bằng mô tả tiếng
Việt/tiếng Anh, xem các frame liên quan và chọn kết quả để nộp lên DRES.

## Nguồn dữ liệu: cuộc thi AIC

Dự án sử dụng bộ dữ liệu video/keyframe của **cuộc thi AI Challenge (AIC)**.
Trang thông tin cuộc thi: <https://aichallenge.vn/>. Video được nhận diện bằng
mã như `L01_V001`; cấu hình hiện tại hỗ trợ các nhóm **L01–L24**.

- **Dữ liệu nguồn:** video/keyframe và CSV ánh xạ được cung cấp trong bộ dữ liệu
  cuộc thi. Giữ nguyên mã video, số ảnh, frame ID và timestamp của nguồn.
- **Dữ liệu dẫn xuất của dự án:** caption, vector SigLIP2, vector/index caption,
  metadata đối tượng và các kết quả OCR do pipeline xử lý hoặc nhập thêm.
  Những thành phần này không được mặc định coi là nhãn chuẩn của ban tổ chức.
- Dataset, video, ảnh, model weights và cấu hình chứa thông tin riêng **không
  được đưa vào Git**. Người dùng cần tự chuẩn bị dữ liệu theo quyền truy cập và
  điều khoản của phiên bản cuộc thi tương ứng. Repo không cấp lại quyền phân phối
  bộ dữ liệu và không chứa đáp án của cuộc thi.

## Cấu trúc dự án

```text
KISBranche/
├── kis/                    # Backend, planner, truy hồi, ranking, API DRES
├── web/                    # Giao diện HTML / CSS / JavaScript
├── config/                 # Model manifest, pipeline và cấu hình mẫu
├── tools/                  # Chuẩn bị dữ liệu, embedding và tạo index
│   └── legacy/             # Tiện ích triển khai cũ, không thuộc luồng demo local
├── scripts/                # Script khởi động
│   ├── start_kis.ps1
│   └── legacy/             # Launcher cũ được giữ để tương thích vận hành
├── docs/                   # Tài liệu cấu trúc dữ liệu và bảo trì
├── dataset/                # Dữ liệu local, không có trong Git
├── tests/                  # Kiểm thử local
├── notebooks/              # Notebook xử lý/thử nghiệm local
├── benchmarks/             # Định nghĩa phép thử local
├── benchmark_results/      # Kết quả thử nghiệm local
├── dist/                   # Gói build/export, không có trong Git
├── volumes/                # Dữ liệu bền vững của Docker, không có trong Git
├── docker-compose.yml      # Milvus và các dịch vụ phụ trợ local
├── requirements.txt
└── start_kis.ps1            # Điểm vào tương thích, chuyển đến scripts/
```

Các thư mục dữ liệu, thử nghiệm và build ở trên có thể chỉ xuất hiện trong
workspace local. Xem [mục lục tài liệu](docs/README.md).

## Cấu trúc bộ dữ liệu

Quy ước tổ chức: **thành phần → nhóm → video → file**. Ví dụ:

```text
dataset/
├── data/L01/L01_V001/001.jpg
├── mapping/L01/L01_V001/L01_V001.csv
├── siglip2-features/L01/L01_V001/L01_V001.npy
├── caption_generator/L01/L01_V001/L01_V001.json
├── caption_mapping/L01/L01_V001/L01_V001.json
├── index/L01/L01_V001/L01_V001.index
├── index/L01/L01_V001/L01_V001.meta.json
├── objects/L01/L01_V001/001.json
└── ocr/                    # Gói OCR hoặc artifact/index OCR tùy định dạng nhập
```

| Thành phần | Nội dung và vai trò |
| --- | --- |
| `data/` | Ảnh keyframe gốc để hiển thị và tạo embedding |
| `mapping/` | CSV nối số ảnh `n` với frame gốc `frame_idx`, thời gian `pts_time` và `fps` |
| `siglip2-features/` | Vector ảnh SigLIP2, 768 chiều; mỗi hàng khớp một dòng CSV |
| `caption_generator/` | Caption/mô tả từng keyframe, dùng cho BM25 và semantic search |
| `caption_mapping/` | Ánh xạ thứ tự caption/index về video và keyframe |
| `index/` | FAISS index caption MiniLM 384 chiều và manifest kiểm tra dữ liệu |
| `objects/` | Metadata đối tượng, bổ sung bằng chứng khi xếp hạng |
| `ocr/` | Chữ nhận diện từ ảnh; tùy chọn, cần đúng định dạng và index trước khi bật |

Một số bản dữ liệu hiện có dùng đường dẫn rút gọn, ví dụ
`mapping/L01/L01_V001.csv` hoặc `caption_generator/L01/L01_V001.json`.
Loader hỗ trợ các bố cục này khi mỗi artifact chỉ có **một bản duy nhất**.
Không cần di chuyển hoặc tạo bản sao dataset để chạy demo.

**Số ảnh không phải frame ID:** `001.jpg` tương ứng `n=1`; frame và thời gian
nộp bài phải đọc từ CSV gốc. Không đổi tên ảnh, suy timestamp từ tên file hay
trộn vector của model khác vào cùng index.

Chi tiết định danh và kiểm tra tính nhất quán:
[Cấu trúc dữ liệu AIC](docs/data/DATASET.md).

## Pipeline tìm kiếm

```text
Query VI/EN → LLM planner → query variants + entities
                              ↓
             SigLIP2 ảnh · BM25 caption · MiniLM caption
                              ↓
                Weighted RRF → object evidence
                              ↓
                  200 frame khác nhau → Web / DRES
```

- Visual model: `timm/ViT-B-16-SigLIP2-384`, cấu hình ở `config/models.json`.
- Launcher mặc định dùng Gemini; có thể chọn Groq bằng `-Provider groq`.
  OpenCode fallback là tùy chọn qua `config/planner-fallback.local.json`, theo
  mẫu `config/planner-fallback.example.json`; cần CLI và model đã kết nối.
- OCR là nhánh tùy chọn, **tắt trong cấu hình pipeline mặc định**.
- Có thể tắt Qwen VLM bằng `KIS_DISABLE_VLM=1`; hệ thống vẫn truy hồi 200 frame,
  chỉ bỏ bước chấm lại ảnh. Điểm truy hồi không xác minh toàn bộ chuỗi hành động.
- Giao diện hỗ trợ tìm frame tương tự và xem trước dữ liệu nộp DRES.

## Cài đặt và chạy local

Cần Python tương thích với các phiên bản trong `requirements.txt`, PowerShell
cho launcher Windows và Docker Compose để chạy Milvus. Cài dependency trong môi
trường riêng; bản PyTorch/CUDA cần phù hợp với máy nếu dùng GPU.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

1. Chuẩn bị ảnh, mapping, vector và caption/index trong `dataset/`.
2. Cấu hình API key qua biến môi trường `GEMINI_API_KEY` hoặc `GROQ_API_KEY`.
   Launcher cũng có thể hỏi key với ký tự được ẩn. Không ghi key vào mã nguồn;
   `.env.example` chỉ là mẫu, ứng dụng không tự đọc file `.env`.
3. Khởi động dịch vụ và tạo collection **lần đầu**, sau khi đã có vector:

```powershell
docker compose up -d
python -m tools.build_milvus build
```

4. Chạy web, không dùng VLM chấm ảnh:

```powershell
$env:KIS_DISABLE_VLM = "1"
.\scripts\start_kis.ps1 -Python ".\.venv\Scripts\python.exe"
```

Dùng Groq thay Gemini:

```powershell
.\scripts\start_kis.ps1 -Python ".\.venv\Scripts\python.exe" -Provider groq
```

Không cần tạo lại collection sau mỗi lần mở web. Đảm bảo không đặt cấu hình
worker từ môi trường cũ khi muốn dùng backend local; xem
[ghi chú bảo trì](docs/maintenance/REPOSITORY.md).

- Đề DRES: <http://127.0.0.1:8000/>
- Tự nhập query, không nộp DRES: <http://127.0.0.1:8000/?mode=manual>
