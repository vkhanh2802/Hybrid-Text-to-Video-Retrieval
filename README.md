# KIS Framefinder

Tìm kiếm trên **L15, L16, L19, L20, L21**. Caption đang dùng là **v2**;
caption/index v1 được lưu riêng, giữ nguyên dữ liệu.

## Pipeline

```text
Query → Gemini tạo variants + entities
                    ↓
      ┌─────────────┼────────────────┐
      ↓             ↓                ↓
 SigLIP/Milvus   BM25/v2       MiniLM/FAISS/v2
      └─────────────┼────────────────┘
                    ↓
        Weighted RRF → Object evidence
                    ↓
      Qwen rerank khi có ảnh → Top 20
```

RRF: `k=60`, visual/BM25/semantic = `0.5/0.2/0.3`; top 100 mỗi nhánh,
Milvus `nprobe=128`, Qwen chấm top 20. Web và notebook dùng chung `SearchService`.

## Cấu trúc

```text
kis/                         # pipeline đang chạy
  config.py, paths.py         # cấu hình và đường dẫn
  visual.py, text.py          # SigLIP/Milvus và BM25/MiniLM
  objects.py, ranking.py      # object evidence, RRF, ranking helpers
  solve_kis.py, web_app.py    # điều phối và API
tools/                       # ingest/index, chuẩn bị Kaggle, đánh giá
config/                      # pipeline.json, models.json, object_labels.json
tests/                       # unit tests
notebooks/runbranch.ipynb     # tìm kiếm qua cùng pipeline với web
web/                         # giao diện
docs/                        # hướng dẫn Kaggle

dataset/
  caption_generator_v2/
    captions/<group>/*.json   # caption v2, một bản duy nhất
    caption_mapping/<group>/*.json
    index/<group>/*.index     # MiniLM + SHA-256/model metadata
    index/<group>/*.meta.json
  mapping/<group>/*.csv
  clip-features-32/<group>/*.npy
  objects/<group>/<video_id>/*.json
  data/<group>/<video_id>/*.jpg # ảnh phục vụ preview/rerank khi có
caption_generator/           # v1 được bảo toàn
caption_mapping/, index/     # artifact v1 được bảo toàn
```

| Nhóm | Visual vector | Caption v2 |
|---|---:|---:|
| L15 | 7.919 | 7.919 |
| L16 | 9.011 | 9.011 |
| L19 | 7.746 | 7.746 |
| L20 | 8.192 | 8.192 |
| L21 | 7.800 | Chưa có |
| **Tổng** | **40.668 / 150 video** | **32.868 / 121 video** |

Scope trong `config/pipeline.json`; build và search chặn các nhóm sau L21.
L21 hiện tham gia visual/object retrieval. Khi có caption v2 L21, lệnh nạp bên
dưới sẽ thêm vào text retrieval. Nạp caption/vector không yêu cầu keyframes.

## Chạy trên máy hiện tại

Tại project root, dùng môi trường `focusguard_ai`:

```powershell
docker start milvus-etcd milvus-minio milvus-standalone
$env:GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
.\start_kis.ps1
```

Mở **http://127.0.0.1:8000**. Có thể chạy bằng `python -m kis.web_app`.
Ảnh ngoài project/Drive Desktop được cấu hình qua `KIS_DATA_ROOT`, trỏ đến thư
mục chứa các nhóm `L15`, `L16`, …; caption giữ đường dẫn logic `data/...`.
Có thể lưu đường dẫn riêng cho máy trong `config/images.local.json` dưới khóa
`data_root`; file này được Git bỏ qua, còn `KIS_DATA_ROOT` luôn được ưu tiên.
Nguồn ảnh chỉ nhận JPEG/PNG có tên số trong layout `<group>/<video_id>/001.jpg`.
Các thư mục `video_*`, video gốc, ZIP và nhóm ngoài scope không được dùng để đọc
keyframe; API cũng từ chối số ảnh không có trong mapping.

### Ảnh Drive không giữ cache trên đĩa

Máy này có reader riêng `kis/local_images.py` và test tương ứng, đều được
`.gitignore` loại khỏi Git. Đặt `KIS_DRIVE_FOLDER_ID` là ID thư mục ảnh chứa
`L15`, `L16`, … và `GOOGLE_APPLICATION_CREDENTIALS` tới credentials Google có
quyền đọc thư mục đó (Drive API phải được bật; với service account, share folder
cho email của account ở quyền Viewer).
Cũng có thể đặt `image_source: "drive_api"`, `drive_folder_id` và
`credentials_file` trong `config/images.local.json` để web/notebook tự nạp cấu
hình của máy. Đường dẫn credentials tương đối được tính từ project root.

Chế độ này đọc **trực tiếp Drive API**: ảnh cho Qwen dùng luồng RAM, đóng khi chấm
xong kể cả khi lỗi; ảnh web trả từ RAM với `no-store`, giữ Blob URL trong trang
và giải phóng khi đổi kết quả/đóng trang. Chế độ `KIS_DATA_ROOT` qua Drive Desktop
vẫn do Drive quản lý cache. Reader không xóa file trên cloud hoặc cache DriveFS.
Không có reader riêng thì bỏ `KIS_DRIVE_FOLDER_ID` để dùng ảnh local như thường.

## Nạp dữ liệu và kiểm tra code

```powershell
python -m tools.ingest                  # chỉ build index thiếu/đã thay đổi
python -m tools.ingest --rebuild-visual # khi thêm/thay visual features
python -m unittest discover -s tests -t .
```

Caption mới có thể nhập bằng `--source <folder-đã-giải-nén>`. Lệnh nạp chuẩn hóa
v2, kiểm tra mapping/index và tái sử dụng index hợp lệ; không chạy smoke test.
Khởi động lại web/kernel sau khi nạp. Trạng thái: `GET /api/status`;
tìm kiếm: `POST /api/search` với `{"query":"mô tả cảnh"}`.

Chuẩn bị trên Kaggle: [hướng dẫn hai profile](docs/KAGGLE.md).
Đóng gói bằng `python -m tools.package_kaggle`.
