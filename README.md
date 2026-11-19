# KIS Framefinder

Tìm kiếm trên **L01–L03 và L07–L21**. Caption và index đang dùng là **v2**,
nằm trực tiếp trong `dataset/` với các thư mục độc lập.

## Pipeline

```text
Query → Groq tạo variants + entities
                    ↓
      ┌─────────────┼────────────────┐
      ↓             ↓                ↓
 SigLIP/Milvus   BM25/v2       MiniLM/FAISS/v2
      └─────────────┼────────────────┘
                    ↓
        Weighted RRF → Object evidence
                    ↓
      Qwen rerank tối đa 20 ảnh đầu → Đủ 100 frame khác nhau để chọn nộp
```

RRF: `k=60`, visual/BM25/semantic = `0.5/0.2/0.3`; top 100 mỗi nhánh,
Milvus `nprobe=128`, Qwen chấm top 20. Web và notebook dùng chung `SearchService`.
Web luôn giới hạn LLM/VLM ở 20 ứng viên đầu trước khi rerank, kể cả profile
`verified40`; response thành công trả đúng 100 frame khác nhau, không tăng budget
LLM theo số card. Thiếu dữ liệu sau truy hồi mở rộng thì báo lỗi, không padding.
Card phân biệt điểm LLM hợp lệ, lỗi/thiếu ảnh và frame chỉ có điểm truy hồi.

**Query có chuỗi sự kiện:** nhánh event retrieval + temporal DP dùng embedding
và timestamp sẵn có, trả evidence span và frame gợi ý mỗi bước ở phần bổ sung,
không thay thế danh sách 100 frame, không dùng Qwen ảnh đơn để xác nhận cả chuỗi.
Danh sách frame vẫn chạy pipeline trên. Cấu hình/tắt bật,
giới hạn và hướng dẫn A/B: [docs/TEMPORAL_RETRIEVAL.md](docs/TEMPORAL_RETRIEVAL.md).
Các ngưỡng temporal là pilot, chưa có benchmark chất lượng mới trên corpus thật.

**Reranker nhẹ trước VLM:** đã có module CPU cross-encoder cho ảnh đơn và caption
từng event trong complete temporal path; có budget, fallback và API diagnostics.
Mặc định `off`, không tự tải model. Hướng dẫn:
[docs/TEXT_RERANKER.md](docs/TEXT_RERANKER.md). Đã đo CPU thật, gain/loss mixed nên
chưa bật mặc định: [kết quả](docs/TEXT_RERANKER_CPU_RESULTS_20261104.md).
Caption/logit không phải bằng chứng xác nhận nội dung ảnh/chuỗi.

**OCR retrieval độc lập:** đã có pixel-OCR schema/provenance, SQLite FTS5/BM25,
candidate union và API diagnostics; mặc định `off`, không dùng OCR do caption sinh
làm nguồn độc lập. Đã chuẩn bị pilot **12 ảnh/12 video** chạy Kaggle CPU trong
venv riêng; **chưa chạy OCR model thật hoặc đo accuracy**. Gói, notebook và các bước:
[docs/OCR_RETRIEVAL.md](docs/OCR_RETRIEVAL.md).

**Public OCR L04 (opt-in):** hỗ trợ JSON array 7 trường tại `dataset/ocr/L04`,
FTS5 trong RAM + OCR riêng cho Qwen, không nối vào caption và không rebuild index.
Nguồn này **UNVERIFIED** (không image SHA/engine). Mặc định vẫn OCR off/strict.
Lệnh PowerShell với env `focusguard_ai`, API và cách tắt lại:
[Public OCR L04](docs/OCR_RETRIEVAL.md#public-ocr-l04-opt-in-12112026).

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
  caption_generator/<group>/<video_id>/<video_id>.json
  caption_mapping/<group>/<video_id>/<video_id>.json
  index/<group>/<video_id>/<video_id>.index        # MiniLM
  index/<group>/<video_id>/<video_id>.meta.json    # SHA-256/model metadata
  mapping/<group>/<video_id>/<video_id>.csv
  clip-features-32/<group>/<video_id>/<video_id>.npy
  objects/<group>/<video_id>/*.json
  ocr/<group>/<video_id>/*.json
  preparation/<group>/<video_id>/<video_id>.json
  data/<group>/<video_id>/*.jpg      # ảnh phục vụ preview/rerank khi có
```

Đây là layout canonical mới: **loại dữ liệu / nhóm / video / file**. Giữ nguyên
tên file và nội dung/mapping/model metadata. Loader vẫn đọc layout cũ nếu chỉ có
một bản; nếu có cả cũ và mới cho cùng artifact trong một root thì báo duplicate,
không tự chọn nhầm. Công cụ chuẩn hóa tạo **root mới**, không xóa nguồn. Dataset
gốc laptop được giữ nguyên; bộ chuẩn hóa trên máy trường là một bản riêng.
Hướng dẫn: [docs/DATASET_LAYOUT.md](docs/DATASET_LAYOUT.md).

| Nhóm | Visual vector | Caption v2 |
|---|---:|---:|
| L01 | 7.339 | 7.339 |
| L02 | 7.616 | 7.616 |
| L03 | 7.119 | 7.119 |
| L07 | 7.364 | 7.364 |
| L08 | 7.725 | 7.725 |
| L09 | 7.773 | 7.773 |
| L10 | 7.873 | 7.873 |
| L11 | 7.497 | 7.497 |
| L12 | 7.951 | 7.951 |
| L13 | 7.214 | 7.214 |
| L14 | 7.094 | 7.094 |
| L15 | 7.919 | 7.919 |
| L16 | 9.011 | 9.011 |
| L17 | 7.158 | 7.158 |
| L18 | 8.405 | 8.405 |
| L19 | 7.746 | 7.746 |
| L20 | 8.192 | 8.192 |
| L21 | 7.800 | Chưa có |
| **Tổng** | **138.796 / 535 video** | **130.996 / 506 video** |

Scope trong `config/pipeline.json`; build và search dùng cùng danh sách nhóm,
giới hạn đến L21.
L21 hiện tham gia visual/object retrieval. Khi có caption v2 L21, lệnh nạp bên
dưới sẽ thêm vào text retrieval. Nạp caption/vector không yêu cầu keyframes.

## Chạy trên máy hiện tại

Hướng dẫn từng bước từ mở PowerShell: [docs/WINDOWS_QUICK_START.md](docs/WINDOWS_QUICK_START.md).

Tại project root, dùng môi trường `focusguard_ai`:

```powershell
docker start milvus-etcd milvus-minio milvus-standalone
.\start_kis.ps1
```

Mở **http://127.0.0.1:8000**. Có thể chạy bằng `python -m kis.web_app`.
Web dùng **duy nhất đề chính thức từ DRES**, không có ô nhập query thay thế đề
hoặc tải video. Với đề video có ô **Text bổ sung** tùy chọn, tối đa 2000 ký tự;
query truy hồi trộn 70% text + 30% video.
Đăng nhập DRES, chờ đề được phát rồi bấm **Tìm theo đề DRES**. Backend tự chọn
text/video của task hiện tại, trả tối đa 100 frame. Khối kết nối tự ẩn sau đăng nhập.
API xử lý text query dùng **Groq** với model
`openai/gpt-oss-20b`; key được nhập ở terminal, giống cách cấu hình Gemini trước đây.
Groq mặc định dùng `json_schema` strict, temperature `0`; lần đầu và lần khôi
phục đều dùng cùng schema và mẫu JSON đầy đủ. `attributes` gửi tới API là danh
sách cặp `{key, value}`, sau đó được chuẩn hóa về object mà pipeline sử dụng.

Ứng dụng có một bước chuẩn hóa chung trước validation: entity dạng chuỗi giữ
nguyên nội dung và được biểu diễn bằng `{type: "concept", value: "...",
attributes: {}}`; kiểu `[type, value, attributes]` được chuyển sang object nếu
type hợp lệ. Thuộc tính số/bool được chuyển sang chuỗi, placeholder rỗng được bỏ.
Không tự đoán tên loài, màu sắc hoặc lớp object cho entity chưa có type.

Nếu Groq trả `json_validate_failed`, ứng dụng thử khôi phục từ `failed_generation`
chỉ khi dữ liệu đủ trường và qua toàn bộ validation; nếu chưa đạt thì sinh lại
tối đa một lần, vẫn dùng schema strict. Mỗi query dùng tối đa hai lần gọi trong
luồng này. Lỗi key, quota, mạng hoặc cấu hình schema được báo trực tiếp; kết quả
chưa hợp lệ không được lưu hay đưa vào retrieval.

### Video đề chính thức và API video tương thích

Video đề lấy trực tiếp qua bridge DRES, không phải tải lại bằng tay. Nhánh chính
thức nhận clip 0,1–60 giây, tối đa 24 MiB; lấy tối đa 6 frame trải trên clip
(không cắt ngầm video 5 giây còn 3 giây). Clip ngoài giới hạn được báo lỗi rõ.
Giới hạn chung: 8.294.400 pixels (4K hoặc portrait tương đương), 120 FPS.
Cần `ffmpeg` và `ffprobe` trên PATH; không cần
`python-multipart`, PyAV hoặc model mới cho nhánh này.

```text
Video đề DRES → FFmpeg (tối đa 6 frame trải đều, tối đa 2 frame/giây)
             → SigLIP image encoder hiện có
             → mean pool + L2 normalize
             → Milvus → tối đa 100 ứng viên video ───────────┐
                                                          │
Text DRES + text bổ sung → pipeline text đầy đủ             │
  Groq → SigLIP + BM25 + semantic caption                   │
  → object/OCR/text reranker/temporal theo cấu hình         │
  → tối đa 100 ứng viên text ──────────────────────────────┤
                                                          ▼
                         Weighted RRF: 70% text + 30% video
                         → đủ 100 frame khác nhau → VLM xét top 20 một lần

Chuỗi temporal → temporal_results riêng, không thay thế danh sách 100 frame
```

Video **không có text kèm theo** là query theo tương đồng hình ảnh, không gọi
Groq, không chạy OCR/ASR/Qwen hoặc xác minh chuyển động/thứ tự sự kiện. Khi DRES
đã phát câu text hoặc người dùng nhập text bổ sung, text chạy **cùng pipeline
với đề text**, rồi hợp nhất hai danh sách bằng weighted RRF (k=60), **70% text +
30% video**. Qwen chỉ so ảnh với text cho tối đa 20 ứng viên đầu sau fusion,
không chấm lặp ở nhánh text và không đọc toàn clip. Khi nhánh text trả chuỗi
temporal, không dùng VLM ảnh đơn xác minh chuỗi. Tổng text tối đa 8000 ký tự,
riêng text bổ sung tối đa 2000; quá tổng giới hạn báo lỗi, không cắt ngầm.
Caption corpus tham gia retrieval nhánh text và bổ sung metadata cho hit video.
Không rebuild index; planner lỗi không âm thầm fallback thành video thuần.
Upload được ghi bằng tên ngẫu nhiên trong thư mục tạm hệ thống, xóa trước
inference; frame mẫu trong RAM được đóng sau request. Browser dùng Blob URL cho
preview, giải phóng khi đổi/dừng task, mất kết nối hoặc rời trang. Âm thanh không
được sử dụng.

API raw-video cũ vẫn giữ để tương thích tooling/kiểm thử (1–3 giây, 20 MiB),
nhưng **frontend không dùng route này**. Nhận raw bytes, không phải multipart:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/search/video `
  -ContentType 'video/mp4' -InFile '.\clip-2s.mp4'
```

`GET /api/status` bổ sung `input_modes` và `video_input`. Response video có
`input_type="video"`, `video_query`, `combined_text_query`, `video_text_guidance`
(với trọng số thật) và `fusion_method="visual_mean_pool"` hoặc
`"video_text_pipeline_weighted_rrf"` (thêm `+text_vlm_rerank` nếu chấm thành công);
kết quả video thuần không chấm LLM ghi
`vlm_status="skipped_video_query"`; frame có điểm/lỗi Qwen giữ trạng thái thật.
Không coi điểm ảnh đơn là xác minh toàn clip.

### Kết nối và nộp DRES

Hướng dẫn cấu hình, xác nhận payload và giới hạn: [docs/DRES.md](docs/DRES.md).
Frontend dùng POST `/api/dres/search` gửi `task_id`, tùy chọn `video_description`
cho đề video; không nhận text/video thay thế đề. Backend ghép **toàn bộ câu text
DRES đã phát** (không chỉ câu đầu), đọc lại đề, kiểm tra task/hint sau inference
và bỏ kết quả nếu DRES đã chuyển câu. Giao diện không polling: bấm **Làm mới đề**
để đọc câu mới, hoặc bấm **Tìm** để vừa cập nhật vừa tìm theo toàn bộ câu đã phát,
không cần reset/kết nối lại. Mỗi response thành công có đúng 100 frame khác nhau;
thiếu ứng viên sẽ mở rộng truy hồi có giới hạn rồi báo lỗi nếu không đủ, không
padding frame. Temporal hiển thị riêng, không làm giảm số frame chính.
Phần kết nối ẩn sau đăng nhập, vẫn giữ chọn frame trên từng card
và dialog xác nhận nộp. Không tự nộp bài khi tìm kiếm. Cần khởi động lại server
sau khi cập nhật code.

Đã có worker CPU Linux của trường qua OpenVPN + SSH: retrieval/ảnh trên server,
Qwen GPU và DRES/Groq trên laptop. Chạy `.\start_school.ps1` sau khi deploy worker;
nguồn local/Drive cũ không tự đổi.
Hướng dẫn: [docs/SCHOOL_WORKER.md](docs/SCHOOL_WORKER.md). Lượt đo hybrid gần nhất
giảm warm median từ 76,62 xuống 41,46 giây/query nhưng còn ảnh L18/L21 thiếu và
một Qwen invalid score; xem báo cáo, không coi là PASS sạch hoặc cam kết latency.

### Nhập Groq API key

Khi chạy `.\start_kis.ps1`, script ưu tiên `GROQ_API_KEY` trong terminal hiện tại;
nếu chưa có, đọc biến `GROQ_API_KEY` ở **User Environment Variables của Windows**.
Vì đọc User scope trực tiếp, terminal mới do một ứng dụng đang mở tạo ra vẫn lấy
được key đã lưu, không cần khởi động lại OpenCode. Không in key ra console hoặc
ghi vào mã nguồn/Git. Biến User lưu lâu dài trên tài khoản Windows, không phải
kho bí mật được mã hóa; chỉ dùng trên máy/tài khoản đáng tin cậy.

Nếu cả hai nơi đều chưa có key, script hỏi bằng ô nhập ẩn ký tự; key nhập ở ô
này chỉ dùng cho phiên terminal, không tự lưu lâu dài. Muốn nhập key khác tạm
thời, bỏ qua cả key hiện tại và key đã lưu: `.\start_kis.ps1 -PromptForKey`.

Hoặc đặt biến environment trước khi khởi động:

```powershell
$env:GROQ_API_KEY = "YOUR_GROQ_API_KEY"
.\start_kis.ps1
```

Đổi model khi cần:

```powershell
.\start_kis.ps1 -Model "openai/gpt-oss-120b"
```

Kiểm tra riêng API và form JSON mà không nạp model tìm kiếm hoặc ghi file:

```powershell
.\start_kis.ps1 -CheckApi
```

Nhập key khi được hỏi, sau đó dán query tại `Query:`. Kết quả hợp lệ có đầy đủ
`query_variants`, `visual_description`, `entities`, `needs_ocr`, `needs_asr`,
`question`, `events`, `temporal_constraints` và `stream_queries`.

Notebook `notebooks/runbranch.ipynb` cũng hỏi `GROQ_API_KEY` bằng `getpass` nếu
kernel chưa có key. CLI `python -m kis.structure_query "mô tả cảnh"` đọc cùng biến
environment. `.env.example` là mẫu biến môi trường; ứng dụng không tự nạp `.env`.

Client HTTP dùng CA bundle của `certifi` và xác minh chứng chỉ/hostname HTTPS.
Điều này tránh lỗi `ASN1: NOT_ENOUGH_DATA` từ kho chứng chỉ Windows bị lỗi. Nếu
mạng dùng CA riêng, trỏ `SSL_CERT_FILE` đến CA bundle được cấp cho mạng đó.
Lỗi TLS, DNS, timeout và HTTP từ provider được báo riêng để dễ chẩn đoán.

### Thư mục ảnh

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
