# OCR retrieval độc lập — triển khai và pilot ngày04/11/2026

## School CPU snapshot (15/11/2026)

Nhánh mới bổ trợ Qwen GPU laptop dùng public OCR được chuyển sang CPU trường:
67.882record L01–L09 giữ nguyên bytes, production phủ44.936record trong scope cũ.
Literal-only FTS và OCR evidence riêng mỗi ảnh, snapshot pinned, bonus positive
giới hạn theo base score; không thay caption hoặc giả provenance. Private school
runtime đã bật OCR sau restart được người dùng cho phép, global config vẫn off.
Xem [SCHOOL_OCR.md](SCHOOL_OCR.md) cho coverage, timing, config/rollback và giới
hạn. Strict/public_json readers bên dưới được giữ nguyên validation; không nới
schema provenance strict chỉ vì public OCR không có engine/confidence/hash ảnh.

## Public OCR L04 opt-in (12/11/2026)

Phần này là đường tích hợp **riêng**, không nới schema pixel-OCR strict ở dưới.
`KIS_OCR_FORMAT=public_json` + `KIS_OCR_ROOT` là opt-in bắt buộc; mặc định
`KIS_OCR_FORMAT=strict`, `KIS_OCR_MODE=off`. Không cài dependency mới, chạy OCR,
ghi vào nguồn OCR/caption, hay build FAISS/Milvus/SQLite trên đĩa.

### Input và join

Mỗi `dataset/ocr/L04/L04_V001/001.json` phải là array đúng **một record, bảy trường**:

```json
[{"video_id":"L04_V001","frame_id":null,"keyframe_n":1,"image_path":"data/L04/L04_V001/001.jpg","ocr_text":"LỚP 6A","retrieval_text":"LỚP 6A","status":"ok"}]
```

- Join bằng `(video_id, keyframe_n)` với candidate/caption gốc; `keyframe_n` và
  filename phải khớp. `dataset/mapping/L04/L04_V001.csv` cung cấp `n -> frame_idx`
  (và `pts_time` nếu có). `frame_id` có thể null, nhưng nếu có phải bằng
  `frame_idx`. **Không dùng n-1 hoặc coi n là frame_idx**. Không đọc blob
  caption_mapping bằng một đường dẫn tự đoán để thay CSV. Caption loader vẫn
  kiểm tra đầy đủ triplet/manifest đang có như trước.
- `image_path` đúng group/video/số ảnh; adapter chuẩn hóa slash, padding và đuôi
  ảnh thực tế theo inventory ảnh numeric duy nhất (`jpg/jpeg/png`). Ảnh thiếu
  vẫn unknown, không biến thành OCR âm tính. Không đọc/hash pixel ảnh để dựng
  provenance. Root tương đối được hiểu theo repo root, giống caption root.
- `retrieval_text` trong JSON OCR phải bằng chính xác `ocr_text`, **không phải
  caption**. Không cho text thứ hai lén đi vào retrieval. `ok` phải có chữ;
  `empty/error` phải trống. Schema/type/status/path/duplicate/mapping sai bị từ
  chối, không tự sửa. Không hỗ trợ technical envelope `ocr-sidecar-v1` ở adapter này.
- Không chấp nhận traversal, symlink, junction/reparse hoặc cloud placeholders,
  kể cả ở ancestor. Nguồn được đọc từ các group/video đã chọn, không quét L05/L06
  rồi mới lọc. Source là immutable snapshot: inventory + nội dung JSON/CSV và
  scope được kiểm tra lại khi dùng; thay đổi => báo lỗi, không phục vụ OCR cũ.
  **Restart sau khi sửa/thêm/xóa assets**. Không theo dõi hash nội dung ảnh.
  Việc đọc/hash lại JSON/CSV có overhead; chưa đo latency trên dữ liệu thật.

Nguồn public luôn `verified=false`, `source_format=public_json`,
`provenance=UNVERIFIED_public_OCR_no_image_SHA_or_engine`; image SHA/engine/
confidence là unknown, region unknown. Ngưỡng confidence strict mặc định **không
được áp dụng hay coi là đã pass**: status/diagnostics cảnh báo rõ. Adapter từ chối
`min_confidence` khác mặc định hoặc `ocr_region` khác `all` vì không có capability
đó. Không thêm `verified=true` hoặc hash ảnh giả để hợp thức hóa dữ liệu public.

### Đường đi trong pipeline

`PublicOCRRetriever.search` dùng FTS5 trong RAM, bind SQL và quote literal, rồi
kiểm tra literal thực. Kết quả đi qua **OCR candidate union + score fusion hiện
có**, độc lập với VLM. Không nối OCR vào `retrieval_text` của caption.

Trong mode `retrieve`, lookup riêng gắn OCR cho **mọi potential rerank candidate**,
không chỉ FTS hits, trước Qwen. `ok/empty/error/missing` đều có metadata unverified;
thiếu/lỗi/trống không xóa candidate hoặc chứng minh ảnh không có chữ. Qwen nhận
keyword `ocr` riêng qua cả direct scorer và `LazyVLMReranker`, với capability
`supports_ocr=True`; scorer cũ không hỗ trợ sẽ báo lỗi rõ, không retry bằng cách
nuốt `TypeError`. Mode off vẫn gọi scorer cũ ba đối số và giữ prompt cũ;
`diagnostics` chỉ tìm/log, không đưa OCR vào Qwen hay thay ranking.
Nếu assets đổi giữa FTS và lookup hoặc join bị sai, pipeline bỏ public OCR
boost/union của request đó và giữ baseline candidates, không dùng một phần dữ
liệu đã stale. Mỗi baseline candidate khi đó có OCR status error, không phải
empty/negative.

OCR được đưa vào prompt bằng JSON quoted tối đa 2.048 ký tự text, tách caption;
không xem như instructions/ground truth. Ticker/timestamp/chữ không chứng minh
scene action; single-image Qwen không xác minh sequence. Temporal route vẫn chỉ
dùng OCR làm seeds và không đổi gates. Muốn kiểm tra đường Qwen ảnh đơn, gửi
`temporal:false` như ví dụ dưới.

### Điều kiện trước khi người dùng chạy

- Env `focusguard_ai` đã có sẵn; các model cần dùng đã được chuẩn bị như pipeline
  hiện tại. Không cài/thay env vì adapter chỉ dùng stdlib.
- Bộ **caption InternVL gốc + caption_mapping + FAISS + manifest hợp lệ cho L04**
  ở `KIS_CAPTION_ROOT` (ví dụ dưới dùng `dataset`), cùng CSV mapping và ảnh local
  hoặc image source đã cấu hình. Không tự chuyển sang staging hoặc tạo caption.
- Milvus hiện tại ở `http://localhost:19530`, collection `clip_keyframes`, cần
  metadata model hợp lệ và dữ liệu L04. Scope production mặc định có thể chưa có
  L04: biến `KIS_VIDEO_GROUPS=L04` **không tự thêm vectors**. Không bypass kiểm tra,
  rebuild index, đổi collection hoặc khởi động Milvus trong thay đổi này.
- Web search vẫn cần Groq/query planner như trước. Script hiện tại tự hỏi key
  bằng `Read-Host -AsSecureString` nếu chưa có `GROQ_API_KEY`, rồi dùng loader
  OpenAI-compatible hiện có; không dán key vào code/chat/command history.

### PowerShell: bật public OCR và đưa vào Qwen

Các lệnh sau để **người dùng tự chạy**, chưa được chạy server/inference trong
lượt tích hợp. Dùng trực tiếp Python của env có sẵn, không cần `conda activate`:

```powershell
Set-Location -LiteralPath 'C:\Users\Khanh\Desktop\KISBranche'
$Python = 'C:\Users\Khanh\miniconda3\envs\focusguard_ai\python.exe'
$env:KIS_CAPTION_ROOT = 'C:\Users\Khanh\Desktop\KISBranche\dataset'
$env:KIS_VIDEO_GROUPS = 'L04'
$env:KIS_OCR_MODE = 'retrieve'
$env:KIS_OCR_FORMAT = 'public_json'
$env:KIS_OCR_ROOT = 'C:\Users\Khanh\Desktop\KISBranche\dataset\ocr\L04'
$env:KIS_DISABLE_VLM = '0'
.\start_kis.ps1 -Python $Python
```

Không dùng `-CheckApi` để start app (switch đó gọi planner test thật). Script chỉ
hỗ trợ `-Python`, `-Model`, `-CheckApi`; server module dùng `127.0.0.1:8000`.
Không có cờ `--ocr-root`, `--group` hoặc API filesystem path override.

Mở PowerShell thứ hai để gửi literal cụ thể (thay `6A` bằng chữ muốn tìm):

```powershell
$body = @{ query = 'A classroom with a board'; ocr_text = '6A'; ocr_region = 'all'; temporal = $false } | ConvertTo-Json
$result = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/api/search' -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
$result.ocr | ConvertTo-Json -Depth 8
$result.results | ConvertTo-Json -Depth 12
Invoke-RestMethod -Uri 'http://127.0.0.1:8000/api/status'
```

UI hiện không có ô `ocr_text`; API trên bảo đảm literal đi thẳng vào OCR route,
không phụ thuộc planner có suy ra `needs_ocr` hay không. `ocr_region=content` không
được hỗ trợ bởi public format và trả 422, không âm thầm bỏ constraint.

### Chỉ retrieval (không Qwen), hoặc tắt OCR

Dừng app bằng **Ctrl+C** trong terminal đầu, đổi env trong **chính terminal đó**,
rồi restart. Để OCR vẫn retrieve nhưng tránh Qwen:

```powershell
$env:KIS_DISABLE_VLM = '1'
.\start_kis.ps1 -Python 'C:\Users\Khanh\miniconda3\envs\focusguard_ai\python.exe'
```

Đây **không phải full Qwen pipeline**; visual/caption retrieval và planner vẫn
chạy, không phải chế độ không-model hoàn toàn. Tắt OCR (sau Ctrl+C):

```powershell
$env:KIS_OCR_MODE = 'off'
.\start_kis.ps1 -Python 'C:\Users\Khanh\miniconda3\envs\focusguard_ai\python.exe'
```

Lệnh off giữ nguyên `KIS_VIDEO_GROUPS=L04` và trạng thái disable-VLM đang chọn;
nếu muốn Qwen lại thì đặt `KIS_DISABLE_VLM=0` trước restart. Không cần xóa nguồn.

**Giới hạn chưa giải quyết:** Qwen đã có rủi ro score NaN ở runtime, Groq phụ thuộc
key/quota/model/API, và collection/caption L04 có thể thiếu. Thay đổi này không
sửa các lỗi đó, không chạy server/GPU/Groq/network/index build/benchmark thật,
không khẳng định OCR làm model tốt hơn. Các kết quả kiểm thử mới là fixture/mock
offline, không phải kết quả suy luận thật; phần pilot cũ dưới đây giữ nguyên.

## Kết quả triển khai, không phải kết quả accuracy

Đã nối một nguồn **pixel OCR** riêng vào pipeline. OCR có thể đưa frame/video
chưa có trong visual/caption hits vào candidate union **trước** object/text/VLM
reranking. Không dùng `ocr_text` nằm trong caption sinh tự động làm OCR độc lập.

| File | Vai trò |
|---|---|
| `kis/ocr.py` | Schema, config, literal routing, SQLite FTS5/BM25, candidate fusion |
| `tools/build_ocr.py` | Extract bounded bằng local Paddle models; import/index có checksum |
| `tools/prepare_ocr_pilot.py` | Đóng gói12 ảnh local và notebook CPU tách môi trường |
| `kis/solve_kis.py` | OCR candidate union trước rerank, fallback khi index/literal thiếu |
| `kis/event_retrieval.py`, `kis/candidate_pool.py` | Event OCR seeds, cùng video/frame budget |
| `kis/web_app.py`, `kis/config.py` | Status, raw OCR provenance, manual query/region overrides |

**OCR mặc định off**, text reranker vẫn off. Ngưỡng temporal0,25/0,45, DP, index
visual/caption và nhãn giữ nguyên. Chưa cài Paddle/PaddleOCR hoặc chạy model
OCR thật trên máy chính, chưa có OCR corpus/index thật, chưa đo Hit@K/CER/WER.
Các kiểm thử dùng ảnh/line giả lập trong thư mục tạm không phải kết quả OCR thật.

## Vì sao dùng nguồn riêng?

Caption có thể bỏ sót/suy đoán chữ. Mỗi OCR frame lưu riêng:

- video_id, frame_id, keyframe_n, pts_time từ mapping;
- checksum ảnh, width/height, trạng thái ok/empty/error;
- từng line: text gốc, confidence hoặc unknown, polygon4 điểm, region;
- engine/package version, model name/revision và file hashes;
- manifest của build, checksum SQLite và mapping từng video.

Import phải khớp ảnh thật/mapping, từ chối checksum sai, frame_id bị lặp/ambiguous,
bbox ngoài ảnh, NaN, metadata thiếu hoặc artifact được khai báo generated_caption.
Các kiểm tra này bảo toàn **provenance đã khai báo**, không chứng minh OCR đã đọc
đúng chữ hoặc có người xác nhận. Recognition confidence không phải probability
điều kiện query đã xảy ra; mọi result vẫn `verified=false`.

Index giữ cả line confidence thấp và unknown; chỉ loại chúng khỏi danh sách
match theo config khi search. Empty/error không biến thành negative hoặc caption
giả. OCR không có text hoặc không match thì pipeline cũ được giữ nguyên.

## Tìm chữ và query routing

- Chỉ lấy `ocr_text` do người dùng nhập, hoặc chữ trong ngoặc kép/entity type=text
  **có nguyên văn trong raw query** khi needs_ocr=true.
- Không tự dịch `Tiếng Anh` thành chữ `English` để giả làm text hiện trên bảng.
- Không gửi nguyên mô tả cảnh dài vào OCR hoặc dùng LLM điền text mờ.
- Case-insensitive, Unicode **NFC** và gộp whitespace. Không bỏ dấu Việt,
  không xóa phủ định, chữ số, toán tử, superscript/subscript. NFKC bị tránh vì
  làm x² thành x2. Đây là lexical search, không phải hiểu/so sánh công thức toán.
- FTS5/BM25 tạo candidate, sau đó kiểm tra literal/word boundaries trên line gốc:
  `6A` không tự khớp `16A`, `x=1` không tự khớp `x+1` hoặc `x=10`.
- Query được escape thành FTS literal và SQL parameters, không thực thi toán tử
  FTS/SQL do người dùng nhập.
- V1 chỉ match trên **một line**, chưa nối nhiều line hoặc fuzzy-correct OCR lỗi.
- Tối đa4 literals ×50 unique frame hits; mỗi query scan tối đa1000 line hits
  (top_k×20, trần2000). Work bound không bảo đảm exhaustive recall.

Region được giữ `content`, `overlay`, `unknown`. Default all không xóa ticker.
Adapter Paddle v1 gắn unknown, **không tự đoán bảng/ticker từ caption hoặc vị trí**.
Có thể import region annotation rõ ràng và lọc content/overlay; nếu chưa có
annotation, lọc content có thể không trả gì. Crop/bảng/handwriting/formula OCR
vẫn cần pilot riêng, chưa được thực hiện ở đây.

## Fusion và temporal: ranh giới an toàn

Khi mode=retrieve và có match, score mới là:

```text
(1 - OCR_weight) × base_fusion_score + OCR_weight × 61/(60 + OCR_rank)
```

Weight v1=0,5 chỉ là engineering default opt-in, **chưa được calibration**.
Frame OCR-only có thể vào shortlist chứ không chỉ boost frame visual tìm sẵn.
Lưu `pre_ocr_score`, `base_fusion_score`, `evidence.ocr` và raw `ocr` riêng; không
chèn OCR vào retrieval_text/caption. No hit/off/diagnostics/error giữ score và
candidate order cũ. Scope video của OCR cũng là scope runtime hiện hành.

Với event sequence, chỉ chuyển OCR hit sang event có chứa literal tương ứng
trong action; không gắn global text vào mọi event. OCR seeds thêm video và frame
để local scorer xem xét, vẫn tối đa12 videos/1.500 frames/video. **Không thay
visual/semantic gate hoặc làm một action unsupported thành retrieved chỉ vì
đọc thấy chữ**. Event/temporal result có thể partial và vẫn verified=false.
Chưa có modality-aware support cho whole text-defined event trong v1.

OCR thêm một stream/budget cho event có text; không gọi đó là same-budget quality
gain khi chưa có A/B. Candidate pooling giữ baseline khi không có OCR hit.

## Pilot đã chuẩn bị: chưa chạy OCR model

Files cho người dùng:

- **`dist/kis-ocr-pilot-20261104.zip`**: source allowlist,12 keyframes local, mapping,
  manifest và optional requirements; không weights/captions/qrels/credentials.
- **`notebooks/ocr_pilot_20261104.ipynb`**: Kaggle CPU, Internet on, venv riêng.
- Input đóng băng: `benchmark_results/ocr_pilot_20261104/`.

Kiểm tra gói thực tế đã đạt: ZIP CRC, đúng allowlist **32 files**, source snapshot
khớp workspace, checksums ảnh/mapping/requirements và join frame/timestamp.
Notebook có **4 code cells** compile được, chưa có outputs và ghim đúng manifest.
Không có captions/qrels/credentials/model weights trong ZIP.

| Metadata đóng băng | Giá trị |
|---|---|
| Pilot | 12 keyframes, 12 video, mỗi ảnh 1280×720 |
| ZIP size | 1.862.611 bytes (khoảng 1,78 MiB) |
| Candidate frames hợp lệ trước sampling | 125.447 |
| Bỏ qua do mapping thiếu/ambiguous | 20 |
| Seed | 2026110401 |
| ZIP SHA-256 | `b8372a034bc183f0bb008a9af2ded61312151227200047b467590437baa5e53b` |
| Manifest SHA-256 | `68fcc552f1eaadb4a46d8aa453fb3f0111df1c3dc3b84b980a60dfc254b30a78` |
| Notebook SHA-256 | `5a93a7625d3a6ea3a31cbbac60fca02bbccbd408b5ae67f2c84ac20551cfdbf9` |

Sampling chỉ dùng metadata, seed2026110401, không đọc caption/nhãn để chọn ảnh.
Ảnh local có thể đã được quan sát trong các lượt trước: đây là **runtime/dev pilot**,
không phải independent holdout hay benchmark accuracy mới. Không có ground truth
OCR/transcription mới được tự tạo.

### Model/runtime đã ghim cho pilot

- PaddleOCR3.7.0, PaddleX3.7.0, PaddlePaddle CPU3.3.1.
- NumPy1.26.4, Pillow11.1.0, huggingface-hub1.27.0; Python3.10–3.12.
- `PaddlePaddle/PP-OCRv5_mobile_det` @
  `0d63e78e2b680928f6b1747d76a08db6e645efb7`.
- `PaddlePaddle/latin_PP-OCRv5_mobile_rec` @
  `ab2cd5cc5fa6309be2e5acdfe66eca2c2c127d57`.
- Explicit PP-OCRv5, local detection/recognition dirs, CPU; tắt doc orientation,
  unwarping và textline orientation để không kéo thêm model.
- Package/model metadata được đối chiếu nguồn công khai, **chưa xác minh inference
  thật hoặc compatibility toàn bộ transitive dependencies trên Kaggle**.

PaddleX3.7 yêu cầu NumPy<2,4 trong khi app hiện dùng2,4,4, vì vậy **không cài OCR
requirements đè lên focusguard_ai**. Notebook tạo venv không dùng site-packages
host, pin direct dependencies, lưu pip freeze; không phải lockfile hermetic cho
mọi transitive dependency. Không nâng/hạ package ngẫu nhiên nếu gặp lỗi.

Official multilingual docs liệt kê Vietnamese cho latin_PP-OCRv5_mobile_rec.
Model card có phần mô tả không nhất quán (tiêu đề/ngôn ngữ); không lấy con số
accuracy trên model card làm metric KIS, không hứa đọc đúng bài giảng viết tay.

## Người dùng cần làm gì trên Kaggle?

1. Tạo **private Dataset** từ `dist/kis-ocr-pilot-20261104.zip`.
2. Tạo notebook/import `notebooks/ocr_pilot_20261104.ipynb`, Add Input dataset đó.
3. **Bật Internet**, để accelerator **None/CPU**; không cần T4 hay API key.
4. Chạy các cell theo thứ tự. Cell đầu xác minh manifest/source/image/mapping
   hashes; hỗ trợ ZIP và trường hợp Kaggle tự giải nén. Sau đó mới tạo venv,
   cài dependencies riêng, tải đúng2 checkpoint và OCR tối đa12 ảnh.
5. Tải **`ocr-pilot-results.zip`** từ link cuối và đặt vào `dist/` trên máy,
   rồi báo lại. Không đổi tên thành vlm-results.zip.

Nếu lỗi cài package/model initialization: gửi traceback, không cài requirements
Windows trên Kaggle, không đổi sang PaddleOCR-VL/PP-OCRv6 mặc định hoặc unpin
NumPy để ép chạy. Nếu Python ngoài3.10–3.12, dừng và báo phiên bản. Nếu một số
frame lỗi, ZIP vẫn giữ status=error/complete=false để kiểm tra, không điền text
hoặc nhãn negative. Không chạy lại chỉ các text/score không ưng rồi trộn chung.

Sau khi nhận ZIP, kiểm tra provenance/coverage và đọc kết quả trước; **chưa tự
promote OCR**. Có thể cần transcription/visible-text qrels do người xem ảnh xác
nhận để đo CER/WER hoặc text-critical retrieval. Không sửa nhãn human-v3 cũ.

## Index/import và bật thử sau khi kết quả được kiểm tra

Ví dụ cho output đã được nhập an toàn, **không chạy lệnh này trước khi có detections thật**:

```powershell
python -B -m tools.build_ocr index --input benchmark_results/ocr_pilot_import_20261104/ocr_detections.json --image-root benchmark_results/ocr_pilot_20261104/images --mapping-dir benchmark_results/ocr_pilot_20261104/mapping --output dataset/ocr_pilot_v1_20261104
```

Build versioned, không ghi đè index cũ. App mở SQLite read-only, lazy-check checksum
index và mapping; mismatch/missing thì OCR fallback, không làm hỏng visual route.
Pilot chỉ có12 frames, không đại diện coverage535 video.

```powershell
$env:KIS_OCR_INDEX = "dataset/ocr_pilot_v1_20261104/index.sqlite3"
$env:KIS_OCR_MODE = "diagnostics"  # search/log OCR, không đổi ranking
# Chỉ khi chấp nhận chạy opt-in candidate union:
# $env:KIS_OCR_MODE = "retrieve"
.\start_kis.ps1
```

`/api/status` có config/available/loaded/error. `/api/search` trả OCR diagnostics,
timing và result.ocr chứa text/bbox/confidence/source/image checksum/build ID.
Manual API override, không đổi server policy off:

```json
{"query":"Tìm ảnh bảng lớp","ocr_text":"6A","ocr_region":"all"}
```

OCR pilot/search core không cần planner/API key. Toàn bộ web search vẫn cần
query planner được cấu hình như trước; không gửi API key trong chat. UI hiện
chưa có ô OCR riêng hay vẽ bbox; có thể dùng raw quoted text khi planner gắn
needs_ocr hoặc dùng API override. Không render raw OCR text thành HTML.

## Kiểm thử và giới hạn

Kết quả suite cuối: **391 tests, 377 passed, 14 skipped, không failures/errors**.
OCR tests riêng: **24/24 passed**. Đây là kiểm tra code/fixtures, không phải OCR
recognition accuracy. Hash human annotations, frozen human labels và
`dist/vlm-results.zip` đã đối chiếu và giữ nguyên; mặc định OCR/text off, temporal
visual=0,25/semantic=0,45, pool12 và condition_evidence=off không đổi.

Đã kiểm tra bằng fixture: Unicode/dấu Việt/numbers/math/superscripts, FTS escaping,
regions/unknown/low confidence, read-only/checksums, lỗi input/engine, bounded
extract, archive/expanded notebook bootstrap, candidate union/API và temporal
seed không vượt gate. Đã sửa đóng SQLite connections rõ ràng để không khóa file
trên Windows; không xử lý bằng cách bỏ qua lỗi cleanup.

Đây là code/integration và gói pilot sẵn chạy, **không phải thông báo OCR đã tăng
accuracy**. Không có local model inference, OCR corpus build, caption generation,
re-embedding hoặc thay Milvus trong lượt này. Giữ riêng Kaggle full và caption-only
bundles; chỉ thêm dependency module stdlib vào allowlist code, không thêm OCR
runtime vào hai profile đó hoặc ghi đè ZIP đã công bố.

Nguồn API/model được kiểm tra04/11/2026:
- https://www.paddleocr.ai/main/en/version3.x/pipeline_usage/OCR.html
- https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/algorithm/PP-OCRv5/PP-OCRv5_multi_languages.en.md
- https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det
- https://huggingface.co/PaddlePaddle/latin_PP-OCRv5_mobile_rec
- PyPI metadata paddleocr3.7.0/paddlex3.7.0/paddlepaddle3.3.1.
