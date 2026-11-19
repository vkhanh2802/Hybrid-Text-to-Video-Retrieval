# Kaggle: PaddleOCR + padding, self-contained v4 / sidecar v1

## Phạm vi và thao tác chính xác

1. **Import notebook** `notebooks/kaggle_ocr_paddle_padding.ipynb` bằng giao diện
   Kaggle. Notebook v4 tự chứa code/config/schema; **không upload/attach package ZIP**.
2. **Attach duy nhất keyframes Dataset**, chọn một folder cho một lần chạy. Không cần
   dataset/caption production, Drive, token hay Milvus. Nhiều video trong folder
   đó được hỗ trợ, kể cả nhiều group.
3. Bật **Internet** trong Kaggle Settings để lấy pinned wheels/models. CPU mặc
   định, không cần GPU; bật GPU nếu có nhu cầu và profile/wheel đã kiểm chứng
   (xem giới hạn GPU bên dưới). Runtime venv không thay package của kernel.
4. Chỉ sửa `INPUT_ROOT` trong code cell đầu, ví dụ `/kaggle/input/my-keyframes/L99`,
   rồi **Run All**. Cell đầu chỉ có assignment, không có biến package/profile cần sửa.
   Không cần sửa code khi đổi group/video. Dataset slug bất kỳ được chấp nhận
   nếu bên dưới có thư mục video/group đúng mẫu. Không thể suy luận ID thì dừng,
   báo cần trỏ path đến thư mục có ID; không đoán ID từ tên máy/dataset slug.
5. Cell cuối validate, tạo ZIP có checksum manifest và hiện **link ZIP + `.sha256`**.
   Download ZIP trước khi kết thúc session. `coverage.json` nêu rõ phần thiếu/lỗi;
   validate thành công không có nghĩa là coverage/accuracy hoàn hảo.

Cell có thể chạy lại. Thư mục run trong `/kaggle/working/kis-kaggle-ocr-v4/<payload_hash>`
được **explicit resume** khi có manifest. Tạo session mới/di chuyển run cũ nếu
đổi input/config/model. Không ghi đè ZIP cũ; tên ZIP kết quả có hash manifest.

## Self-contained bootstrap và tái tạo notebook

Notebook là deliverable chính; chỉ cần import một file `.ipynb`. Payload nội tại
JSON/base64 chỉ gồm đúng sáu relative POSIX files:

```text
config/ocr-kaggle-py313.json
config/ocr-kaggle.json
config/requirements-kaggle-ocr-py313.txt
config/requirements-kaggle-ocr.txt
schemas/ocr-sidecar-v1.schema.json
tools/kaggle_ocr.py
```

Trước mọi write/import, bootstrap kiểm tra allowlist/membership/order, path,
base64 canonical, size từng file/tổng, UTF-8/LF, SHA256 từng file và SHA256 payload.
Payload/notebook giới hạn ≤1 MiB. Materialize vào
`/kaggle/working/kis-kaggle-ocr-v4/<payload_hash>/code`; kiểm tra ancestor/target
symlink, junction, Windows reparse point, file dư và cached bytes bị sửa; không
ghi đè source bị tamper. Sau đó import **exact runner path** vừa materialize,
không import repo production hay runner từ `sys.path`/cwd. Không tìm/đọc ZIP
v1/v2/v3, package Dataset hay `MANIFEST.sha256` bên ngoài. Chỉ bước advanced offline
mới tìm folder cache có tên `ocr-wheelhouse`/`ocr-models`, không dùng chúng làm code.

Không embed credentials, keyframes, weights, docs hay outputs. SHA256 là kiểm tra
integrity, **không phải chữ ký**: chỉ import notebook từ nguồn tin cậy. Các check
filesystem phục vụ session một người dùng; không phải sandbox chống tiến trình
khác đồng thời thay filesystem trong lúc check/write.

Generator stdlib, sorted payload, UTF-8/LF canonical, cell IDs cố định, không
timestamp/random ID; code cells compile được và outputs rỗng:

```bash
python -B tools/build_kaggle_ocr_notebook.py
python -B tools/build_kaggle_ocr_notebook.py --check
python -B -m unittest tests.test_kaggle_ocr tests.test_caption_artifacts
```

`--check` regenerate trong temp, so byte-identical với notebook hiện tại, validate
payload allowlist/hashes/size và compile tất cả code cells; không cài package,
không network/Paddle/model. Khi thay một trong sáu source/config files phải
regenerate notebook. Config advanced (padding/GPU) cũng cần regenerate bản riêng,
không sửa base64/hash thủ công. Flow thông thường vẫn chỉ sửa `INPUT_ROOT`.

Optional source distribution: `dist/kis_kaggle_ocr_paddle_padding_v4_py313.zip`
và `.zip.sha256`. ZIP chỉ để lưu/chia sẻ source; lấy notebook ra để import,
**không attach ZIP này vào Kaggle**. ZIP v1/v2/v3 cũ được giữ nguyên, không overwrite.

## Layout và identity

Các layout được nhận diện (ví dụ dùng group bất kỳ, không whitelist L04/L05/L06):

```text
INPUT_ROOT=L99/                 INPUT_ROOT=L99_V003/
  L99_V003/                      1.jpg
    1.jpg                        002.png
    002.png                      mapping.csv  # optional
    mapping.csv  # optional

INPUT_ROOT=my-upload/           INPUT_ROOT=my-upload/
  L99_V003/1.jpg                  L99/L99_V003/1.jpg
  L100_V001/10.jpg                L100/L100_V001/10.jpg
```

Chỉ JPEG/PNG, stem `[0-9]+`, sắp thứ tự theo `(video_id, integer keyframe_n)`.
`001.jpg` và `1.png` là duplicate identity, nên fail. Thư mục con bất ngờ,
symlink/junction, ID group mâu thuẫn, filename sai hoặc file khác trong video
(trừ `mapping.csv`) bị từ chối. Không scan toàn repo, không đọc caption.

Optional `mapping.csv` trong từng video có header `n,frame_idx,pts_time`:
`n` là keyframe filename, `frame_idx` là frame của video gốc, `pts_time` là giây.
Không có mapping: `mapping=null`, tuyệt đối không lấy `n` làm `frame_idx`.
Mapping có `n` nhưng ảnh chưa upload: tạo record `missing`, image fields null.
Không suy diễn các số filename bị gián đoạn là ảnh thiếu nếu không có mapping.

## Dependencies, model provenance và offline

Kaggle không cho chọn Python version; notebook đọc `sys.version_info` của kernel
và chọn profile tự động. **Venv chỉ cô lập package, kế thừa Python của kernel,
không đổi Python 3.13 thành 3.12.** CLI dùng cùng selector và version/import checks.

| Interpreter thực tế | Requirements / config | Exact pins |
|---|---|---|
| Python 3.10–3.12 | `config/requirements-kaggle-ocr.txt` / `config/ocr-kaggle.json` | paddleocr 3.7.0, paddlex 3.7.0, paddlepaddle 3.3.1, **numpy 1.26.4**, Pillow 11.1.0, huggingface-hub 1.27.0, jsonschema 4.23.0 |
| Python 3.13 | `config/requirements-kaggle-ocr-py313.txt` / `config/ocr-kaggle-py313.json` | paddleocr 3.7.0, paddlex 3.7.0, paddlepaddle 3.3.1, **numpy 2.2.6**, Pillow 11.1.0, huggingface-hub 1.27.0, jsonschema 4.23.0 |

Profile 3.13 là **candidate, cần verified-by-Kaggle-run; chưa được local inference
xác nhận**, chưa có run Kaggle xác nhận trong task này. Không tuyên bố OCR accuracy
hay binary/ABI compatibility của toàn dependency graph. Baseline 3.10–3.12 giữ
nguyên, cũng không đảm bảo wheel cho mọi image. Ngoài 3.10–3.13: dừng rõ
`DEPENDENCY_UNAVAILABLE` kèm Python/executable/platform/profile; không bỏ guard.

Rationale và nguồn metadata (evidence PyPI 2026 do brief cung cấp; task offline
không truy cập mạng để tái kiểm chứng):
- [PaddlePaddle 3.3.1 metadata](https://pypi.org/pypi/paddlepaddle/3.3.1/json):
  wheel `cp313-cp313-manylinux1_x86_64`, `numpy>=1.21`, `protobuf>=3.20.2`,
  Pillow, `opt_einsum==3.3.0`… không tự nó chứng minh runtime import thành công.
- [PaddleOCR 3.7.0 metadata](https://pypi.org/pypi/paddleocr/3.7.0/json):
  py3 wheel, Python >=3.8, classifier Python 3.13,
  `paddlex>=3.7.0,<3.8.0`; pin [PaddleX 3.7.0](https://pypi.org/pypi/paddlex/3.7.0/json)
  chính xác để không tự trôi minor/patch.
- [NumPy 2.2.6 files](https://pypi.org/project/numpy/2.2.6/#files): chọn exact
  2.x có cp313 wheels thay cho 1.26.4 không có baseline cp313. Không dùng NumPy
  unbounded. Giữ Pillow 11.1.0 (cp313 wheels), jsonschema 4.23.0 và Hub 1.27.0
  từ profile cũ; các package py3/transitive vẫn phải qua resolver trên image thật.

Kaggle Python 3.13 có thể thiếu `ensurepip`. Mode **`venv-host-pip`** tạo venv bằng
`venv.EnvBuilder(with_pip=False, system_site_packages=False)`, không gọi ensurepip.
Host/kernel `sys.executable -I -B -m pip --python <venv_python>` điều khiển cả
resolver, install và check, kể cả check trong child preflight; không cần pip trong
venv, không cài global/kernel, không upgrade pip. Host pip thiếu `--python`,
`--dry-run` hoặc `--report`: fail-closed rõ `DEPENDENCY_UNAVAILABLE` trước model/output;
dùng image có pip hỗ trợ, không fallback sang ensurepip/global install.

Trước cài, host pip chạy `--python <venv_python> install --only-binary=:all:
--dry-run --report ...` kiểm tra toàn dependency graph. Install cũng binary-only:
**không build source, không tự bỏ pin**. Sau cài/cả lần rerun cached: host pip chạy
`--python <venv_python> check`,
kiểm tra exact versions và import smoke `numpy`, `paddle`, `paddleocr`, `paddlex`,
`PIL`, `jsonschema`, không khởi tạo/tải model. Thiếu wheel, dependency conflict
hoặc import/ABI lỗi: `DEPENDENCY_UNAVAILABLE`, trước model/output.

`interpreter.json` ghi fingerprint kernel (Python version/executable/implementation,
platform/machine/sysconfig platform), mode `venv-host-pip`, host/target Python,
host pip version, profile, requirements/config hashes; `venv-interpreter.json`
ghi version/base prefix/prefix/executable/user-site để kiểm tra venv kế thừa kernel
và cô lập system/user packages. Child environment bỏ Python/pip overrides từ kernel.
`resolver-report.json` giữ report resolver của pip. Success marker
`venv/requirements.sha256` chỉ ghi sau **toàn bộ** check pass, hash bao gồm
fingerprint mode/controller/Python/platform/profile/requirements/config; failed cached
recheck xóa marker. Marker/mode cũ bị từ chối; venv partial không có marker được
xóa và tạo mới trong đúng working folder, không reuse partial install.
Các file runtime diagnostics nằm trong working folder, không xuất
vào sidecar ZIP. Chưa lock toàn bộ transitive versions/wheel hashes; metadata
có thể thay đổi, resolver/import checks trên Kaggle là gate bắt buộc.

Không tải/cài dependency khi test local. Nếu fail, cung cấp wheelhouse đầy đủ
hoặc cập nhật exact pins có evidence trong package version mới; không unpin để ép chạy.

Hai model nhỏ PP-OCRv5 mobile det + latin rec có repo/full revision trong JSON,
giữ nguyên từ pilot. Latin model bao gồm Vietnamese, **không phải language
detector**; per-line `language/script=null` nếu API không trả. Đường dẫn cache
không xuất vào payload; hash từng inference file, revision và package versions
được lưu. Confidence chưa calibrated. Cần benchmark OCR accuracy riêng.

Internet off: attach dataset có folder **`ocr-wheelhouse`** chứa đủ wheels
(bao gồm transitive dependencies) phù hợp Python/Linux image. Notebook tự tìm,
dùng `pip --no-index --find-links`. Attach folder **`ocr-models`**:

```text
ocr-models/
  PP-OCRv5_mobile_det/
    inference.json
    inference.pdiparams
    inference.yml
    config.json        # nếu snapshot có
    model-lock.json
  latin_PP-OCRv5_mobile_rec/...
```

Mỗi `model-lock.json`: `{"repo":"PaddlePaddle/<name>","revision":"<full revision
từ config>","files_sha256":{"inference.json":"<sha256>",
"inference.pdiparams":"<sha256>","inference.yml":"<sha256>"}}`.
Include `config.json` trong hashes nếu dùng. Chuẩn bị từ **đúng pinned snapshot**
trên máy có Internet, không đổi revision bằng cách chỉ sửa lock. Không có
weights trong distribution ZIP. Model file hash được kiểm tra trước và sau run.
Notebook tự đặt `KIS_OCR_MODEL_ROOT` khi thấy đúng một `ocr-models`; offline
models thì không gọi Hub. Có thể dùng wheels online + models offline hoặc ngược
lại. Nếu không có cả Internet lẫn dataset cần thiết, dừng, không báo `empty`.

GPU là **advanced opt-in**, không bắt buộc để đáp ứng luồng chỉ sửa INPUT_ROOT.
Trong config source rồi regenerate notebook riêng, chọn `device="gpu:0"`,
`paddle_distribution="paddlepaddle-gpu"`, cung cấp wheel GPU đã kiểm chứng với
CUDA/cuDNN/Python của Kaggle. Notebook thay đúng dependency CPU bằng distribution
GPU với pin `versions.paddlepaddle`; không cài đồng thời hai loại. Repo không
có baseline GPU đã kiểm chứng nên **không đảm bảo có GPU wheel 3.3.1 cho image
của bạn**. Nếu phải đổi version, cập nhật config và requirements thành package
version mới, test lại. Không silent fallback GPU→CPU hoặc tự đoán CUDA wheel URL.

## Padding và tọa độ

Default 32 px trắng cả bốn phía; thay đổi có kiểm soát trong config JSON (advanced).
v1 chỉ hỗ trợ `resize=null`, `batch_size=1`, `checkpoint_every=1`; các giá trị khác
fail thay vì im lặng bỏ qua. Giữ nguyên chiều pixel khi convert RGB; không EXIF
auto-rotate (tọa độ theo stored pixels, không theo viewer tự xoay).

`PaddleOCR.predict` v3 trả `rec_polys` đã scale-back về ảnh **được truyền vào**.
Adapter theo API pilot, truyền ndarray BGR từ Pillow RGB. Paddle có thể resize
nội bộ cho detection/recognition; wrapper **không** ghi hoặc tuyên bố biết
transform tensor/model-space. Cần integration smoke riêng nếu đổi Paddle API.
Tắt doc orientation, unwarping, textline orientation. Wrapper chỉ kiểm soát border:

```text
x_padded = x_original + left; y_padded = y_original + top
x_original = x_padded - left; y_original = y_padded - top
```

Mỗi record xử lý ảnh lưu original/padded `[width,height]`, padding/color,
forward/inverse matrix, resize null. Record missing/unknown hoặc lỗi decode chưa
biết kích thước có `preprocessing=null` (không bịa transform). Polygon padded
phải nằm trong padded bounds, nếu không thì record error. `polygon_original`
và `bbox_original` là inverse **chính xác**, có thể âm/vượt ảnh nếu detection nằm
trong padding. Trường `*_original_clipped` riêng được clip trong `[0,w] × [0,h]`
để vẽ. Không ghi clipping như một invertible transform. Thứ tự line cố định theo
top, left, bottom, right, rồi canonical line JSON; `line_index` bắt đầu từ 0.

## Contract output và resume

```text
outputs/<run_id>/
  ocr/<group>/<video_id>.jsonl
  records/<group>/<video_id>/<n>-<sha256>.json
  manifest.json
  checkpoint.json
  coverage.json
  run_metadata.json
  README.md
```

Schema `schemas/ocr-sidecar-v1.schema.json` validate **mỗi JSONL record**:
top-level `schema="ocr-sidecar-v1"`, `schema_version=1`, canonical key
`(video_id,keyframe_n)`, image name/relative source/SHA256, optional mapping,
engine packages/model revisions/hashes/config hash, preprocessing, lines/errors.
Manifest có cùng schema/version, inventory đầy đủ, input/config/model/source
hash và artifact hashes. Validator kiểm tra schema **và** uniqueness, provenance,
inverse/bounds/line ordering, JSONL↔checkpoint, coverage và checksums.

| Status | Ý nghĩa |
|---|---|
| `ok` | OCR thành công và có line không rỗng |
| `empty` | OCR thành công, không có line; không chứng minh ảnh thật sự không có chữ |
| `missing` | Mapping khai báo ảnh chưa có, hoặc ảnh trong inventory biến mất |
| `error` | Decode/inference/geometry lỗi; error type/code riêng, không lưu message có absolute path |
| `unknown` | Chưa xử lý/interrupted sau checkpoint ban đầu, không phải âm tính |

Per-record là content-addressed immutable JSON, ghi atomic; checkpoint trỏ phiên
bản hiện hành. Crash giữa record write và checkpoint update chỉ để orphan không
được skip; resume đọc lại checkpoint cũ. Video JSONL/summary được ghi atomic từng
file cuối run, manifest ghi cuối cùng. Không claim transaction atomic toàn video
và tất cả reports cùng lúc; nếu crash lúc publish thì chạy resume để tạo lại
summary trước validate/package. ZIP chỉ include records đang được checkpoint
tham chiếu, không include orphan, ảnh nguồn, weights, credentials hoặc logs.

Resume đòi code/input/config/model/runtime hashes match. Ảnh thay byte, file mới,
mapping đổi, model/config/code đổi: phải run mới; ảnh biến mất giữ hash đã biết
và đổi missing. Chỉ skip records đã validate schema + checksum + provenance
và ảnh hiện tại hash-matching. Error chỉ retry khi thêm `--retry-errors`.
Unknown tiếp tục xử lý; missing có thể chỉ hồi phục nếu ảnh có identity/hash
đã biết từ inventory, không cho thêm ảnh chưa biết vào run cũ.
Không overwrite mặc định. Init dependency/model lỗi dừng trước tạo run.

CLI độc lập (không cần production module):

```bash
python tools/kaggle_ocr.py validate-input --input-root /path/to/keyframes
python tools/kaggle_ocr.py run --input-root /path/to/keyframes --output-root outputs/ocr-v1
python tools/kaggle_ocr.py run --input-root /path/to/keyframes --output-root outputs/ocr-v1 --resume --retry-errors
python tools/kaggle_ocr.py validate-output --input-root /path/to/keyframes --output-root outputs/ocr-v1
python tools/kaggle_ocr.py package --input-root outputs/ocr-v1 --output-root outputs/ocr-v1.zip
python tools/kaggle_ocr.py preflight  # isolated venv only: pip check + imports, no model
```

Source distribution v4 có thể tạo offline bằng `ocr.write_zip` với allowlist
`ocr.DIST_FILES`, thêm `tools/build_kaggle_ocr_notebook.py`, và README package;
`MANIFEST.sha256` chỉ dùng kiểm tra archive này, không dùng trong notebook bootstrap.

`--config` optional cho validate-input/preflight/run; default profile theo Python
thực tế, config tương đối package. Validate-input không import Paddle backend,
không phụ thuộc cwd. Không có nhiều biến cần đổi cho group/video. Lệnh
validate-output báo coverage incomplete nhưng trả thành công nếu contract đúng;
caller phải xem `complete` và `status_counts` để quyết định downstream.

## Future merge — không làm trong task này

Một bước explicit trong tương lai join `(video_id,keyframe_n,image_sha256)` với
caption_generator provenance. Hash mismatch/missing/null phải reject/quarantine,
không join bằng frame_idx hoặc chỉ lấy số ảnh làm frame_idx. Mapping/timestamp
là metadata tùy chọn, cần kiểm tra CSV gốc nếu dùng. Giữ caption source nguyên
trạng; không ghi OCR vào caption, không tạo `retrieval_text`. Sidecar không tự
rebuild FAISS/Milvus hay bật production OCR. Không có tuyên bố OCR accuracy
đã đạt benchmark hoặc model đã chạy trên máy local trong task đóng gói này.
