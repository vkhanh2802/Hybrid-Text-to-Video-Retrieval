# Kaggle OCR v5 — backend CPU oneDNN/PIR gate

## Cách dùng (chỉ một chỗ cần sửa)

1. Import `notebooks/kaggle_ocr_paddle_padding_v5.ipynb` vào Kaggle.
2. Attach dataset keyframes của bạn, bật Internet để tải wheels/models đã khóa.
3. Cell 1 chỉ có `INPUT_ROOT = "/kaggle/input/your-keyframes-dataset"`.
   Thay đường dẫn này bằng root chứa các video/group, group root, hoặc video root.
4. **Run All**. Nếu smoke và processing thành công, xem coverage rồi tải **cả ZIP
   và `.sha256`** qua hai FileLinks cuối notebook.

Không upload ZIP mã nguồn/manifest, không sửa cell khác, không chạy notebook v4
hay dùng lại output/checkpoint v4. CPU mặc định; chỉ bật GPU trong Kaggle không
chuyển profile này thành GPU.

Output riêng:
`/kaggle/working/kis-kaggle-ocr-v5/<payload_hash>/outputs/ocr-v1`.
Diagnostics ở thư mục ngang hàng `outputs/diagnostics-v5/`, không nằm trong ZIP
sidecar. Nếu smoke thất bại, **không có `ocr-v1`/checkpoint mới**; tải diagnostics
từ Kaggle Output nếu cần chẩn đoán. Không có hàng nghìn error records giả.

### Khi thấy `OCR_STOPPED`: đọc lỗi gốc, không chạy lại cả dataset

`RuntimeError: OCR_STOPPED: child exit ...` là **wrapper của notebook khi child
process trả mã khác 0**, không phải lỗi Paddle gốc và không có nghĩa OCR rỗng.
Run cell in riêng **child stdout và stderr** (redact secrets, tối đa 65,536 ký tự
mỗi stream, giữ đầu/cuối) **trước khi raise**, kèm đường dẫn tuyệt đối tới
`run-fatal.json`. Đọc `phase`, `type`, `message`, `traceback` trong file này; nếu
phase là `backend-smoke`, xem thêm từng candidate `api.json`, `mkldnn-flag.json`
và `failed.json` để thấy nguyên nhân trước wrapper.

Sau cell lỗi, có thể thêm và chạy **đúng cell chẩn đoán dưới đây**; không sửa
`INPUT_ROOT`, không chạy lại OCR, không dò thư mục payload cũ:

```python
import json
from pathlib import Path

diagnostics = Path(work) / "outputs/diagnostics-v5"
print("Current diagnostics:", diagnostics)
marker_path = diagnostics / "invocation.json"
if not marker_path.is_file():
    print("Chưa tạo được invocation marker; xem traceback của cell lỗi phía trên.")
else:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    print(json.dumps(marker, ensure_ascii=False, indent=2))
    for name in ("run-fatal.json", "preflight.json", "api.json", "mkldnn-flag.json",
                 "failed.json", "selected.json", "run-smoke.json"):
        path = diagnostics / name
        if not path.is_file():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("invocation", {}).get("id") != marker["id"]:
            print("SKIP stale diagnostic:", path)
            continue
        print("\n===", path, "===")
        print(ocr.safe_text(json.dumps(report, ensure_ascii=False, indent=2)))
```

Mỗi run/setup/CLI có marker giới hạn kích thước (`id`, command, PID, UTC), và chỉ
xóa reports **do v5 sở hữu** của invocation cũ trong thư mục diagnostics đó.
Child preflight/candidate dùng chung ID với controller, không xóa evidence hiện
tại. Validate-output/package CLI có thư mục riêng `diagnostics-v5/cli-validate-output`
và `diagnostics-v5/cli-package`, không xóa báo cáo của run vừa xong. Không xóa
output/checkpoint hay dữ liệu v4. Setup, input/config, resume và runtime-preflight
failure cũng có `run-fatal.json`; lỗi child preflight giữ stdout/stderr trong báo
cáo fatal. Diagnostics không nằm trong ZIP sidecar. Lưu diagnostics trước một
invocation mới nếu cần giữ lịch sử; review trước khi chia sẻ.

Trước mọi tạo/xóa diagnostics, v5 kiểm tra containment và toàn bộ ancestors của
input/output/diagnostics: không ghi trong `INPUT_ROOT`, không chấp nhận symlink,
junction/reparse point hay parent traversal. Nếu chính đường dẫn không an toàn,
dừng **trước mọi ghi/xóa** và xem exception ở notebook/CLI stderr; không cố ghi
`run-fatal.json` vào đường dẫn bị từ chối. Setup cell tạo invocation mới trước khi
kiểm tra wheelhouse/model cache hoặc `reject_links`; lỗi setup dùng phase
`notebook-setup`, giữ hai cờ ready/success false và không tạo `outputs/ocr-v1`.

Log `parse_common_args() missing ... keyword-only ... default_enable_hpi` là
**lỗi verifier API của v5 cũ**, chưa chứng minh inference oneDNN/PIR thất bại.
Bản này inspect signature, source hash và call-site của parser đã cài, chỉ truyền
`default_enable_hpi` chỉ khi call-site chứa trực tiếp literal `True`/`False` và
các call-site thống nhất. Bảo thủ từ chối tên biến/module attributes, computed
defaults, local shadowing, reassignments hoặc method calls; giá trị module hiện
tại không chứng minh lexical binding của call-site. Không đoán giá trị, không
thực thi biểu thức source tùy ý. Nếu không xác minh được, API mode vẫn dừng;
evidence giữ signature, parser/source hashes đã đọc được, args/kwargs và traceback
introspection gốc, kể cả khi constructor sau đó fail. `mkldnn-flag` là gate riêng:
API probe fail vẫn có thể chạy candidate này **chỉ khi** Paddle xác nhận flag được
hỗ trợ và effective false/0 trong child mới; phải vượt qua smoke ảnh thật.

Bước an toàn: import notebook v5 vừa regenerate vào **Kaggle session mới**, attach
keyframes, chỉ sửa `INPUT_ROOT`, Run All và đọc diagnostics hiện tại. Source hash
thay đổi nên không dùng lại checkpoint cũ. Không tự unpin, đổi model/flag, cài
wrapt toàn cục hay retry cả dataset để che lỗi verifier/backend.

## Nguyên nhân và căn cứ workaround

Traceback Kaggle v4 đã xác minh: Python 3.13.16, PaddleOCR/PaddleX 3.7.0,
PaddlePaddle 3.3.1 CPU. Text detection → static runner →
`onednn_instruction.cc:116`, `NotImplementedError: ConvertPirAttribute2RuntimeAttribute
not support [pir::ArrayAttribute<pir::DoubleAttribute>]`.
Models/checksum đã OK; warning `sitecustomize`/`wrapt` không phải kết luận inference.

**Không có Paddle stack/source exact-version trong môi trường local; không dùng
network hay cài OCR để suy đoán flag.** Do đó căn cứ hỗ trợ API/flag được kiểm tra
trên **chính target venv** trước khi cho chạy dataset, không được coi là đã xác
nhận inference trên Kaggle:

1. Subprocess mới, backend environment chuẩn bị **trước import paddle**. Không
   kế thừa `FLAGS_use_mkldnn`/`FLAGS_enable_pir_api` từ kernel. Trước tiên thử
   `enable_mkldnn=False` **chỉ khi** tham số được khai báo trong signature, hoặc
   installed `paddleocr._common_args.parse_common_args` thực sự nhận và giữ giá trị
    `False`, đồng thời AST của installed pipeline base có gọi parser này.
    Keyword-only `default_enable_hpi` được lấy từ semantics call-site đã xác minh,
    không dùng default đoán mò; evidence được lưu trước khi khởi tạo model.
   `**kwargs` tự nó không được coi là chứng cứ. Lưu signature hoặc SHA256 của
   source parser/pipeline base. Constructor reject thì candidate fail, không
   retry mù bằng cách bỏ option.
2. Nếu candidate đầu fail, chỉ được thử `FLAGS_use_mkldnn=0` khi
   `paddle.get_flags(["FLAGS_use_mkldnn"])` ở candidate trước xác nhận flag tồn tại.
   Flag phải đặt trong env của **subprocess mới** trước import. Kiểm tra lại
   effective value bằng Paddle API, kết hợp `enable_mkldnn=False` nếu API hỗ trợ,
    rồi bắt buộc inference smoke lại. API-option verification và flag gate độc lập;
    API không xác minh được không tự veto một flag đã được xác minh effective false.
    Env var có mặt không phải bằng chứng flag có
   tác dụng.
3. **Không thử `FLAGS_enable_pir_api=0`**: chưa xác minh tương thích với model/PIR
   stack này. Chỉ log hỗ trợ/effective value nếu Paddle API cho đọc được.
4. Không candidate nào inference thành công → fail rõ, không xử lý dataset.

Smoke dùng ảnh hiện hữu đầu tiên theo `(video_id,keyframe_n)` đã sort, đúng model
revisions/files, PaddleOCR adapter BGR, padding và normalization của full run.
Ảnh đầu corrupt thì dừng, không âm thầm đổi sang synthetic image. Sau child smoke,
run process kiểm tra cùng engine/backend identity và smoke adapter thực tế một
lần nữa **trước khi tạo output/checkpoint**. Backend inference/adapter failure ở
bất cứ ảnh nào sau đó đều fatal; không chuyển thành per-frame error rồi tiếp tục.
Ảnh corrupt/invalid decode sau smoke vẫn ghi `error` và tiếp tục các ảnh khác.
Backend fatal giữa run giữ record đang làm ở `unknown`, `complete=false`, xóa
coverage cũ; không xuất ZIP hợp lệ từ run bị ngắt.

## Dependencies, models, isolation

Pins giữ nguyên v4: PaddleOCR 3.7.0, PaddleX 3.7.0, PaddlePaddle 3.3.1,
Pillow 11.1.0, huggingface-hub 1.27.0, jsonschema 4.23.0.
Python 3.13 dùng NumPy **2.2.6**, Python 3.10–3.12 dùng **1.26.4**.
`config/ocr-kaggle-py313-v5.json` giữ nguyên nội dung/pins/model revisions của
profile 3.13, nhưng có tên version riêng. Không unpin NumPy.

`venv.EnvBuilder(with_pip=False, system_site_packages=False)`; host
`python -I -B -m pip --python <target>` dùng binary-only exact requirements,
resolver dry-run/report, pip check và child import preflight. Không ensurepip,
không `-S` làm mất site-packages, không global/kernel install, không pip upgrade,
không source build. Cache identity gồm interpreter/controller/pins/config/source.

Notebook chứa tám file allowlisted với hash từng file và payload hash; có cả
`tools/kaggle_ocr.py` **nguyên trạng** để tái dùng primitives sidecar/schema/ZIP.
V5 load đúng sibling file đã embedded/hash-check, không tìm package v3/v4 bên ngoài,
không import production/caption. Builder v5 tái dùng bootstrap helpers nguyên tắc
từ builder cũ; builder cũ không bị sửa và không cần có trên Kaggle.
Không embed credentials, images, weights hay đường dẫn máy người phát triển.

Models tự tải theo full Hugging Face revision:

- `PaddlePaddle/PP-OCRv5_mobile_det` — `0d63e78e2b680928f6b1747d76a08db6e645efb7`.
- `PaddlePaddle/latin_PP-OCRv5_mobile_rec` — `ab2cd5cc5fa6309be2e5acdfe66eca2c2c127d57`.

Hash các file `inference.json`, `inference.pdiparams`, `inference.yml` và optional
`config.json`, recheck trước/sau inference. Offline nâng cao: attach một
`ocr-wheelhouse` đầy đủ và/hoặc `ocr-models/<model-name>/model-lock.json`, lock có
`repo`, `revision`, `files_sha256` bao phủ đúng toàn bộ inference/config files.
Không cần các cache này ở luồng keyframes-only có Internet.

Warning **`WRAPT_SITECUSTOMIZE_WARNING`** giải thích rằng startup hook có thể nhắc
thiếu `wrapt`; warning đơn lẻ không chứng minh thành công/thất bại. V5 không tự cài
wrapt toàn cục hay thêm pin đoán mò. Transitive metadata/pip resolver/pip check
phải giải quyết dependencies **trong target venv**; nếu stack còn lỗi thì dừng.

## Contract và diagnostics

- Sidecar v1, trắng 32 px mỗi cạnh, không wrapper resize; inverse translation
  nguyên vẹn và coordinates clipped riêng. Key `(video_id,keyframe_n,image_sha256)`.
- `ok`, `empty`, `missing`, `error`, `unknown` phân biệt; `complete=true` chỉ khi
  không có error/missing/unknown. Không đánh đồng failure với empty OCR.
- Checkpoint mỗi ảnh, immutable hashed records; resume đối chiếu input/config/
  model/source **và backend hash**. Run All lặp lại không tự retry error.
- Full exception phase/type/message/traceback ở diagnostics, redaction các env
  secrets phổ biến/URL credentials, giới hạn traceback 65,536 ký tự (head/tail có
  marker nếu truncate), message 4,096 ký tự. Không bỏ traceback thành mỗi tên lỗi.
  Review diagnostics trước khi chia sẻ; file local traceback có thể chứa paths.
- ZIP sidecar deterministic, explicit allowlist, manifest SHA256, chống traversal/
  symlink/case collision. `backend-v5.json` được hash trong run manifest và ZIP.
  Diagnostics/stdout/stderr không export trong ZIP sidecar. Không sửa dataset,
  captions, FAISS, Milvus hay production.

## Kiểm chứng và giới hạn

Chỉ offline/static/mock. **Chưa chạy actual Kaggle install, network hay OCR; chưa
khẳng định workaround inference thành công trên Kaggle.** Nếu target API không
khớp hoặc oneDNN/PIR vẫn lỗi sau các candidate an toàn, v5 dừng có diagnostics.

```powershell
& 'C:\Users\Khanh\miniconda3\envs\focusguard_ai\python.exe' -B -m unittest tests.test_kaggle_ocr_v5 tests.test_kaggle_ocr tests.test_caption_artifacts
& 'C:\Users\Khanh\miniconda3\envs\focusguard_ai\python.exe' -B tools/build_kaggle_ocr_notebook_v5.py --check
git diff --check
```

Notebook JSON/cell compile/payload source/hash, bootstrap chỉ từ notebook,
profile/pins, verified API/fallback order, env-before-import, smoke fail-before-
output, corrupt-image continuation, fatal backend errors, traceback redaction/
bounds, checkpoint/hash mismatch và deterministic ZIP/path safety có offline tests.
SHA notebook/payload/distribution ZIP được báo riêng sau khi build để tránh vòng
lặp tự-hash trong notebook hoặc docs.
