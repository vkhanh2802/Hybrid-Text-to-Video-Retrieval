# OCR bổ trợ trên CPU trường

## Active mới nhất: đủ dữ liệu L01–L21 (18/11/2026)

- Đã nhập OCR L20 (8.192 record), OCR L21 (7.800 record) và bộ artifact L21
  mới: 7.800 ảnh / 29 video, mapping, visual feature, object, caption và index.
- Worker active `103338`, cổng CPU `8021`, tunnel laptop `18011`.
  `config/school-runtime.local.json` đã chuyển atomically và handshake thành công.
- Dataset `~/kis-l21-data-20261118T051746Z/dataset`; L01–L20 dùng
  hardlink regular-file từ dataset cũ, không ghi đè hay sửa file chia sẻ tại chỗ.
- Caption nguồn: **161.742**; caption và visual retrieval: **161.722 vector**.
  OCR: **161.742 record**, **161.722** binding không mơ hồ, **161.677** record
  có text để tìm kiếm. Không còn nhóm thiếu ảnh, caption hoặc OCR trong L01–L21.
- 20 ảnh/caption/OCR L21 thuộc 10 identity `frame_idx=0` mơ hồ vẫn được giữ
  nguyên; cả hai thành viên mỗi identity bị loại khỏi retrieval. Không suy ra
  frame ID từ timestamp. Bản `172(1).jpg` giống ảnh gốc được bỏ qua khi nhập.
- Snapshot `~/kis-ocr-import-20261118T052311Z/snapshot`, pin
  `3ae8377f8362424ce5cda4baabc7b3c831d2c240402c65fa0f6db04c5887f949`.
  JSON nguồn giữ nguyên bytes; OCR L01–L19 có hash như lần nhập trước.
- Đã decode đủ 7.800 ảnh L21; kiểm caption-image hash, manifest, ảnh/OCR mẫu
  qua worker và cả 20 ảnh bị quarantine. Feature L21 mẫu khớp encoder CPU với
  cosine **0,999865**. Hai query smoke L20/L21 có OCR hit, top100 distinct.
  Đây là kiểm tra tích hợp, không phải benchmark độ chính xác.
- 80 test liên quan pass. Giữ nguyên model, retrieval, OCR weight **0,1** và
  Qwen **batch12/top20**. Không chạy Qwen/DRES, không restart/hot-reload web.
  Web đang chạy cần được nạp lại cấu hình riêng để dùng worker mới.
- Worker trước `101754/8020` và dataset cũ vẫn giữ nguyên. Backup private:
  `config/school-runtime.before-complete-l01-l21.local.json`.
- Biên nhận: `benchmark_results/school_l01_l21_deploy_20261118_run01/FINAL_VERIFICATION.json`.
  Các mục dưới đây là lịch sử, không mô tả coverage active hiện tại.

## Lịch sử scope: L01–L21 (thêm L04/L05/L06)

Active worker101754/8020qua18010 dùng đủ21nhómretrieval; OCRL01–L19 đãnạp đủ
145750record. L04/L05/L06 **khôngcònngoàiscope**. Caption153942vector/L01–L20,
visual161722vector/L01–L21. OCRL20/L21unknown; ảnh/captionL21trênCPU cònthiếu.
Cácbảngscope18nhóm phía dưới làlịchsử trước lầnmởrộngnày.
Kiểm đầyđủmapping/features/caption/index/images22946frame của3nhóm mới và
3queryCPUsmoke đạt trước cutover. Khôngrestartweb/Qwen/DRES. Xem
`benchmark_results/school_full_scope_20261117_run01/FINAL_VERIFICATION.json`.

## Active mới nhất: OCR L01–L19 nhập ngày17/11/2026

- Snapshot nhập đủ19nhóm/145750record, root
  `~/kis-ocr-import-20261117T163125Z/snapshot`, pin
  `15405f1a222bc6d37a5e853b0576e84c3b6b246dc95c4d566ed9264907a3ef5d`.
- Activeworker101286/8019 qua18009; config `config/school-runtime.local.json`
  đãchuyển/handshake, bảnstaging `config/school-ocr-l01-l19.local.json`.
- Giữscope18nhóm, nạpOCR16nhóm/122804record:
  L01/L02/L03/L07/L08/L09/L10/L11/L12/L13/L14/L15/L16/L17/L18/L19.
- L04/L05/L06 lưu trênCPU nhưngngoàiscope; L20/L21missingunknown.
- L06–L10ZIPngoài chứa mainZIP vàdiagnostics. ChỉmainOCRđượcnhập, bytesJSON
  khôngđổi; originalwrapper/diagnostics giữ nguyênlaptop, khôngđưavàosearch.
-32evidencesample khớpZIP, CPU FTSsmoke2hit/top100distinctpass. KhôngQwen,
  khôngrestartweb/DRES; model/scope/OCRweight0.1/Qwenbatch12giữnguyên.
- L18ảnhđãnạp, workerstartupchỉ cònL21trongmissing_image_groups.
- Receipts: `benchmark_results/ocr_l01_l19_deploy_20261117_run01/FINAL_VERIFICATION.json`.
  Cácmụcactive phía dưới làlịch sử trước lầncậpnhậtnày.

## Active mới: L01/L02/L03, snapshot nhập L01–L05 (16/11/2026)

- Đã nhập5ZIP mới/hiện có trong `dataset/ocr`:36992record; L01/L02hashkhôngđổi.
- Active worker96666/8018 qua18008, source OCRrouting/parallel/selective mới.
- Snapshot `~/kis-ocr-import-20261116T123527Z/snapshot`, pin
  `7fddd5bf7a46262a8d7a607e1daf2c0daa23a5a958143ee62887e2ac7230b6de`.
- Scope retrieval18nhómgiữ nguyên; activeOCR **L01/L02/L03,22074record**.
  L04/L05lưu trênCPU nhưngngoài scope tìmkiếm. Cácnhóm còn lạiOCRunknown.
- Privateactive `config/school-runtime.local.json`, bảnstaging
  `config/school-ocr-selective.local.json`. Current-onlyconfig trướcđó giữ ở
  `config/school-runtime.before-l01-l05.local.json` (không phảiOCRlegacyđãxóa).
- Cấu hìnhđãatomiccutover/handshake; khôngrestartweb/DRES. Chưahotreloadweb.
- Báocáo `benchmark_results/user_five_selective_ocr_20261116_run01/REPORT.md`,
  đánhgiá `ASSESSMENT.md`. OCRretrieval cólợi ởq4/q5; OCRpromptselective chưa
  tăngmetric vàvẫnoutlier60–82s. Xemđầyđủlimitations trướcso với lượtbenchmarkcũ.
- Cácmục current-only L01/L02 phía dưới làlịchsửtrước lầnnhậpnày.

## Hiện tại: chỉ giữ OCR mới L01/L02 sau xác nhận xóa toàn bộ OCR cũ

Người dùng đã chọn **“Toàn bộ OCR cũ”**, hiểu rằng L03–L09 mất OCR và không còn
snapshot rollback. Cấu hình active hiện là worker94039/8017 qua local18007:

- Root: `~/kis-ocr-l01-l02-20261115T172101Z/updated-groups-snapshot`.
- Manifest SHA-256: `0fdb5b786abe23f23a843cd352e7497aba9f076e83d89cd0c5464dad0acc81c6`.
- Chỉ L01/L02, **14.955records**. Scope retrieval/caption vẫn18groups, không thu
  corpus theo OCR. Các nhóm khác missing/unknown, không phải negative evidence.
- Active private settings: `config/school-runtime.local.json`; current-only copy:
  `config/school-ocr-current-only.local.json`. Private OCR rollback/staging cũ
  đã xóa; baseline OCR-off config vẫn giữ cho các ablation sắp tới.
- Worker OCR cũ90716 và92720 đã dừng trước cleanup để không giữ RAM FTS cũ;
  baseline89224/OCRoff không bị thay đổi. Web không khởi động/restart, không DRES.
- Laptop `dataset/ocr` chỉ còn ZIP L01/L02 mới, giữ nguyên hash. Archive scalar
  và diagnostics scalar cũ đã xóa. Notebook/code/model/ảnh/báo cáo benchmark giữ.
- Cleanup trên CPU dùng exact allowlist/hash gate, không xóa broad data roots.
  Archive chuyển dữ liệu hỗn hợp được lọc riêng `dataset/ocr/*` và kiểm hash mọi
  payload caption/mapping/feature/config còn lại trước khi thay archive.

Biên nhận/tiến độ dọn: `benchmark_results/school_ocr_cleanup_20261116_run01/`.
Cleanup đã hoàn tất và kiểm chứng:6root/98.083file OCR cũ đã xóa; archive hỗn hợp
loại15.087OCR members, giữ nguyên hash334.013payload khác.14.955rawJSON mới và
2ZIP mới giữ hash; worker current-only ready. Xem `FINAL_VERIFICATION.json`.
**Các kết quả ablation8query bên dưới là lịch sử trước cleanup**, khi OCR các nhóm
L03/L07/L08/L09 vẫn còn; không dùng chúng làm baseline mặc định cho trạng thái mới.
Lượt test tiếp theo chờ query/ground truth người dùng, chưa chạy query mới.

## Cập nhật L01/L02 mới và ablation ngày16/11/2026

Hai ZIP mới đã nạp vào snapshot riêng, thay group L01/L02 mà không sửa nguồn cũ:

- L01:7.339records (7.338ok,1empty); L02:7.616records (đềuok).
- Root: `~/kis-ocr-l01-l02-20261115T172101Z/snapshot`.
- Manifest SHA-256: `4c2483e2c93cd42b7aa9f056e94bdc68dd2cdb1087ced82da157d8012d607f8c`.
- L03–L09 giữ nguyên bundle bytes; snapshot đủ67.882records. Production scope
  vẫn18groups, OCR44.936records thuộcL01/L02/L03/L07/L08/L09. Không thu scope L01/L02.
- New worker92720/8016 qua local18006; config active
  `config/school-runtime.local.json` đã chọn worker/pin mới. Rollback private config:
  `config/school-runtime.before-l01-l02.local.json`. Không in/commit token.
- Source/config/models CPU giống worker OCR trước; không cài hoặc chạy OCR engine.
  Old snapshot và workers89224/8014,90716/8015 vẫn nguyên vẹn.
- Web cổng8000 không phản hồi lúc bắt đầu; không khởi động/restart web trong cập
  nhật này. Config mới áp dụng cho client mới/lần mở web tiếp theo. Không DRES.

Audit/cutover: `benchmark_results/school_ocr_l01_l02_update_20261116_run01/`.
8query đã đóng băng trước retrieval,4mỗi nhóm, gồm literal/visual control/ca OCR
lỗi đã biết. CPU retrieval đã hoàn tất; A=OCRoff, B=new OCR retrieval/no OCR Qwen,
C=same B pool/order/images + OCR Qwen. Đã hoàn tất24lượt, Qwen giữbatch12/top20,
420inputs, không OOM, mỗi output100distinct mapped frame IDs.7/140invalid scores
mỗi arm giữ unknown. Ảnh L18/L21 thiếu vẫn báo missing, không backfill.

| Chỉ số trên8known anchors | A | B | C |
|---|---:|---:|---:|
| Known Hit@1 | 2/8 | 3/8 | 4/8 |
| Known Hit@100 | 6/8 | 6/8 | 6/8 |
| Known MRR@100 | 0,3932 | 0,4970 | 0,5595 |
| Median Qwen inference | 14,55s | 14,42s | 17,83s |

Assistant partial-arm-blind top1 review:2/3/4full matches,2/1/1missing/unknown.
Không human ground truth hoặc heldout, không coi frame chưa gán nhãn là negative.
Retrieval có tín hiệu tăng hạng; prompt C có một cải thiện top1 nhưng9/15score
changes B→C xảy ra trên ứng viên OCR missing, nên chưa tách được lợi ích nội dung
khỏi cấu trúc prompt. C có outliers49,74/43,26s. Formol có OCR hit đúng nhưng
anchor vẫn rank299, ngoài100/top20. Không tự chỉnh weight/prompt sau thử nghiệm.

Kết quả đầy đủ:
`benchmark_results/school_ocr_l01_l02_ablation_20261116_run01/REPORT.md`.
Đã kiểm tra lại sau lần tiếp tục phiên: new worker vẫnready, đúngpin44.936records;
không chạy lại benchmark hoặc restart web. Không dùng pilot cũ bên dưới làm kết
luận cho dữ liệu mới.

> **Đánh giá bổ sung:** pilot6query đã hoàn tất tại
> [SCHOOL_OCR_ACCURACY.md](SCHOOL_OCR_ACCURACY.md). Chưa chứng minh tăng độ chính
> xác: exact OCR retrieval không giúp trong bộ thử, known-target metrics không
> tăng. Có1top1 phù hợp hơn khi AI xem ảnh nhưng chưa tách được lợi ích nội dung
> OCR khỏi thay đổi prompt; hai lượt Qwen có OCR chậm49,99/70,35s. Không mặc định
> OCR bật đại trà có lợi hoặc giữ nguyên tốc độ; cấu hình production chưa tự sửa.

## Lịch sử triển khai trước cập nhật L01/L02

Đã chuyển bảy ZIP L01/L03/L05–L09, kiểm tra SHA-256 trên trường, import vào
**root mới** `~/kis-ocr-import-20261114T212719Z`. Không sửa ZIP laptop,
không ghi đè dataset trường, không cài/chạy OCR engine. L02/L04 được sao chép từ
dataset trường cũ; JSON giữ nguyên bytes và tên canonical.

- OCR L01–L09: **67.882 record**, trong đó **9 empty**, không thiếu mapping.
- Phạm vi production vẫn L01–L03, L07–L21: **44.936 record OCR** thuộc
  L01/L02/L03/L07/L08/L09. L04–L06 giữ trong snapshot nhưng ngoài scope hiện tại.
- L10–L21 chưa có OCR. Không thu hẹp scope để làm đẹp coverage; `missing` không
  phải negative evidence. Ảnh L18/L21 vẫn thiếu, trả 404 như trước.
- Snapshot: `snapshot/MANIFEST.json` và mỗi group một bundle JSON. SHA-256 manifest:
  `7d25f15444b34ab8f60bb8d61efea3fe329c68f6f4875c62cd5722f008ebe56a`.
- Mặc định global OCR vẫn **off**. Sau xác nhận của người dùng, private school
  runtime đã chuyển sang worker OCR90716/8015 qua SSH18005 và web đã restart.
  Qwen đã warm và chạy query public thật; không truy cập/nộp đề DRES.

## Nhánh dữ liệu

```text
Text đề đầy đủ + mô tả bổ sung
  ├─ planner → visual / caption BM25 / caption semantic → text reranker
  └─ literal OCR nguyên gốc → CPU FTS5 trong RAM → bonus ứng viên có giới hạn
                             │
                             └─ OCR riêng theo (video_id,n) → frame_idx từ CSV

Video đề → CPU image encoder → video retrieval
                     │
text pipeline đầy đủ ┴→ fusion 70% text / 30% video (không đổi)
                         → chọn top20 gốc → lấy OCR cho ứng viên sau fusion
                         → ảnh + caption gốc + OCR_JSON riêng
                         → Qwen GPU laptop microbatch12, một pass
                         → trả đúng100 frame thật, distinct
```

Text-only cũng nhận OCR riêng trước Qwen. Với video không có text, pipeline vẫn
appearance-only, không tự tạo text query hoặc thêm một Qwen pass khác. Ứng viên
thiếu ảnh trong top20 không được bù từ rank21+. Qwen chỉ chia batch nếu CUDA OOM,
và phải báo `attempts/effective_batches/oom_splits` thật.

## Vai trò OCR và trọng số

1. **Literal retrieval**: chỉ `ocr_text` người dùng nhập hoặc đoạn chữ nguyên
   văn trong raw query được router OCR hiện có chấp nhận. Giữ dấu tiếng Việt,
   chữ số, phủ định, ranh giới từ; không tìm bằng bản dịch/caption sinh ra.
2. **Separate evidence**: gắn cho từng ảnh, độc lập việc có literal hit hay không.
   Không nối vào `caption`/`retrieval_text`, không rebuild caption vectors/index.
3. **Positive auxiliary**: snapshot dùng bonus tối đa `weight × max(base_score)`
    nhân `61/(60+rank_OCR)`. School runtime chọn **weight=0,1**, tức bonus tối đa10% điểm
   base cao nhất trong pool. Điểm của ứng viên không có OCR hit không bị giảm.
   Cách này không lấy bonus0,1 cộng trực tiếp vào RRF thường chỉ khoảng0,016.
4. Đây là **engineering weight chưa calibration**, không phải xác suất/độ tin
   cậy OCR hay kết luận accuracy tăng. OCR-only hit có thể chưa vào top100 nếu
   không có hỗ trợ từ các stream khác; chủ đích ban đầu là bổ trợ thận trọng.
5. Temporal OCR chỉ là seed khớp chữ cho event tương ứng; không chứng minh hành
   động/thứ tự và không mở hoặc hạ evidence gate hiện tại.

`strict` và `public_json` cũ giữ validation và fusion cũ. Bonus positive chỉ dùng
reader snapshot khai báo `bounded_positive_auxiliary`; OCR off giữ đường cũ.

## Provenance và an toàn prompt

Nguồn public chỉ có bảy field, không engine/confidence/boxes/hash ảnh. Vì vậy:

- `verified=false`, engine/confidence/image_sha256 đều unknown/null.
- SHA của ZIP/JSON/bundle chứng minh tính toàn vẹn **dữ liệu OCR**, không chứng
  minh OCR đã chạy trên đúng pixels ảnh. Không đổi thành provenance verified.
- Null `frame_id` phải join `(video_id,keyframe_n)` với CSV; không dùng `n-1`,
  không coi n là frame_idx. Duplicate hoặc mapping mơ hồ bị từ chối.
- OCR_JSON là JSON có giới hạn2048 ký tự, escape `<`/`>`, là untrusted data,
  không phải instruction. Ticker/phụ đề/timestamp không chứng minh hành động.
- `empty/error/missing` là unknown; không suy không có chữ trong ảnh, không trừ
  điểm retrieval. Public OCR không hỗ trợ confidence/region filters tùy chỉnh.

## Tính bất biến và hiệu năng

Importer giữ archives, JSON nguyên bản, CSV nguyên bản; manifest ghi SHA từng
JSON nguồn, SHA bundle, SHA CSV và số lượng/status. Runtime chỉ tiêu thụ bundle
pinned + CSV authoritative của worker, không mở lại44.936 JSON mỗi Search.
FTS5 chỉ nằm trong RAM, không cài thêm package hoặc ghi SQLite lên nguồn dữ liệu.

Mỗi lookup/search hash lại assets tiêu thụ, không chỉ dựa size/mtime. Thay nội
dung nhưng giữ size và khôi phục mtime vẫn bị phát hiện. Sửa JSON nguyên bản
**sau khi build** không làm snapshot cũ tự đổi; muốn dùng dữ liệu mới phải build
snapshot mới/pin mới/restart. Không hot-reload âm thầm.

Nếu OCR lỗi, text pipeline bỏ OCR boost/union và tiếp tục bằng baseline với
evidence error. Lỗi network worker/tunnel vẫn là lỗi Search, **không fallback
Drive/local**. Endpoint `/ocr-evidence` authenticated chỉ nhận tối đa100 identity,
trả evidence đúng thứ tự và build pin; không nhận caption/query/DRES session.

## Config opt-in và cutover

Config ready private ở `config/school-ocr-ready.local.json`; sau cutover được
chọn trong `config/school-runtime.local.json`, giữ URL/token riêng. Config trước
OCR được giữ nguyên tại `config/school-runtime.ocr-before.local.json`.
Không commit/copy token vào báo cáo. `start_school.ps1 -WorkerConfig ...`
đọc `ocr_format/mode/root/manifest_sha256/weight` và batch12 từ file này.

Handshake thêm pin manifest vào runtime signature cho `school_snapshot`; path
server không phải path laptop nên không so root string xuyên hai OS. Scope,
model revisions và các config khác vẫn phải khớp đầy đủ như trước.

**Restart cần xác nhận** vì mất phiên DRES trong RAM. Người dùng đã đồng ý lần
cutover này. Giữ
worker cũ, tunnel cũ, config cũ và mọi snapshot để rollback bằng restart có chủ
đích. Không tự tìm task, tự Search theo timer hoặc tự submit DRES.

## Bằng chứng và giới hạn kiểm thử

- Transfer/import: `benchmark_results/school_ocr_import_20261115_run01/`.
- Ready worker: `benchmark_results/school_ocr_worker_20261115_run02/`, PID90716,
  port8015 (SSH local18005); initial staging worker90307 retired, old live
  baseline worker89224 untouched.
- Unit/integration kiểm tra schema, mapping, duplicate, partial coverage,
  literal/diacritics, mutation cùng size/mtime, caption không đổi, OCR per-row
  vào batch, post-fusion attach trước một Qwen pass, 70/30 và budget20.
- CPU smoke dùng query public; `HTV9` được chọn vì thấy nguyên văn trong nguồn,
  **không phải benchmark relevance độc lập**. Lần đầu dùng `VTV` trả no_match
  hợp lệ; assertion yêu cầu positive hit của script fail, không phải OCR crash.
- Chưa có ground truth để khẳng định OCR tăng chất lượng. Qwen đã được kiểm tra
  bằng query public sau restart; không nạp bản Qwen thứ hai để tranh VRAM.

### Kết quả CPU final build

`benchmark_results/school_ocr_cpu_check_20261115_run03/`: ba lượt literal HTV9
đều50hit, pool295→345. Search + gắn OCR **0,393 / 0,391 / 0,446s** (median0,393s).
Endpoint evidence riêng100identity **0,197s RPC**,24ok+76missing, không đổi caption
hoặc điểm. Đây không phải chi phí Qwen hoặc tổng pipeline. Source audit đã đọc
lại67.882 JSON và toàn bộ ZIP/bundle, hash/inventory đều khớp; L02/L04 gốc còn
nguyên hash. Không có OCR engine mới hoặc inferred provenance.

Người dùng đã **cho phép restart web để thử OCR với Qwen** sau khi được thông
báo mất phiên DRES RAM. Chỉ dùng query công khai để kiểm tra; không lấy/nộp đề.

### Kiểm tra live Qwen và giới hạn tốc độ

`benchmark_results/school_ocr_live_enabled_20261115_run01/`: HTTP200,100 frame
distinct mỗi lượt, OCR riêng của từng ảnh, không OOM và không bù ảnh ngoài top20.
Qwen được nạp lại trong web mới, lượt cold65,83s; không dùng làm warm latency.

| Query public | OCR CPU | Qwen ảnh đọc được / batch thực tế | Warm tổng |
| --- | ---: | --- | ---: |
| Chữ HTV9 | 0,650s | 5ảnh /5 (15ảnh top20 thiếu) | 12,00s |
| Đường phố / giao thông | 0,566s | 18ảnh /12+6 (2ảnh thiếu) | 46,29s |

Query đường phố có24,46s tại fusion/object-evidence, không phải operator OCR;
nguyên nhân chi tiết chưa được profile. Không tự sửa object retrieval ngoài
phạm vi yêu cầu OCR. Baseline trước cutover (OCRoff) hai query lần lượt20,27s
và23,69s nhưng số ảnh/model inputs/planner/pool khác, nên **không phải A/B kiểm
soát**, không gọi OCR nhanh hơn8s hay chậm hơn23s dựa vào bảng này.

Một lần lặp public query đường phố sau đó **25,50s**, OCR **0,326s**, fusion
**2,96s**, VLM **18,69s**,17ảnh chạy12+5, không OOM. Xác nhận độ trễ fusion biến
động; không có cơ sở gán toàn bộ lượt46,29s cho OCR. Vẫn không phải median/P95.
Status cuối: webready, workerready, OCRschool_cpu44.936record, Qwen giữ nóng.

Qwen có output parse không hợp lệ: warm OCR HTV9 **2/5**, giao thông **1/18**;
baseline trước OCR cũng **2/8** và **2/17**. Giữ null/invalid, không bịa điểm hoặc
thử lại scalar để che lỗi. Đây là hạn chế có thật, chưa khẳng định OCR làm tốt
hơn hoặc xấu hơn; cần frozen-pool A/B có ground truth để kết luận chất lượng.

Full suite cuối: **968 tests — 952 pass,16 skip**,158,583s. Post-fusion OCR→Qwen
được kiểm tra integration/unit, endpoint thật đã kiểm tra trên CPU. Chưa chạy
một DRES video/task thật hoặc xác nhận nộp bài thành công trong lần OCR này.

Web **http://127.0.0.1:8000/** dùng OCR mới, Qwen giữ nóng, cần đăng nhập DRES lại
sau restart. Old worker/config được giữ cho rollback có chủ đích.

Code: `kis/school_ocr.py`, `tools/import_school_ocr.py`,
`tools/transfer_school_ocr.py`, `tools/benchmark_school_ocr.py`.
