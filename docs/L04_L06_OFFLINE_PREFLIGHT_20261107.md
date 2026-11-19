# Audit/preflight offline L04-L06 — 2026-11-07

## Phạm vi và giới hạn

Đã kiểm tra **92 video artifacts**: L04 V001–V030, L05 V001–V031,
L06 V001–V031. Đây là inventory/read-only preflight, **không phải nghiệm thu
retrieval hoặc cho phép bật production scope**. Không sửa `config/pipeline.json`,
không rebuild Milvus/visual index, không ghi đè dataset, không nạp model,
không gọi network/Drive API và không giải nén ZIP.

Luồng `python -B -m tools.temporal_preflight` không có tùy chọn mới giữ nguyên
scope, schema và hành vi cũ. Khi dùng `--groups`, `--image-root` hoặc `--archive`,
tool chuyển sang `kis_offline_preflight_v1`, tạo report mới bằng exclusive create
và từ chối output nằm trong dataset/image input roots.

Chế độ inventory tái dùng kiểm tra mapping/caption/features/index manifest của
P0, kiểm tra object JSON theo keyframe number và báo image coverage riêng:

- Mapping, caption/caption-mapping identity và row order được đối chiếu.
- Features được mmap read-only và kiểm tra count/dimension/finite/non-zero theo
  block; không chạy encoder. Đây không phải xác minh provenance của embeddings.
- Index file được hash, manifest được đối chiếu count/model/revision/hashes theo
  config hiện tại; **không import FAISS**, index content luôn `unknown`.
- Object JSON được parse là object và đối chiếu độ phủ keyframe; không xác minh
  độ đúng semantic của detection.
- Ảnh local chỉ kiểm tra filesystem metadata/nonzero size, **không decode, hash
  pixel hay chứng minh khả năng đọc ảnh ở runtime**. Cloud placeholder có cờ
  offline/recall được ghi `unknown`, không hydrate. Metadata của Drive có thể
  không thể hiện đầy đủ placeholder; không được hiểu `present` là ảnh đã đọc được.
- ZIP chỉ đọc central directory, không mở payload, verify CRC hay extract.
  `listed` không phải `ready`; canonicalize member path theo POSIX (bỏ `/./`),
  phát hiện duplicate destination sau normalize và case-fold bảo thủ trên mọi
  host để bao phủ Windows, kể cả collision giữa file và directory. `duplicate_members`
  chứa destination canonical đã case-fold, không còn là raw member name.
  Path traversal, absolute/drive paths, backslash, symlink, component rỗng (`//`),
  ký tự/tên không hợp lệ trên Windows (gồm trailing dot/space và reserved names)
  bị reject; dấu `/` cuối directory được phép. `//` vẫn được normalize để phát
  hiện collision, nhưng member đó cũng nằm trong `rejected_members` (raw name).
  Archive `unsafe` trả image coverage rỗng và `image_entries=0`, kể cả khi có
  member ảnh hợp lệ; không đóng góp archive image coverage hay runtime readiness.
- Exit `0` ở chế độ inventory nghĩa là đã tạo report, **không phải runtime-ready**.
  Input/output lỗi hoặc output tồn tại trả exit `2`. Nhóm thiếu mapping được báo
  riêng; video discovery vẫn mapping-based, không thể phát hiện cả video vắng
  khỏi mọi mapping nếu không có external expected inventory.

## Kết quả chính xác

| Nhóm | Video | Mapping rows = caption rows = feature rows | Object JSON | Ảnh local (metadata) | Ảnh trong ZIP |
|---|---:|---:|---:|---:|---:|
| L04 | 30 | 7,471 | 7,471 | 7,471 / 7,471 | 7,471 listed |
| L05 | 31 | 7,447 | 7,447 | 7,447 / 7,447 | 7,447 listed |
| L06 | 31 | 8,028 | 8,028 | **0 / 8,028 — missing** | **0 — missing** |
| Tổng | **92** | **22,946** | **22,946** | **14,918 / 22,946** | **14,918 listed** |

Mỗi video có một mapping CSV, caption JSON, caption-mapping JSON, NPY features,
index file và index manifest: lần lượt **30/31/31** file mỗi loại cho L04/L05/L06,
tổng **92 file mỗi loại**. Mapping/caption/caption-mapping/features/manifest đều
`ok`, objects đủ độ phủ `present`; không có cross-artifact issues. Nội dung
**92 index vẫn `unknown`**, dù file và manifest hashes hợp lệ. Vì vậy artifact
overall của cả 92 video là `unknown`, không phải `ready`.

Local image root dùng cho report là `dataset/data`: đủ metadata ảnh L04/L05,
thiếu toàn bộ L06. Drive `data/L04`, `data/L05` là directory placeholders theo
kiểm tra nguồn đã xác nhận; ảnh thực thể chưa chứng minh truy cập được. Một lượt
diagnostic có thêm Drive directory root hết timeout 120 giây; report chính dùng
local image root và các Drive ZIP, không lấy placeholders làm bằng chứng sẵn sàng.
Drive `data/L06` không tồn tại.

Ba ZIP đều liệt kê đủ artifact files/object JSON tương ứng; không có unsafe hay
duplicate members. Số central-directory entries lần lượt là **15,152 / 15,111 /
8,290**. **L06.zip không chứa ảnh**; cần nguồn ảnh L06 riêng. Muốn dùng ảnh từ
Drive ZIP phải **giải nén/đưa vào data root trước runtime**, ở một bước được phê
duyệt riêng; lượt này không thực hiện. Ảnh L04/L05 đã có metadata local không
được xem là kiểm thử đọc/hiển thị end-to-end.

## Lệnh tái lập (output phải chưa tồn tại)

```powershell
python -B -m tools.temporal_preflight `
  --groups L04,L05,L06 `
  --image-root dataset/data `
  --archive 'G:\My Drive\dataAIC\L04.zip' `
  --archive 'G:\My Drive\dataAIC\L05.zip' `
  --archive 'G:\My Drive\dataAIC\L06.zip' `
  --output benchmark_results/l04_l06_preflight_20261107_v2.json

python -B -m tools.p0_audit `
  --source-run benchmark_results/human_holdout_v3_evaluation_20261102 `
  --labels benchmark_results/human_holdout_v3_seed2026110203_v2/frozen_human_labels_v1.json `
  --mapping-root dataset/mapping `
  --caption-root dataset `
  --output benchmark_results/p0_audit_20261107_v7.json
```

## Reports và determinism

Ở lượt ban đầu (preflight v1 và P0 v7), hai CLI exit `0`, đã chạy lại với cùng inputs và output khác trong
`C:\Users\Khanh\AppData\Local\Temp\opencode`; report bytes/hashes trùng nhau.

| Report | SHA-256 |
|---|---|
| `benchmark_results/l04_l06_preflight_20261107.json` | `6198a11d5697904803719ab7d28f44b28c99a5275afa2341c8771c2e3c8d383a` |
| `benchmark_results/l04_l06_preflight_20261107_v2.json` | `26dd3720ca4968c9ca2b080f6fd5788af3484ee69f3c4e61a013470ca01b814a` |
| `benchmark_results/p0_audit_20261107_v7.json` | `8a7cd1e70d091d4e46846430bd285344b3b779d7482704cc61da2ef419a3e73d` |

Preflight v2 sửa finding duplicate destination sau normalize; report cũ giữ
nguyên. Hai lần chạy CLI v2 exit `0`, byte-identical; trên ba ZIP thực tế không
có rejected/duplicate destination. So với v1 chỉ thay đổi source hash của
`temporal_inventory.py`; summary, archive directory hashes và kết quả từng video
không đổi. Không chạy lại hoặc tạo phiên bản P0 mới vì logic P0 không đổi.

**P0 v7 vẫn audit saved human holdout run ngày 2026-11-02, không chạy retrieval
mới và không chứng minh retrieval của L04-L06.** P0 tool và saved inputs không
đổi nên v7 byte-identical với v6. Kết quả giữ nguyên: `status=issues`, 36 query
`ok` (26 event-frame + 10 sequence), 61 issues, 222 unresolved stage units.
Không overwrite v6 hoặc frozen labels. Xem [P0_AUDIT.md](P0_AUDIT.md) để hiểu
denominators và unknowns; không dùng holdout này để retune.

## Kiểm thử và blockers

- `python -B -m unittest tests.test_temporal_preflight tests.test_p0_audit`:
  **PASS, 78 tests**. Regression tests bao phủ groups override không đổi config,
  default behavior, ảnh L06 missing, ZIP inventory không đọc payload, exact
  duplicate, `/./`, `//`, case collision, file/directory collision,
  traversal/absolute/backslash/symlink, component không hợp lệ, unsafe archive
  không đóng góp image coverage, placeholder unknown, missing mapping group,
  output protection và determinism.
- Lượt trước (v1), `python -B -m unittest tests.test_ingest tests.test_runtime_config tests.test_temporal_preflight tests.test_p0_audit`:
  **85 tests, 4 errors do môi trường**: thiếu `sentence_transformers`, thiếu
  `pymilvus` (2 tests) và `kis.text` không import được trong runtime-config mock
  (`AttributeError`). Không cài dependency hoặc sửa test ngoài phạm vi.
- Môi trường không có Black, Ruff, Mypy và repo không có cấu hình formatter/
  linter/typecheck riêng. Kiểm tra syntax, whitespace trực tiếp và
  `git diff --check` thay thế không đồng nghĩa đã chạy các công cụ bị thiếu.

Chưa bật L04-L06 trong production config. Blockers còn lại: thiếu 8,028 ảnh L06,
index contents chưa kiểm tra, chưa verify image payload/runtime/end-to-end
retrieval và chưa có đầy đủ dependency để chạy toàn bộ required suite.
