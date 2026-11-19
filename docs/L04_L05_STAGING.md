# L04/L05 staging offline — preflight và structural smoke

`tools/l04_l05_staging.py` chỉ cho phép `preflight` / `structural-smoke` trên
**L04 V001–V030, L05 V001–V031**: 61 video, 14.918 visual records
(L04: 7.471; L05: 7.447). Không enumerate/read/ingest/query L06, không quét root,
không dùng production pipeline config, service client, Docker, network, Drive
hoặc encoder/model. Input phải là local hydrated files; reparse/symlink/cloud
placeholder bị từ chối hoặc báo `unknown`, không hydrate.

## Chạy lại

Output directory phải **chưa tồn tại**, nằm ngoài dataset và không chứa dataset.
Tool reserve directory bằng exclusive create; nếu bị gián đoạn có thể để lại
directory chưa có report. Không tự xóa hay dùng lại directory đó; chọn version mới.

```powershell
$py = 'C:\Users\Khanh\miniconda3\envs\focusguard_ai\python.exe'
& $py -B -m tools.l04_l05_staging structural-smoke `
  --dataset-root 'C:\Users\Khanh\Desktop\KISBranche\dataset' `
  --image-root 'C:\Users\Khanh\Desktop\KISBranche\dataset\data' `
  --output-dir benchmark_results/l04_l05_staging_20261107_v1
```

`preflight` thực hiện cùng validation nhưng không tạo/query structural index.
Exit `0`: mọi kiểm tra trong mode đã chọn `ok`; `1`: có issues/unknown;
`2`: input/output/CLI không hợp lệ. Exit `0` **không phải** production approval.

## Ý nghĩa report

- Đọc trực tiếp canonical mapping/caption/features/object/image, tái dùng helper
  local smoke nhưng **không gọi hàm inventory hoặc local smoke `run`**.
  Objects chỉ probe hai đường dẫn grouped/legacy explicit; không enumerate.
- Đối chiếu mapping uniqueness `(video_id, frame_id)`, frame/time/keyframe order,
  caption identity/count/order, object JSON coverage, image decode và caption hash.
  Thiếu bằng chứng không được coi là pass. Hashes/digests cho phép so sánh inputs.
- Visual NPY phải có shape `(mapping_count, 768)`, float, finite và nonzero.
  NumPy exhaustive IP local dùng **visual vectors**, float64, fixed reduction order,
  không normalize lại. **Không đọc/query caption FAISS 384 chiều**.
- Lấy deterministic một hàng giữa mỗi video: 61 self-retrieval queries, top-5,
  kiểm tra result IDs/order/scope/mapping. Tie bằng score chính xác được sắp global
  row ID tăng dần, báo tie ở top/cutoff. Self không bắt buộc top-1 (raw vectors có
  thể khác norm; duplicate vectors có thể tie). Self rank **không là accuracy**.
- `milvus.status="not_started"`; `accuracy.status`, `Hit@K`, `MRR` đều
  `"not_evaluated"` vì chưa có L04/L05 queries/qrels. Đây **chỉ là structural
  smoke, không phải Milvus integration test hoặc semantic retrieval benchmark**.
- Safety proof theo construction: allowlist kiểm tra trước path/lstat, chỉ direct
  paths; tests chặn enumeration, network/import service và mọi L06 filesystem I/O.
  Đây không phải OS sandbox; inputs phải ổn định trong khi chạy.
- Provenance gồm source/helper/config-model hashes, versions và roots local;
  không ghi credentials, không đọc production `config/pipeline.json`.
  Output path/timestamp không nằm trong report để chạy lại byte-identical.

## Kiểm chứng implementation (2026-11-07)

Hai CLI dữ liệu thật đã hoàn tất với **exit `0` cho cả canonical và temp repeat**;
thu kết quả từ hai process đã chạy, không restart hay ghi đè output.

| Phạm vi | Videos | Visual/mapping records | Object JSON / ảnh decoded |
|---|---:|---:|---:|
| L04 | 30 | 7.471 | 7.471 / 7.471 |
| L05 | 31 | 7.447 | 7.447 / 7.447 |
| Tổng | **61** | **14.918** | **14.918 / 14.918** |

Preflight, exact counts, vectors và index đều `ok`. Visual dimension **768**,
finite/nonzero; norm min/max lần lượt `0.999999912444656` /
`1.0000001087466905`. NumPy exhaustive float64-IP chạy **61 queries**, top-5,
**305 result records** hợp scope/mapping. Self rank 1 ở 61 queries chỉ là kết quả
structural self-retrieval, **không phải Hit@1/accuracy**. Mẫu thật không có tie ở
top/cutoff; tie policy được kiểm tra bằng fixtures có duplicate vectors.
`milvus=not_started`; accuracy/Hit@K/MRR vẫn `not_evaluated`.

- Canonical: `benchmark_results/l04_l05_staging_20261107_v1/report.json`.
- Repeat: `C:\Users\Khanh\AppData\Local\Temp\opencode\l04_l05_staging_repeat_98c158cba74149d98a0611e8af904560\report.json`.
- Hai reports **byte-identical**, mỗi file **180.696 bytes**, cùng SHA-256:
  `bceb72ea19708499604d706cd100ee8bc07ee9b1052f332cd8b5843f00baf913`.

- `C:\Users\Khanh\miniconda3\envs\focusguard_ai\python.exe -B -m unittest
  tests.test_l04_l05_staging tests.test_l04_l05_local_smoke`: **PASS, 38 tests**.
- `python -B -m unittest tests.test_temporal_preflight tests.test_p0_audit`:
  **PASS, 78 tests** (fixtures tạm, không audit lại dataset/P0 reports).
- Chạy gộp cả bốn module trên bằng Python `focusguard_ai`: **PASS, 116 tests**.
- Black/Ruff/Mypy không có trong môi trường (`No module named ...`); không cài
  dependency. Syntax compile in-memory, whitespace scan và `git diff --check`
  được dùng bổ sung, không thay thế formatter/linter/typecheck thực sự.

## Điều kiện cho bước tiếp theo (chưa thực hiện)

Real isolated Milvus staging cần phê duyệt riêng: endpoint local/staging cô lập,
credentials qua secret mechanism (không ghi report), namespace/collection tạm
không alias production, schema/IP visual dimension 768 và explicit L04/L05
allowlist, dependency phù hợp đã có sẵn, kế hoạch ingest/rollback/cleanup được
duyệt. Không mở rộng scope, deploy hay bật production config từ kết quả offline.
Đánh giá semantic còn cần query/qrels L04/L05 được kiểm định/freeze riêng và
protocol accuracy/Hit@K/MRR; structural smoke không thay thế được bước đó.
