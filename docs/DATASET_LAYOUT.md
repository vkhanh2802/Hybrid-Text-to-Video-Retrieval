# Cấu trúc dataset thống nhất và chuyển keyframe L02–L08

## Layout canonical

```text
<project-root>/
  config/                                      # chỉ cấu hình không bí mật
  dataset/
    caption_generator/L02/L02_V001/L02_V001.json
    caption_mapping/L02/L02_V001/L02_V001.json
    index/L02/L02_V001/L02_V001.index
    index/L02/L02_V001/L02_V001.meta.json
    mapping/L02/L02_V001/L02_V001.csv
    clip-features-32/L02/L02_V001/L02_V001.npy
    objects/L02/L02_V001/001.json
    ocr/L02/L02_V001/001.json
    preparation/L02/L02_V001/L02_V001.json
    data/L02/L02_V001/001.jpg
  archives/                                    # ZIP gốc, không dùng trực tiếp để đọc ảnh
    L02.zip
  reports/
  gallery/
```

Tất cả dữ liệu theo video dùng **component / group / video / file**. Giữ nguyên
basename artifact, video ID và số ảnh; không renumber, sửa CSV/NPY/caption hoặc
rebuild index chỉ vì đổi thư mục. Manifest caption dùng hash nội dung nên vẫn
kiểm được sau chuẩn hóa. ZIP là archive theo nhóm, nằm riêng ngoài dataset.

## Tương thích và giữ nguồn

- `canonical_video_artifact_path` và các writer dùng folder video mới.
- `resolve_video_artifact` vẫn đọc flat/grouped legacy khi chỉ có một bản.
- `CaptionRetriever` tìm recursive **trong group được chọn**, đọc được layout mới.
- Object/image folders vốn đã theo group/video; giữ nguyên quy tắc đó.
- Public OCR hỗ trợ CSV ở `mapping/group/video/video.csv`, vẫn giữ kiểm tra
  không symlink/cloud và phát hiện duplicate.
- Không để hai bản cũ/mới cho cùng artifact trong một root vì loader từ chối
  duplicate. Bản chuẩn hóa được tạo ở **project root mới**, không gộp lên nguồn.
- Dataset nguồn laptop, `~/KISBranche`, `~/kis-data-*` và pilot L01 cũ đều giữ
  nguyên. Không xóa ZIP hoặc bản giải nén sau truyền. Không đổi scope production
  hoặc tự chuyển backend DRES sang root mới.

## Công cụ trên máy trường

`tools/normalize_school_dataset.py`:

1. Kiểm ZIP metadata và SHA-256 local trước truyền, kiểm size/mtime để không dùng
   bản ZIP vừa bị thay đổi giữa lúc chạy. Không chuyển credentials.
2. SFTP known-host checking vào `~/kis-normalized-<UTC timestamp>/archives` mới;
   `.partial` được tạo rỗng trước `put -a`, không ghi đè thư mục người dùng.
3. Snapshot source riêng không có `.env`, config local, Drive credentials/Git.
4. Sao chép artifact có sẵn trên server sang layout mới, đối chiếu SHA-256 nguồn
   và đích từng file với manifest lần chuyển trước. Không rebuild hoặc sửa bytes.
5. Sao chép ảnh L01 đã xác minh từ pilot trước vào `dataset/data/L01`; không
   tải/chuyển lại ZIP L01 qua VPN.
6. Kiểm hash ZIP từng nhóm trước giải nén; bỏ nesting download, dùng đúng
   `data/group/video/filename`. Đọc hết member kiểm CRC; giữ hash từng ảnh.
7. Tên duplicate do browser như `020(1).jpg` chỉ được gộp khi có primary
   `020.jpg` và bytes/SHA-256 giống hệt. Khác nội dung hoặc không có primary thì
   dừng, không tự chọn đè hoặc tự suy ảnh đúng.
8. Đối chiếu CSV mapping và **giải mã toàn bộ ảnh** bằng Pillow, ghi missing/extra/
   decode errors. Không coi có file trên đĩa là đã đọc được ảnh.
9. Thử retrieval CPU riêng từng nhóm: SigLIP + BM25 + MiniLM + RRF, 100 frame,
   giải mã 100 ảnh và tạo gallery. Loại identity frame_idx mơ hồ khỏi các nhánh
   pilot, giữ nguyên dữ liệu gốc và ghi rõ số identity loại.
10. Ghi `LAYOUT_MANIFEST.json`, `reports/Lxx_images.json`, `reports/Lxx/CPU_REPORT.json`
    và `IMPORT_VERIFIED.json`. Trạng thái partial/failed không được báo là pass.

Ví dụ lần chuyển được người dùng yêu cầu:

```powershell
& "$env:USERPROFILE\miniconda3\envs\focusguard_ai\python.exe" -u -m tools.normalize_school_dataset --zip-dir dataset/data --groups L02,L03,L04,L05,L06,L07,L08 --target <user>@<school-host> --artifact-source ~/kis-data-20261114T145713Z --l01-root ~/kis-l01-pilot-20261114T152802Z --output-dir dist/school-normalized-l02-l08-20261114-01
```

Output phải mới. Nếu VPN lỗi, giữ `.partial` và STATE.json để kiểm tra; không
chạy lại toàn workflow vào đích đã có hoặc xóa dữ liệu để thử lại.

## L06 và duplicates trong đợt này

ZIP L06 cũ chỉ có 3.894/8.028 ảnh nên người dùng yêu cầu tạm dừng. Sau đó người
dùng tải ZIP mới và cho phép chuyển: bản mới **8.028/8.028**, không thiếu/extra,
1.352.372.190 bytes, nesting `L06-KF/L06/L06_V001/001.jpg`. Chỉ bản mới được chọn.

L03 có một `020(1).jpg`, L07 có `226 (1).jpg` và `226(1).jpg`. Đã kiểm SHA-256
local và server, cả ba trùng nội dung với primary cùng số ảnh. ZIP gốc vẫn được giữ.

## Kết quả đã xác minh ngày 14/11/2026

Root mới trên máy trường: **`~/kis-normalized-20261114T160546Z`**.
Dataset runtime là `<root>/dataset`; ZIP gốc là `<root>/archives/Lxx.zip`.

- Chuẩn hóa và xác minh SHA-256 **349.100 artifact/config**, 3.437.064.840 bytes,
  giữ nguyên contents và các root cũ.
- L02–L08: **52.770 ảnh / 214 video**, tất cả khớp mapping và giải mã thành công.
  Số ảnh theo nhóm: L02 7.616; L03 7.119; L04 7.471; L05 7.447;
  **L06 8.028**; L07 7.364; L08 7.725. Không missing/extra/decode errors.
- L01 sao chép nội bộ **7.339 ảnh**, coverage/decode đạt; cây chung có 60.109 ảnh.
- CPU smoke từng nhóm trả đúng **100 frame khác nhau** và 100 thumbnail thật.
- Gallery HTTP của cả bảy nhóm đạt; thumbnail và ảnh gốc mỗi nhóm tải/giải mã
  được. L06 được kiểm trực tiếp trong trình duyệt, có screenshot lưu local.
- `IMPORT_VERIFIED.json` có `status=passed`; các ZIP, dataset nguồn laptop và
  các root server trước vẫn giữ nguyên. HTTP server/tunnel test là tạm thời.

Chi tiết và bằng chứng:
[báo cáo import/CPU/HTTP](../benchmark_results/school_normalized_l02_l08_20261114/SUMMARY.md).
Chưa repoint/restart backend DRES production hoặc triển khai remote worker/RPC.

## Giới hạn phép thử CPU

Thử riêng từng nhóm, query tiếng Anh cố định, không gọi Groq/Qwen hoặc DRES.
Visual dùng exact cosine trên NPY thay Milvus. Không cài đè venv/model cache đang
có. Đây là kiểm chứng layout, đọc ảnh và retrieval; không phải benchmark accuracy
hoặc cam kết latency full corpus/production. Web/tunnel xem gallery phải bind
loopback và chỉ dừng đúng process do test tạo.
