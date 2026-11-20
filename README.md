# KIS Framefinder

Tìm keyframe theo mô tả (Known-Item Search) trên **L01–L24**, giao diện web để chọn và nộp
frame lên DRES.

## Pipeline

```text
Query (VI/EN) → Gemini planner (fallback OpenCode) → query variants + entities
        ↓
  SigLIP2-384 (ảnh)  ·  BM25 caption  ·  MiniLM caption  ·  OCR (L01–L21)
        ↓
  Weighted RRF 0.5 / 0.2 / 0.3 → object evidence → 200 frame khác nhau
```

- Visual: `timm/ViT-B-16-SigLIP2-384`, vector trong `dataset/siglip2-features/`
  (cấu hình ở `config/models.json`). Qwen VLM tắt mặc định.
- Tìm frame tương tự: nút trên card, phím `S` khi rê chuột lên ảnh, hoặc Ctrl+click (30 frame gần nhất).
- Nộp DRES: xem trước JSON, chỉnh keyframe trước/sau hoặc ±1 frame trước khi gửi.

## Cài đặt

Python 3.11, Windows PowerShell (script khởi động), Docker nếu chạy Milvus local.

```powershell
pip install -r requirements.txt
```

Key planner: đặt biến môi trường Windows User `GEMINI_API_KEY` (hoặc `GROQ_API_KEY` khi dùng
`-Provider groq`). Không ghi key vào file trong repo. Xem `.env.example`.

## Dữ liệu (không có trong repo)

```text
dataset/
  mapping/<group>/<video>.csv                    # n, pts_time, fps, frame_idx
  siglip2-features/<group>/<video>/<video>.npy   # float16, 768 chiều, theo thứ tự dòng mapping
  caption_generator/<group>/<video>/<video>.json
  caption_mapping/<group>/<video>/<video>.json
  index/<group>/<video>/<video>.index + .meta.json   # MiniLM
  objects/<group>/<video>/*.json
  data/<group>/<video>/<n>.jpg                   # ảnh keyframe để hiển thị
```

Tạo vector SigLIP2 (GPU nếu có, đọc thẳng từ zip keyframe, chạy lại được giữa chừng):

```powershell
python tools/embed_siglip2.py --dataset dataset --output dataset/siglip2-features --groups L01 L02 --zips dataset/data/L01.zip dataset/data/L02.zip
```

Build caption index: `python -m tools.ingest`. Đóng gói ảnh từ máy khác: `tools/prepare_siglip2_bundle.py`.

## Chạy demo

**Cách 1: worker CPU trường (khuyên dùng).** Retrieval chạy trên máy trường qua SSH tunnel.

```powershell
python -m tools.deploy_school_worker --target <user>@<school-host> --dataset <remote-dataset> --output dist/worker-01 --settings config/school-runtime-siglip2.local.json --video-groups L01 L02 ...
.\start_school.ps1
```

**Cách 2: chạy local với Milvus.**

```powershell
docker compose up -d
python -m tools.build_milvus build      # tạo collection siglip2_keyframes
.\start_kis.ps1
```

Mở `http://127.0.0.1:8000` (đề DRES) hoặc `http://127.0.0.1:8000/?mode=manual` (tự nhập query).

## DRES

Đặt `KIS_DRES_BASE_URL` / `KIS_DRES_RUN_ID` hoặc `config/dres.local.json` (gitignored).
Đăng nhập trong giao diện; session chỉ giữ trong RAM của server. Chỉ nộp video có trong collection
của run, DRES từ chối tên video không có (`Could not find media item`).
