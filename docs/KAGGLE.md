# Chuẩn bị dữ liệu trên Kaggle

**Đã có keyframes, không cần xử lý video?** Dùng
[`notebooks/keyframesprocessing.ipynb`](../notebooks/keyframesprocessing.ipynb)
và xem [hướng dẫn từ keyframes](KAGGLE_FROM_KEYFRAMES.md). Runner mới giữ các
stage feature/object/caption/index, yêu cầu mapping CSV gốc, không decode video.

## 1. Chọn gói code

Chạy tại project root:

```powershell
python -m tools.package_kaggle
python -m tools.package_kaggle --profile keyframes-caption
```

| Gói trong `dist/` | Các bước chạy |
|---|---|
| `kis-kaggle-code.zip` | Keyframes → SigLIP → objects → caption → text index |
| `kis-kaggle-keyframes-caption.zip` | Keyframes → caption |

Cả hai dùng InternVL3.5-1B-HF, batch caption 16, tối đa 96 token.
Revision model được pin trong `config/models.json`. Bật **Internet** và **GPU**
trong notebook Kaggle, upload gói code và video thành Dataset rồi Add Input.

## 2. Cài và chạy

Giải nén ZIP vào `/kaggle/working`, hoặc đặt `CODE_DIR` tới folder `kis_kaggle`
đã được Kaggle giải nén. Cấu trúc gói code gồm `kis/`, `tools/`, `config/`.

```python
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path('/kaggle/working/kis_kaggle')
VIDEO_DIR = Path('/kaggle/input/kis-videos')
subprocess.run([
    sys.executable, '-m', 'pip', 'install',
    '-r', str(CODE_DIR / 'requirements-kaggle.txt'),
], check=True)
subprocess.run([
    sys.executable, '-u', '-m', 'tools.kaggle_prepare',
    '--video-dir', str(VIDEO_DIR),
    '--output-root', '/kaggle/working/kis_output',
    '--caption-batch', '16', '--seed', '42',
], cwd=CODE_DIR, check=True)
```

Entry point giống nhau cho hai gói; gói caption-only không chạy feature/object/index.
Có thể lặp `--video-id L19_V001` để chọn video. Gói full nhận thêm
`--visual-batch 8 --object-batch 8`. Keyframes lấy frame 0 rồi random gap 80–160 frame.

Caption dùng BF16 native khi GPU hỗ trợ, còn T4/P100 dùng FP32. Batch tự giảm
khi CUDA OOM. `--caption-4bit` bật NF4; `--caption-backend qwen` chọn Qwen.
Khi đổi model/prompt, thêm `--restart-captions` một lần để lưu caption cũ vào
backup rồi sinh lại; bỏ flag đó khi resume. Khi đổi session, copy artifact đã
lưu vào output root trước khi chạy lại cùng config.

## 3. Nhập output vào pipeline local

Save Version để giữ output và tải ZIP `kis_output-artifacts.zip` (full) hoặc
`kis_output-keyframes-captions.zip`. Gói caption-only cố lưu ZIP `-partial`
khi gặp lỗi để có thể resume.

Giải nén vào folder nhập riêng. Chép `mapping`, `clip-features-32`, `objects`
của các nhóm đến L21 vào `dataset/`; ảnh nếu có đặt trong `dataset/data/`.
Caption v2 được nhập bằng:

```powershell
python -m tools.ingest --source <folder-output> --rebuild-visual
```

Nếu chỉ cập nhật caption, bỏ `--rebuild-visual`. Caption và index đang dùng nằm
trong `dataset/caption_generator/`, `dataset/caption_mapping/` và `dataset/index/`.
Khởi động lại web/kernel sau khi nhập.
