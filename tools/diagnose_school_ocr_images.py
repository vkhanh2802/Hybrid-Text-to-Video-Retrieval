"""Non-inference notebook/pixel audit. Never installs Paddle or changes sources.

Manual diagnostic crops are NOT detector boxes. Downscaled panels simulate only
the documented detector geometry, not an OCR rerun or recognition result.
"""

import ast
import hashlib
from io import BytesIO
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "benchmark_results/school_ocr_accuracy_20261115_run01"


def notebook_worker(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    for cell in document["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id == "WORKER_SOURCE" for n in node.targets):
                return document, ast.literal_eval(node.value)
    raise ValueError("Notebook embedded worker missing")


def detector_size(size, padding=32, limit=960):
    width, height = size
    w, h = width + 2 * padding, height + 2 * padding
    ratio = min(1., limit / max(w, h))
    return max(32, round(int(w * ratio) / 32) * 32), max(32, round(int(h * ratio) / 32) * 32)


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    notebook = ROOT / "notebooks/kaggle_ocr_simple_gpu.ipynb"
    document, worker = notebook_worker(notebook)
    namespace = {"__name__": "ocr_diagnostic_no_inference", "__file__": str(notebook)}
    # Worker imports stdlib only at module level; Paddle is lazy in Adapter.__init__.
    exec(compile(worker, "<audited-notebook-worker>", "exec"), namespace)
    text = "Quảng Ngãi\nMÌ TRỨNG\n3,2 tấn cá khoai chứa formol"
    record = dict(status="ok", video_id="L09_V011", keyframe_n=120, relative_path="L09_V011/120.jpg",
                  lines=[{"text": line} for line in text.splitlines()])
    public = namespace["public_record"](record)
    raw = json.dumps([public], ensure_ascii=False).encode("utf-8")
    assert json.loads(raw)[0]["ocr_text"] == text
    sources = json.loads((INPUT / "POST_FREEZE_SOURCE_OCR_AUDIT.json").read_text(encoding="utf-8"))
    lookup = {r["id"]: r for r in sources["rows"]}
    specs = [
        ("q03_purple_quangngai", "anchors/L09_V011_120.jpg", (0, 645, 950, 705)),
        ("q04_noodles_text", "anchors/L08_V002_050.jpg", (260, 310, 650, 490)),
        ("q05_formol_literal", "anchors/L02_V015_181.jpg", (0, 640, 880, 705)),
    ]
    rows = []
    for qid, name, box in specs:
        path = INPUT / name
        original = path.read_bytes()
        padded, decoded_size = namespace["decode"](original)
        with Image.open(BytesIO(original)) as im:
            image = im.convert("RGB")
        # Verify the real wrapper preserves every original decoded raster pixel.
        assert padded.size == (image.width + 64, image.height + 64)
        assert padded.crop((32, 32, 32 + image.width, 32 + image.height)).tobytes() == image.tobytes()
        dw, dh = detector_size(image.size)
        reduced = padded.resize((dw, dh), Image.Resampling.BILINEAR)
        sx, sy = dw / padded.width, dh / padded.height
        mapped = tuple(round((v + 32) * (sx if i % 2 == 0 else sy)) for i, v in enumerate(box))
        native_crop = image.crop(box)
        det_crop = reduced.crop(mapped)
        native_crop.save(output / (qid + "_native.png"))
        det_crop.save(output / (qid + "_det_simulated.png"))
        # Nearest-neighbor enlargement for honest pixel inspection, not super-resolution.
        a = native_crop.resize((native_crop.width * 2, native_crop.height * 2), Image.Resampling.NEAREST)
        b = det_crop.resize((det_crop.width * 3, det_crop.height * 3), Image.Resampling.NEAREST)
        width = max(1100, a.width + 20, b.width + 20)
        sheet = Image.new("RGB", (width, a.height + b.height + 160), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 10), qid + " | manual diagnostic crop, NOT OCR detector box", fill="black")
        draw.text((10, 35), "Original raster crop enlarged 2x nearest-neighbor (no enhancement)", fill="black")
        sheet.paste(a, (10, 60))
        y = a.height + 80
        draw.text((10, y), f"Detector-only resize simulation {padded.size} -> {(dw, dh)}, crop enlarged 3x; no OCR inference", fill="black")
        sheet.paste(b, (10, y + 25))
        sheet.save(output / (qid + "_compare.png"))
        rows.append({"id": qid, "source_image": str(path.relative_to(ROOT)),
                     "image_sha256": hashlib.sha256(original).hexdigest(), "size": list(image.size),
                     "manual_crop_xyxy": list(box), "original_crop_size": list(native_crop.size),
                     "wrapper_padding_size": list(padded.size), "detector_simulated_size": [dw, dh],
                     "detector_scale_x": sx, "detector_scale_y": sy,
                     "wrapper_preserves_decoded_pixels": True, "source_ocr": lookup[qid]})
    functions = {}
    lines = worker.splitlines()
    for node in ast.parse(worker).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {"decode", "public_record", "Adapter", "make_profile", "record_for"}:
            functions[node.name] = {"embedded_worker_line": node.lineno, "end_line": node.end_lineno,
                                    "source": "\n".join(lines[node.lineno - 1:node.end_lineno])}
    report = {"status": "non_inference_audit_completed", "notebook": str(notebook.relative_to(ROOT)),
              "notebook_sha256": hashlib.sha256(notebook.read_bytes()).hexdigest(),
              "worker_sha256": hashlib.sha256(worker.encode()).hexdigest(),
              "execution_counts_present": sum(c.get("execution_count") is not None for c in document["cells"]),
              "output_cells_present": sum(bool(c.get("outputs")) for c in document["cells"]),
              "models": namespace["MODELS"], "options": namespace["OPTIONS"],
              "unicode_export_roundtrip_exact": True, "rows": rows, "code_excerpts": functions,
              "limitations": ["No OCR engine installed or run; cannot apportion causal model-vs-image error without controlled rerun",
                              "Public records lack confidence/boxes/image hashes; cannot prove exports came from this exact notebook version",
                              "Detector simulation not actual model output or recognizer input; PaddleX crops from original raster using detector polygons",
                              "Manual crop rectangles are analysis aids, not recovered OCR polygons"]}
    with (output / "REPORT.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in {"code_excerpts", "rows"}}, ensure_ascii=True))
    print(json.dumps({"images": [{"id": r["id"], "size": r["size"], "padded": r["wrapper_padding_size"],
                      "detector_simulated": r["detector_simulated_size"]} for r in rows]}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
