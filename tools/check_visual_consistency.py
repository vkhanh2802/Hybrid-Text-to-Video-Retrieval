"""Small dev-only read-only audit: cached SigLIP image vectors vs stored NPY.

Computes a few temporary vectors on CPU, never regenerates/publishes corpus
features. Does not load Qwen, MiniLM, captions, or holdout images.
"""

import argparse
from contextlib import redirect_stdout
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from kis.artifact_io import atomic_write_json
from kis.config import VISUAL_CONFIG
from kis.event_retrieval import LocalEventScorer, _unit_vectors
from kis.paths import local_image_settings
from tools.groundtruth_evaluation import load_labels


def select_dev_images(labels):
    queries = [q for q in labels["queries"] if q["split"] == "dev" and q.get("task") == "image"]
    if not 1 <= len(queries) <= 4:
        raise ValueError("Diagnostic budget is one to four development images")
    return queries


def run(labels_path, output):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    labels = load_labels(labels_path)
    queries = select_dev_images(labels)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    from kis.local_images import DriveImages
    from kis.visual import SiglipEncoder
    import torch
    torch.set_num_threads(4)
    settings = local_image_settings()
    folder = os.environ.get("KIS_DRIVE_FOLDER_ID") or settings.get("drive_folder_id")
    if not folder:
        raise RuntimeError("Configure read-only Drive image source")
    source = DriveImages(folder, credentials_file=settings.get("credentials_file"))
    records = []
    try:
        with (output / "audit.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            encoder = SiglipEncoder(device="cpu")
            for query in queries:
                data, _ = source.read(query["video_id"], query["anchor_keyframe_n"])
                digest = hashlib.sha256(data).hexdigest()
                if digest != query["image_sha256"]:
                    raise ValueError("Drive image changed since annotation; stop audit")
                frames, vectors = LocalEventScorer._video_data(query["video_id"])
                positions = [i for i, f in enumerate(frames) if f["frame_id"] == query["frame_id"] and f["keyframe_n"] == query["anchor_keyframe_n"]]
                if len(positions) != 1 or vectors is None:
                    raise ValueError("Missing or ambiguous stored visual vector")
                stored = _unit_vectors(vectors[positions])[0]
                with Image.open(BytesIO(data)) as original:
                    image = original.convert("RGB")
                    try:
                        computed = _unit_vectors(encoder.encode_images([image]))[0]
                    finally:
                        image.close()
                text = _unit_vectors([encoder.encode_text(query["english"])])[0]
                records.append({"query_id": query["query_id"], "video_id": query["video_id"],
                                "keyframe_n": query["anchor_keyframe_n"], "image_sha256": digest,
                                "new_vs_stored_image_cosine": float(np.dot(computed, stored)),
                                "new_image_text_cosine": float(np.dot(computed, text)),
                                "stored_image_text_cosine": float(np.dot(stored, text))})
                atomic_write_json(output / "report.json", {"split": "dev_only", "model": VISUAL_CONFIG,
                                  "label_sha256": hashlib.sha256(Path(labels_path).read_bytes()).hexdigest(),
                                  "corpus_modified": False, "temporary_image_vectors": len(records), "records": records})
        print(json.dumps({"output": str(output.resolve()), "records": records}, indent=2))
    finally:
        LocalEventScorer._video_data.cache_clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.labels, args.output_dir)
