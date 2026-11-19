"""Read-only recognizer metadata audit. Downloads config/text ONLY, never weights.

Compares output alphabet against Vietnamese strings, then optionally scans the
already supplied school snapshot using stdlib SSH. No OCR packages/inference.
"""

import hashlib
import json
import ssl
from pathlib import Path
import unicodedata
from urllib.request import urlopen

from tools.transfer_school_data import remote_python


URLS = {
    "inference.yml": "https://huggingface.co/PaddlePaddle/latin_PP-OCRv5_mobile_rec/resolve/ab2cd5cc5fa6309be2e5acdfe66eca2c2c127d57/inference.yml",
    "training.yml": "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/v3.2.0/configs/rec/PP-OCRv5/multi_language/latin_PP-OCRv5_mobile_rec.yml",
    "training_dict.txt": "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/v3.2.0/ppocr/utils/dict/ppocrv5_latin_dict.txt",
    "detector_processors.py": "https://raw.githubusercontent.com/PaddlePaddle/PaddleX/v3.2.0/paddlex/inference/models/text_detection/processors.py",
    "recognizer_processors.py": "https://raw.githubusercontent.com/PaddlePaddle/PaddleX/v3.2.0/paddlex/inference/models/text_recognition/processors.py",
    "ocr_pipeline.py": "https://raw.githubusercontent.com/PaddlePaddle/PaddleX/v3.2.0/paddlex/inference/pipelines/ocr/pipeline.py",
    "ocr_api.py": "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/v3.2.0/paddleocr/_pipelines/ocr.py",
}


def missing_characters(text, alphabet):
    return sorted({c for c in unicodedata.normalize("NFC", text) if not c.isspace() and c not in alphabet})


def run(output, target):
    import yaml  # Existing PyYAML dependency, not an installation.
    import certifi  # Existing CA bundle; avoid malformed Windows certificate store.
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    context = ssl.create_default_context(cafile=certifi.where())
    for name, url in URLS.items():
        with urlopen(url, timeout=60, context=context) as response:
            raw = response.read(1024 * 1024 + 1)
            resolved = response.geturl()
        if len(raw) > 1024 * 1024:
            raise ValueError("Unexpected metadata size")
        with (output / name).open("xb") as stream:
            stream.write(raw)
        sources[name] = {"url": url, "resolved_url": resolved, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    info = yaml.safe_load((output / "inference.yml").read_text(encoding="utf-8"))
    dictionary = info["PostProcess"]["character_dict"]
    assert info["Global"]["model_name"] == "latin_PP-OCRv5_mobile_rec"
    alphabet = set(dictionary) | {" "}  # CTC decoder's extra space does not add accents.
    examples = ["Quảng Ngãi", "tấn", "chứa", "Thừa Thiên-Huế", "MÌ TRỨNG", "Hà Nội", "chiều", "nắng"]
    chars = "ảấứừếộềắụệịỈỰỨẤẢ"
    report = {"status": "metadata_audited", "model": info["Global"]["model_name"],
              "sources": sources, "alphabet_unique_entries": len(alphabet),
              "examples": [{"text": s, "missing": missing_characters(s, alphabet)} for s in examples],
              "key_characters": [{"char": c, "codepoint": f"U+{ord(c):04X}", "name": unicodedata.name(c),
                                  "present": c in alphabet} for c in chars],
              "combining_marks_in_alphabet": sorted(c for c in alphabet if len(c) == 1 and unicodedata.combining(c)),
              "limitations": ["Pinned official model metadata, not the user's original Kaggle cached inference.yml/checkpoint",
                              "Notebook pins package/model names but not downloaded model file hashes or revision",
                              "No inference rerun or new model/package download; dictionary alone cannot explain all visual errors"]}
    if target:
        # Count every Vietnamese character and combiner in all nine supplied groups.
        # Only small aggregate output crosses SSH, no OCR package imports remotely.
        script = r'''
import pathlib,json,collections,unicodedata
base=pathlib.Path.home()/'kis-ocr-import-20261114T212719Z/snapshot'
counts=collections.Counter();groups={};characters=0
for path in sorted(base.glob('L*.json')):
 rows=json.loads(path.read_bytes());groups[path.stem]=len(rows)
 for row in rows:
  text=row['ocr_text'];characters+=len(text)
  counts.update(c for c in text if ord(c)>127)
print(json.dumps({'groups':groups,'records':sum(groups.values()),'text_characters':characters,'non_ascii_counts':dict(counts)}))
'''
        corpus = remote_python(target, script)
        all_counts = corpus["non_ascii_counts"]
        corpus["key_character_counts"] = {c: all_counts.get(c, 0) for c in chars}
        corpus["non_ascii_characters_outside_alphabet"] = {c: n for c, n in all_counts.items() if c not in alphabet}
        report["supplied_corpus"] = corpus
    with (output / "REPORT.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in {"sources", "supplied_corpus"}}, ensure_ascii=True))
    if target:
        print(json.dumps({k: v for k, v in report["supplied_corpus"].items() if k != "non_ascii_counts"}, ensure_ascii=True))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target")
    args = parser.parse_args()
    run(args.output, args.target)
