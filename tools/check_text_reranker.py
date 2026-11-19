"""Inspect the pinned text model cache without downloading/loading any model."""

import json
from pathlib import Path

from kis.config import load_runtime_config


def check_cache(config):
    from huggingface_hub import try_to_load_from_cache

    files = {}
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "vocab.txt", "model.safetensors"):
        result = try_to_load_from_cache(config.model, name, revision=config.revision)
        files[name] = isinstance(result, str) and Path(result).is_file()
    ready = (files["config.json"] and files["model.safetensors"]
             and files["tokenizer_config.json"] and (files["tokenizer.json"] or files["vocab.txt"]))
    return {"config": config.public(), "cache_ready": ready, "cached_files": files,
            "model_loaded": False, "downloaded": False,
            "note": "Cache presence only; real checkpoint compatibility/quality requires a separately approved inference run."}


if __name__ == "__main__":
    report = check_cache(load_runtime_config().text_reranker)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["cache_ready"] else 2)
