"""Append explicitly selected, complete ZIP groups to a normalized school root.

Existing images, archives, artifact bytes, source snapshots and reports are never
overwritten. Each import keeps its own immutable source/config/report directory.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import zipfile

from tools.normalize_school_dataset import BOOTSTRAP, extract_group, image_relative, validate_images
from tools.transfer_school_data import CONFIG_FILES, ROOT, remote_python, sha256_file, ssh_options


def preflight(archive, mapping_root, group):
    if not re.fullmatch(r"L\d{2}", group):
        raise ValueError("Invalid group")
    archive = Path(archive)
    before = archive.stat()
    expected = set()
    for mapping in (Path(mapping_root) / group).rglob(f"{group}_V*.csv"):
        with mapping.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                key = (mapping.stem, int(row["n"]))
                if key in expected:
                    raise ValueError("Duplicate mapping identity")
                expected.add(key)
    actual, duplicates, total = {}, [], 0
    with zipfile.ZipFile(archive) as z:
        members = sorted(z.infolist(), key=lambda m: bool(re.search(r"\(\d+\)", PurePosixPath(m.filename).stem)))
        for member in members:
            if member.is_dir():
                continue
            if stat.S_ISLNK(member.external_attr >> 16) or member.flag_bits & 1:
                raise ValueError("Symlink/encrypted ZIP member")
            relative = image_relative(member.filename, group)
            if not 0 < member.file_size <= 25 * 1024 ** 2:
                raise ValueError("Invalid image size")
            total += member.file_size
            key = (relative.parts[1], int(Path(relative.name).stem))
            if re.search(r"\(\d+\)", PurePosixPath(member.filename).stem) and key not in actual:
                raise ValueError("Duplicate has no primary numbered image")
            if key in actual:
                primary = actual[key]
                if member.file_size != primary.file_size or hashlib.sha256(z.read(member)).digest() != hashlib.sha256(z.read(primary)).digest():
                    raise ValueError("Conflicting duplicate image")
                duplicates.append({"member": member.filename, "same_as": primary.filename})
            else:
                actual[key] = member
    if not expected or set(actual) != expected:
        raise ValueError(f"Incomplete {group}: expected={len(expected)}, missing={len(expected-set(actual))}, extra={len(set(actual)-expected)}")
    digest = sha256_file(archive)
    after = archive.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("ZIP changed during preflight")
    return {"images": len(actual), "videos": len({k[0] for k in actual}),
            "zip_bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
            "unpacked_bytes": total, "sha256": digest, "duplicates": duplicates}


def remote_run(config):
    base, root = Path(config["destination"]), Path(config["target_root"])
    dataset = root / "dataset"
    reports = base / "reports"
    reports.mkdir(mode=0o700)
    (base / "source/dataset").symlink_to(dataset, target_is_directory=True)
    results = {}
    for group in config["groups"]:
        archive = root / "archives" / f"{group}.zip"
        partial = archive.with_suffix(".zip.partial")
        if archive.exists() or (dataset / "data" / group).exists():
            raise FileExistsError("Refusing to overwrite existing group: " + group)
        if sha256_file(partial) != config["inventory"][group]["sha256"]:
            raise ValueError("ZIP checksum mismatch: " + group)
        partial.rename(archive)
        manifest = extract_group(archive, dataset / "data", group)
        coverage = validate_images(dataset, group, manifest["files"])
        (reports / f"{group}_images.json").write_text(json.dumps(dict(manifest, coverage=coverage)), encoding="utf-8")
        result = {k: v for k, v in manifest.items() if k != "files"}
        result["coverage"] = coverage
        if coverage["status"] == "passed":
            output = reports / group
            snippet = ("from tools.school_l01_pilot import run_cpu,QUERIES; "
                       f"run_cpu({str(dataset)!r},{str(dataset / 'data')!r},{str(output)!r},"
                       f"groups=({group!r},),queries=(QUERIES[0],),repetitions=1)")
            env = dict(os.environ, PYTHONPATH=str(base / "source"), CUDA_VISIBLE_DEVICES="",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="4",
                       MKL_NUM_THREADS="4", FAISS_OPT_LEVEL="generic")
            with (reports / f"{group}_cpu.log").open("x") as log:
                child = subprocess.run([config["python"], "-u", "-c", snippet], cwd=base / "source",
                                       env=env, stdout=log, stderr=subprocess.STDOUT)
            if child.returncode:
                result["retrieval"] = {"status": "failed", "log": str(reports / f"{group}_cpu.log")}
            else:
                cpu = json.loads((output / "CPU_REPORT.json").read_text())
                result["retrieval"] = {"status": cpu["status"], "runs": [
                    {k: run[k] for k in ("count", "unique_count", "retrieval_seconds", "images_decode_thumbnail_seconds")}
                    for run in cpu["runs"]], "ambiguous_frame_ids_excluded": cpu["ambiguous_frame_ids_excluded"]}
                (root / "gallery" / group).symlink_to(output / "gallery", target_is_directory=True)
        results[group] = result
        (reports / f"{group}_VERIFIED.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    result = {"status": "passed" if all(r["coverage"]["status"] == "passed" and r.get("retrieval", {}).get("status") == "passed" for r in results.values()) else "partial",
              "target_root": str(root), "import_root": str(base), "groups": results,
              "sources_deleted": False, "existing_artifacts_modified": False,
              "limitations": ["per-group CPU smoke, fixed English query", "no Groq/Qwen/DRES", "exact NPY cosine instead of Milvus"]}
    (base / "IMPORT_VERIFIED.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def orchestrate(args):
    groups = tuple(args.groups.split(","))
    if not groups or len(set(groups)) != len(groups) or any(not re.fullmatch(r"L\d{2}", g) for g in groups):
        raise ValueError("Select unique explicit Lxx groups")
    inventory = {g: preflight(Path(args.zip_dir) / f"{g}.zip", ROOT / "dataset/mapping", g) for g in groups}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    sources = [p for p in (ROOT / "kis").glob("*.py") if p.name != "local_images.py"]
    sources += [ROOT / "tools" / n for n in ("__init__.py", "append_school_keyframes.py", "normalize_school_dataset.py", "school_l01_pilot.py", "transfer_school_data.py")]
    sources += [ROOT / "config" / n for n in CONFIG_FILES]
    source_zip = output / "source.zip"
    with zipfile.ZipFile(source_zip, "x", zipfile.ZIP_DEFLATED) as z:
        for p in sources:
            z.write(p, p.relative_to(ROOT).as_posix())
    name = "keyframes-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prepared = remote_python(args.target, "import pathlib,shutil,json\n"
        f"root=pathlib.Path({args.target_root!r})\ngroups={groups!r}\n"
        "assert (root/'IMPORT_VERIFIED.json').is_file(), 'Not a verified normalized root'\n"
        "for g in groups:\n"
        "    assert (root/'dataset/mapping'/g).is_dir(), 'Missing mapping'\n"
        "    assert not (root/'dataset/data'/g).exists(), 'Images already exist'\n"
        "    assert not (root/'gallery'/g).exists(), 'Gallery already exists'\n"
        "    assert not (root/'archives'/(g+'.zip')).exists(), 'Archive already exists'\n"
        "    assert not (root/'archives'/(g+'.zip.partial')).exists(), 'Partial already exists'\n"
        f"assert shutil.disk_usage(root).free>{sum(v['zip_bytes']+v['unpacked_bytes'] for v in inventory.values())+3*1024**3}, 'Insufficient disk'\n"
        "(root/'imports').mkdir(mode=0o700,exist_ok=True)\n"
        f"base=root/'imports'/{name!r}\nbase.mkdir(mode=0o700)\n"
        "(base/'source.zip.partial').open('xb').close()\n"
        "for g in groups:\n    (root/'archives'/(g+'.zip.partial')).open('xb').close()\n"
        "print(json.dumps({'destination':str(base),'python':str(pathlib.Path.home()/'KISBranche/.venv/bin/python')}))")
    config = dict(prepared, target_root=args.target_root, groups=groups, inventory=inventory,
                  source_sha256=sha256_file(source_zip), source_members=[p.relative_to(ROOT).as_posix() for p in sources])
    (output / "STATE.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    uploads = [("source", source_zip, config["destination"] + "/source.zip.partial")]
    uploads += [(g, Path(args.zip_dir) / f"{g}.zip", args.target_root + f"/archives/{g}.zip.partial") for g in groups]
    for group, path, remote in uploads:
        if group != "source" and (path.stat().st_size, path.stat().st_mtime_ns) != (inventory[group]["zip_bytes"], inventory[group]["mtime_ns"]):
            raise RuntimeError("ZIP changed since preflight: " + group)
        print(json.dumps({"phase": "upload", "group": group, "bytes": path.stat().st_size}), flush=True)
        batch = f'put -a "{path.resolve().as_posix()}" "{remote}"\n'
        subprocess.run(["sftp", "-q", *ssh_options(), "-b", "-", args.target], input=batch, text=True, check=True)
        if group != "source" and (path.stat().st_size, path.stat().st_mtime_ns) != (inventory[group]["zip_bytes"], inventory[group]["mtime_ns"]):
            raise RuntimeError("ZIP changed during transfer: " + group)
    bootstrap = BOOTSTRAP.replace("tools.normalize_school_dataset", "tools.append_school_keyframes")
    print(json.dumps({"phase": "extract_decode_cpu", "destination": config["destination"]}), flush=True)
    result = remote_python(args.target, bootstrap.replace("CONFIG", repr(config), 1))
    (output / "REPORT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "complete", **result}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-config", type=Path)
    parser.add_argument("--zip-dir", type=Path)
    parser.add_argument("--groups")
    parser.add_argument("--target")
    parser.add_argument("--target-root")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.remote_config:
        print(json.dumps(remote_run(json.loads(args.remote_config.read_text()))), flush=True)
    else:
        if not all([args.zip_dir, args.groups, args.target, args.target_root, args.output_dir]):
            parser.error("Specify zip-dir, groups, target, target-root and output-dir")
        orchestrate(args)


if __name__ == "__main__":
    main()
