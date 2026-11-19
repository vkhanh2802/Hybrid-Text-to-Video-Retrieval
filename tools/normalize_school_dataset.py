"""Non-destructive canonical dataset copy and L02-L08 image import on school CPU."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import time
import zipfile

from tools.transfer_school_data import ROOT, ARTIFACT_DIRS, CONFIG_FILES, remote_python, sha256_file, ssh_options


VIDEO = re.compile(r"L\d+_V\d+")


def artifact_destination(relative):
    p = PurePosixPath(relative)
    if not p.parts or p.is_absolute() or ".." in p.parts or "\\" in relative:
        raise ValueError("Unsafe artifact path")
    if p.parts[0] == "config" and len(p.parts) == 2 and p.name in CONFIG_FILES:
        return p
    if len(p.parts) < 3 or p.parts[:1] != ("dataset",) or p.parts[1] not in ARTIFACT_DIRS:
        raise ValueError("Artifact outside allowlisted dataset components")
    ids = {part for part in p.parts[2:-1] if VIDEO.fullmatch(part)}
    match = re.match(r"^(L\d+_V\d+)\.", p.name)
    if match:
        ids.add(match[1])
    if len(ids) != 1:
        raise ValueError("Artifact does not have one unambiguous video identity")
    vid = ids.pop()
    return PurePosixPath("dataset", p.parts[1], vid.split("_")[0], vid, p.name)


def copy_checked(source, destination, expected_hash, expected_size):
    source, destination = Path(source), Path(destination)
    if source.is_symlink() or source.stat().st_size != expected_size:
        raise ValueError("Source identity/size changed")
    digest = hashlib.sha256()
    with source.open("rb") as src, destination.open("xb") as dst:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(chunk)
            dst.write(chunk)
    if digest.hexdigest() != expected_hash:
        raise ValueError("Copied artifact checksum mismatch")
    # Re-read destination; do not change permissions or contents of the source.
    if destination.stat().st_size != expected_size or sha256_file(destination) != expected_hash:
        raise ValueError("Destination checksum mismatch")


def normalize_artifacts(source_root, destination):
    source_root, destination = Path(source_root), Path(destination)
    if (destination / "dataset").exists() or (destination / "config").exists():
        raise FileExistsError("Canonical dataset/config destination must be new")
    manifest = json.loads((source_root / "TRANSFER_MANIFEST.json").read_text(encoding="utf-8"))
    planned, seen = [], set()
    for row in manifest["files"]:
        target = artifact_destination(row["path"])
        if target.as_posix() in seen:
            raise ValueError("Multiple source files map to the same canonical artifact")
        seen.add(target.as_posix())
        planned.append((row, target))
    required = sum(row["size"] for row, _ in planned)
    if shutil.disk_usage(destination).free < required + 2 * 1024 ** 3:
        raise RuntimeError("Insufficient disk space for non-destructive normalization")
    parents, records, counts = set(), [], Counter()
    for row, relative in planned:
        output = destination.joinpath(*relative.parts)
        if output.parent not in parents:
            output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            parents.add(output.parent)
        copy_checked(source_root / row["path"], output, row["sha256"], row["size"])
        records.append({"source": row["path"], "path": relative.as_posix(), "size": row["size"], "sha256": row["sha256"]})
        counts[relative.parts[1] if relative.parts[0] == "dataset" else "config"] += 1
    report = {"status": "verified", "layout": "component/group/video/file", "files": len(records),
              "bytes": required, "components": dict(counts), "sources_deleted": False, "records": records}
    (destination / "LAYOUT_MANIFEST.json").write_text(json.dumps(report), encoding="utf-8")
    return {k: v for k, v in report.items() if k != "records"}


def image_relative(name, group):
    if "\\" in name or ":" in name or "\0" in name:
        raise ValueError("Unsafe ZIP member")
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("Unsafe ZIP member path")
    videos = [part for part in p.parts if VIDEO.fullmatch(part)]
    image = re.fullmatch(r"(\d+)(?:\s*\(\d+\))?(\.(?:jpg|jpeg|png))", p.name, re.I)
    if (len(videos) != 1 or videos[0].split("_")[0] != group
            or image is None or int(image[1]) < 1):
        raise ValueError("ZIP member is not a numbered image in the selected group")
    return PurePosixPath(group, videos[0], image[1] + image[2])


def extract_group(archive, data_root, group):
    data_root = Path(data_root)
    if (data_root / group).exists():
        raise FileExistsError("Image group already exists; refusing overwrite")
    with zipfile.ZipFile(archive) as z:
        planned, identities, duplicates = [], {}, []
        # Prefer the actual numeric filename over browser duplicate names like 020(1).jpg.
        for member in sorted(z.infolist(), key=lambda m: bool(re.search(r"\(\d+\)", PurePosixPath(m.filename).stem))):
            if member.is_dir():
                continue
            if stat.S_ISLNK(member.external_attr >> 16) or member.flag_bits & 1:
                raise ValueError("Symlink/encrypted keyframe member")
            relative = image_relative(member.filename, group)
            if not 0 < member.file_size <= 25 * 1024 ** 2:
                raise ValueError("Invalid keyframe size")
            key = (relative.parts[1], int(Path(relative.name).stem))
            if re.search(r"\(\d+\)", PurePosixPath(member.filename).stem) and key not in identities:
                raise ValueError("Duplicate-suffix image has no primary numbered image")
            if key in identities:
                previous = identities[key]
                # Never silently choose between different images with one identity.
                if member.file_size != previous.file_size or hashlib.sha256(z.read(member)).digest() != hashlib.sha256(z.read(previous)).digest():
                    raise ValueError("Conflicting duplicate keyframe contents")
                duplicates.append({"member": member.filename, "same_as": previous.filename})
                continue
            identities[key] = member
            planned.append((member, relative))
        total = sum(member.file_size for member, _ in planned)
        if not planned or total > 30 * 1024 ** 3:
            raise ValueError("Empty/oversized keyframe group")
        if shutil.disk_usage(data_root.parent).free < total + 1024 ** 3:
            raise RuntimeError("Insufficient disk space for image extraction")
        (data_root / group).mkdir(parents=True, mode=0o700, exist_ok=False)
        records = []
        for member, relative in planned:
            output = data_root.joinpath(*relative.parts)
            output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            digest = hashlib.sha256()
            with z.open(member) as src, output.open("xb") as dst:
                for chunk in iter(lambda: src.read(1024 * 1024), b""):
                    dst.write(chunk)
                    digest.update(chunk)
            if output.stat().st_size != member.file_size:
                raise ValueError("Extracted image size mismatch")
            records.append({"path": relative.as_posix(), "bytes": member.file_size,
                            "sha256": digest.hexdigest(), "zip_member": member.filename})
    return {"group": group, "images": len(records), "videos": len({k[0] for k in identities}),
            "bytes": total, "crc_checked": True, "identical_zip_duplicates": duplicates, "files": records}


def validate_images(dataset, group, records):
    from PIL import Image
    dataset = Path(dataset)
    expected, actual = {}, {}
    for path in (dataset / "mapping" / group).rglob(f"{group}_V*.csv"):
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                key = (path.stem, int(row["n"]))
                if key in expected:
                    raise ValueError("Duplicate image number in authoritative mapping")
                expected[key] = int(row["frame_idx"])
    decoded, errors = 0, []
    for row in records:
        relative = PurePosixPath(row["path"])
        actual[(relative.parts[1], int(Path(relative.name).stem))] = relative.as_posix()
        try:
            with Image.open(dataset / "data" / relative) as image:
                image.load()
                if image.width < 1 or image.height < 1:
                    raise ValueError("Invalid image dimensions")
            decoded += 1
        except Exception as exc:
            errors.append({"path": relative.as_posix(), "error": type(exc).__name__})
    missing, extra = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
    return {"status": "passed" if expected and not missing and not extra and not errors else "incomplete",
            "expected_images": len(expected), "actual_images": len(actual), "decoded_images": decoded,
            "missing": missing, "extra": extra, "decode_errors": errors}


def remote_run(config):
    base, source_root = Path(config["destination"]), Path(config["artifact_source"])
    reports = base / "reports"
    reports.mkdir(mode=0o700)
    layout = normalize_artifacts(source_root, base)
    dataset = base / "dataset"
    # Keep L01 in the shared canonical data tree without retransferring its ZIP.
    previous = Path(config["l01_root"])
    old_manifest = json.loads((previous / "KEYFRAMES_MANIFEST.json").read_text())
    l01_records = old_manifest["files"]
    for row in l01_records:
        output = dataset / "data" / row["path"]
        output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        copy_checked(previous / "images" / row["path"], output, row["sha256"], row["bytes"])
    (reports / "L01_images.json").write_text(json.dumps(validate_images(dataset, "L01", l01_records)), encoding="utf-8")
    group_reports = {}
    for group in config["groups"]:
        archive = base / "archives" / (group + ".zip")
        partial = archive.with_suffix(".zip.partial")
        if sha256_file(partial) != config["hashes"][group]:
            raise ValueError("Uploaded ZIP checksum mismatch: " + group)
        partial.rename(archive)
        manifest = extract_group(archive, dataset / "data", group)
        coverage = validate_images(dataset, group, manifest["files"])
        (reports / (group + "_images.json")).write_text(json.dumps(dict(manifest, coverage=coverage)), encoding="utf-8")
        group_reports[group] = {k: v for k, v in manifest.items() if k != "files"}
        group_reports[group]["coverage"] = coverage
    (base / "source" / "dataset").symlink_to(dataset, target_is_directory=True)
    gallery = base / "gallery"
    gallery.mkdir()
    cards = []
    for group in config["groups"]:
        if group_reports[group]["coverage"]["status"] != "passed":
            continue
        log = reports / (group + "_cpu.log")
        output = reports / group
        snippet = ("from tools.school_l01_pilot import run_cpu,QUERIES; "
                   f"run_cpu({str(dataset)!r},{str(dataset / 'data')!r},{str(output)!r},"
                   f"groups=({group!r},),queries=(QUERIES[0],),repetitions=1)")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", FAISS_OPT_LEVEL="generic",
                   PYTHONPATH=str(base / "source"))
        with log.open("x") as stream:
            child = subprocess.run([config["python"], "-u", "-c", snippet], cwd=base / "source",
                                   env=env, stdout=stream, stderr=subprocess.STDOUT)
        if child.returncode:
            group_reports[group]["retrieval"] = {"status": "failed", "log_tail": log.read_text()[-2000:]}
        else:
            report = json.loads((output / "CPU_REPORT.json").read_text())
            group_reports[group]["retrieval"] = {"status": report["status"], "runs": [
                {k: r[k] for k in ["count", "unique_count", "retrieval_seconds", "images_decode_thumbnail_seconds"]}
                for r in report["runs"]], "ambiguous_frame_ids_excluded": report["ambiguous_frame_ids_excluded"],
                "peak_rss_kib": report["peak_rss_kib"]}
            (gallery / group).symlink_to(output / "gallery", target_is_directory=True)
            cards.append(f'<li><a href="{group}/index.html">{group}: 100 frame, ảnh thật</a></li>')
    page = '<!doctype html><html lang="vi"><meta charset="utf-8"><title>Ảnh CPU trường L02–L08</title><h1>Dataset chuẩn hóa — CPU retrieval</h1><p>Pilot từng nhóm, không Groq/Qwen/Milvus. Không phải benchmark accuracy.</p><ul>' + ''.join(cards) + '</ul></html>'
    (gallery / "index.html").write_text(page, encoding="utf-8")
    status = "passed" if all(r["coverage"]["status"] == "passed" and r.get("retrieval", {}).get("status") == "passed" for r in group_reports.values()) else "partial"
    result = {"status": status, "destination": str(base), "layout": layout, "groups": group_reports,
              "sources_deleted": False, "old_l01_retained": previous.is_dir(),
              "limitations": ["per-group CPU smoke only", "fixed query, no Groq/Qwen", "NPY exact cosine instead of Milvus"]}
    (base / "IMPORT_VERIFIED.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


BOOTSTRAP = '''
import json,pathlib,zipfile,subprocess,os,hashlib,stat
cfg=CONFIG
base=pathlib.Path(cfg['destination'])
archive=base/'source.zip.partial'
if hashlib.sha256(archive.read_bytes()).hexdigest()!=cfg['source_sha256']: raise RuntimeError('Source checksum mismatch')
archive.rename(base/'source.zip')
source=base/'source'
source.mkdir(mode=0o700)
with zipfile.ZipFile(base/'source.zip') as z:
    if set(z.namelist())!=set(cfg['source_members']): raise RuntimeError('Unexpected source members')
    for i in z.infolist():
        p=pathlib.PurePosixPath(i.filename)
        if p.is_absolute() or '..' in p.parts or stat.S_ISLNK(i.external_attr>>16): raise RuntimeError('Unsafe source')
        out=source.joinpath(*p.parts)
        out.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        with out.open('xb') as stream: stream.write(z.read(i))
(base/'IMPORT_CONFIG.json').write_text(json.dumps(cfg))
env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
with (base/'import.log').open('x') as stream:
    r=subprocess.run([cfg['python'],'-u','-m','tools.normalize_school_dataset','--remote-config',str(base/'IMPORT_CONFIG.json')],cwd=source,env=env,stdout=stream,stderr=subprocess.STDOUT)
if r.returncode: raise RuntimeError('Import failed; originals retained. Log tail: '+(base/'import.log').read_text()[-4000:])
print((base/'IMPORT_VERIFIED.json').read_text())
'''


def orchestrate(args):
    groups = tuple(args.groups.split(","))
    if not groups or any(not re.fullmatch(r"L0[2-8]", g) for g in groups) or len(set(groups)) != len(groups):
        raise ValueError("Select unique groups L02-L08 explicitly")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    hashes, sizes, stamps = {}, {}, {}
    for group in groups:
        path = Path(args.zip_dir) / (group + ".zip")
        with zipfile.ZipFile(path) as z:
            for member in z.infolist():
                if not member.is_dir():
                    image_relative(member.filename, group)
            sizes[group] = {"zip_bytes": path.stat().st_size, "unpacked_bytes": sum(i.file_size for i in z.infolist())}
        stamps[group] = (path.stat().st_size, path.stat().st_mtime_ns)
        hashes[group] = sha256_file(path)
    sources = [p for p in (ROOT / "kis").glob("*.py") if p.name != "local_images.py"]
    sources += [ROOT / "tools" / n for n in ["__init__.py", "normalize_school_dataset.py", "school_l01_pilot.py", "transfer_school_data.py"]]
    sources += [ROOT / "config" / n for n in CONFIG_FILES]
    source_zip = output / "source.zip"
    with zipfile.ZipFile(source_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sources:
            z.write(p, p.relative_to(ROOT).as_posix())
    name = "kis-normalized-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prepared = remote_python(args.target, "import json,pathlib,shutil\nh=pathlib.Path.home()\n"
        f"assert shutil.disk_usage(h).free>{sum(v['zip_bytes'] + v['unpacked_bytes'] for v in sizes.values()) + 10 * 1024 ** 3}, 'Insufficient disk'\n"
        f"p=h/{name!r}\np.mkdir(mode=0o700)\n(p/'archives').mkdir(mode=0o700)\n"
        f"groups={groups!r}\n"
        "for g in groups:\n    (p/'archives'/(g+'.zip.partial')).open('xb').close()\n"
        "(p/'source.zip.partial').open('xb').close()\n"
        "print(json.dumps(dict(destination=str(p),python=str(h/'KISBranche/.venv/bin/python'))))")
    config = dict(prepared, groups=groups, hashes=hashes, sizes=sizes, artifact_source=args.artifact_source,
                  l01_root=args.l01_root, source_sha256=sha256_file(source_zip),
                  source_members=[p.relative_to(ROOT).as_posix() for p in sources])
    (output / "STATE.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    for group, path, remote in [("source", source_zip, "source.zip.partial")] + [
            (g, Path(args.zip_dir) / (g + ".zip"), "archives/" + g + ".zip.partial") for g in groups]:
        if group != "source" and (path.stat().st_size, path.stat().st_mtime_ns) != stamps[group]:
            raise RuntimeError("ZIP changed after preflight: " + group)
        print(json.dumps({"phase": "upload", "group": group, "bytes": path.stat().st_size, "destination": config["destination"]}), flush=True)
        batch = f'put -a "{path.resolve().as_posix()}" "{config["destination"]}/{remote}"\n'
        subprocess.run(["sftp", "-q", *ssh_options(), "-b", "-", args.target], input=batch, text=True, check=True)
        if group != "source" and (path.stat().st_size, path.stat().st_mtime_ns) != stamps[group]:
            raise RuntimeError("ZIP changed during transfer: " + group)
    print(json.dumps({"phase": "normalize_extract_decode_and_cpu_test", "destination": config["destination"]}), flush=True)
    result = remote_python(args.target, BOOTSTRAP.replace("CONFIG", repr(config), 1))
    (output / "REPORT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "complete", **result}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-config", type=Path)
    parser.add_argument("--zip-dir", type=Path)
    parser.add_argument("--groups")
    parser.add_argument("--target")
    parser.add_argument("--artifact-source")
    parser.add_argument("--l01-root")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.remote_config:
        print(json.dumps(remote_run(json.loads(args.remote_config.read_text()))), flush=True)
    else:
        if not all([args.zip_dir, args.groups, args.target, args.artifact_source, args.l01_root, args.output_dir]):
            parser.error("Specify zip-dir, groups, target, artifact-source, l01-root and output-dir")
        orchestrate(args)


if __name__ == "__main__":
    main()
