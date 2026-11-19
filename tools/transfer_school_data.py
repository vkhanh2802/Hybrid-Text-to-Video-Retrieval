"""Copy non-image KIS artifacts to a NEW SSH directory; never delete sources.

Only explicit dataset artifact directories and three public config files are
eligible. No Drive access, keyframes, source ZIPs, credentials, or model cache.
Uses the existing OpenSSH agent/config; passwords are never accepted by this tool.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import time


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIRS = ("caption_generator", "caption_mapping", "clip-features-32",
                 "index", "mapping", "objects", "ocr", "preparation")
CONFIG_FILES = ("models.json", "pipeline.json", "object_labels.json")
EXCLUDED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".avi", ".mkv",
                     ".zip", ".7z", ".pem", ".key", ".pfx", ".ovpn", ".pyc"}
SECRET_NAME = re.compile(r"credential|password|secret|service.?account|\.local\.", re.I)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eligible(relative):
    relative = PurePosixPath(relative)
    parts = relative.parts
    if relative.is_absolute() or ".." in parts or any(part.startswith(".") for part in parts):
        return False
    if relative.suffix.lower() in EXCLUDED_SUFFIXES or SECRET_NAME.search(relative.name):
        return False
    return ((len(parts) >= 3 and parts[0] == "dataset" and parts[1] in ARTIFACT_DIRS)
            or (len(parts) == 2 and parts[0] == "config" and parts[1] in CONFIG_FILES))


def inventory(root):
    root = Path(root).resolve()
    records, counts, excluded = [], Counter(), Counter()
    for component in ARTIFACT_DIRS:
        directory = root / "dataset" / component
        if not directory.is_dir() or directory.is_symlink():
            continue
        for base, dirs, files in os.walk(directory, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not (Path(base) / d).is_symlink())
            for name in sorted(files):
                path = Path(base) / name
                relative = path.relative_to(root).as_posix()
                if path.is_symlink() or not eligible(relative):
                    excluded[component] += 1
                    continue
                stat = path.stat()
                records.append({"path": relative, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
                counts[component] += 1
    for name in CONFIG_FILES:
        path = root / "config" / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Public configuration missing: {path}")
        stat = path.stat()
        records.append({"path": f"config/{name}", "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    if not counts:
        raise ValueError("No dataset artifacts to transfer")
    return {"schema_version": 1, "root": str(root), "files": records,
            "file_count": len(records), "bytes": sum(row["size"] for row in records),
            "components": dict(counts), "excluded_in_components": dict(excluded),
            "excluded_roots": ["dataset/data", "external Drive/keyframes", "config/*.local.json", "credentials"]}


class HashReader:
    def __init__(self, stream):
        self.stream, self.digest = stream, hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def pack(plan, output):
    root, output = Path(plan["root"]), Path(output)
    if output.exists() or output.with_suffix(output.suffix + ".partial").exists():
        raise FileExistsError("Refusing to overwrite an existing archive or partial archive")
    if shutil.disk_usage(output.parent).free < plan["bytes"] * 1.15 + 512 * 1024 ** 2:
        raise RuntimeError("Not enough local staging space")
    partial = output.with_suffix(output.suffix + ".partial")
    manifest = {key: value for key, value in plan.items() if key != "root"}
    manifest["files"] = []
    with tarfile.open(partial, "w:gz", compresslevel=1) as archive:
        for index, row in enumerate(plan["files"], 1):
            if not eligible(row["path"]):
                raise ValueError("Archive plan includes a non-allowlisted path")
            source = root / row["path"]
            if source.is_symlink() or root.resolve() not in source.resolve().parents:
                raise ValueError("Source escaped dataset root")
            before = source.stat()
            if (before.st_size, before.st_mtime_ns) != (row["size"], row["mtime_ns"]):
                raise RuntimeError(f"Source changed since inventory: {row['path']}")
            member = tarfile.TarInfo(row["path"])
            member.size, member.mode, member.mtime = before.st_size, 0o600, int(before.st_mtime)
            with source.open("rb") as stream:
                reader = HashReader(stream)
                archive.addfile(member, reader)
            after = source.stat()
            if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
                raise RuntimeError(f"Source changed while packing: {row['path']}")
            manifest["files"].append(dict(row, sha256=reader.digest.hexdigest()))
            if index % 25000 == 0:
                print(json.dumps({"phase": "packing", "files": index, "total": plan["file_count"]}), flush=True)
        data = json.dumps(manifest, ensure_ascii=True).encode("utf-8")
        member = tarfile.TarInfo("TRANSFER_MANIFEST.json")
        member.size, member.mode = len(data), 0o600
        archive.addfile(member, io.BytesIO(data))
    partial.rename(output)
    return {"archive": str(output), "archive_bytes": output.stat().st_size,
            "archive_sha256": sha256_file(output), "source_files": plan["file_count"],
            "source_bytes": plan["bytes"]}


def ssh_options():
    return ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=3"]


def remote_python(target, code):
    result = subprocess.run(["ssh", *ssh_options(), target, "python3", "-"],
        input=code, text=True, encoding="utf-8", capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"SSH operation failed: {result.stderr[-2000:]}")
    return json.loads(result.stdout)


REMOTE_VERIFY = '''
import hashlib,json,pathlib,shutil,tarfile
cfg=CONFIG
dest=pathlib.Path(cfg['destination'])
part=dest/'artifacts.tar.gz.partial'
archive=dest/'artifacts.tar.gz'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()
if not part.is_file() or digest(part)!=cfg['archive_sha256']:
    raise RuntimeError('Uploaded archive checksum mismatch; partial retained, no sources deleted')
if archive.exists():
    raise RuntimeError('Refusing to overwrite remote archive')
if shutil.disk_usage(dest).free < cfg['source_bytes']+2*1024**3:
    raise RuntimeError('Insufficient disk space for extraction')
part.rename(archive)
seen=set()
with tarfile.open(archive,'r:gz') as tar:
    for member in tar:
        rel=pathlib.PurePosixPath(member.name)
        if (not member.isfile() or rel.is_absolute() or '..' in rel.parts
                or member.name in seen or any(p.startswith('.') for p in rel.parts)
                or (member.name!='TRANSFER_MANIFEST.json' and rel.parts[0] not in {'dataset','config'})):
            raise RuntimeError('Unsafe or duplicate archive member')
        seen.add(member.name)
        output=dest.joinpath(*rel.parts)
        output.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        with output.open('xb') as out,tar.extractfile(member) as source:
            shutil.copyfileobj(source,out,1024*1024)
        output.chmod(0o600)
manifest=json.loads((dest/'TRANSFER_MANIFEST.json').read_text())
expected={row['path'] for row in manifest['files']}
if seen!=expected|{'TRANSFER_MANIFEST.json'}:
    raise RuntimeError('Archive membership does not match manifest')
total=0
for row in manifest['files']:
    p=dest/row['path']
    if p.stat().st_size!=row['size'] or digest(p)!=row['sha256']:
        raise RuntimeError('Extracted data checksum mismatch: '+row['path'])
    total+=row['size']
receipt=dict(status='verified',destination=str(dest),files=len(expected),bytes=total,
             archive_sha256=cfg['archive_sha256'],components=manifest['components'],
             keyframes_included=False,credentials_included=False,sources_deleted=False)
with (dest/'TRANSFER_VERIFIED.json').open('x') as stream:
    json.dump(receipt,stream,indent=2)
print(json.dumps(receipt))
'''


REMOTE_PREPARE = '''
import json,os,pathlib
cfg=CONFIG
p=pathlib.Path(cfg['destination'])
marker=p/'UPLOAD_PACKAGE.json'
part=p/'artifacts.tar.gz.partial'
expected={k:cfg[k] for k in ('archive_sha256','archive_bytes','source_files','source_bytes')}
if cfg['resume']:
    if (not p.is_dir() or p.is_symlink()
            or (hasattr(os,'getuid') and p.stat().st_uid!=os.getuid())):
        raise RuntimeError('Resume destination must be an existing owned directory')
    if {q.name for q in p.iterdir()}-{'UPLOAD_PACKAGE.json','artifacts.tar.gz.partial'}:
        raise RuntimeError('Resume directory contains other data; refusing to overwrite')
    if marker.exists():
        if marker.is_symlink() or json.loads(marker.read_text())!=expected:
            raise RuntimeError('Resume package does not match original upload')
    elif part.exists():
        raise RuntimeError('Unmarked partial file; refusing to resume unknown data')
else:
    p.mkdir(mode=0o700,exist_ok=False)
if not marker.exists():
    with marker.open('x') as stream:
        json.dump(expected,stream)
    marker.chmod(0o600)
if part.exists():
    if part.is_symlink() or not part.is_file() or part.stat().st_size>cfg['archive_bytes']:
        raise RuntimeError('Invalid partial upload')
else:
    with part.open('xb'):
        pass
    part.chmod(0o600)
print(json.dumps(dict(destination=str(p),resume_bytes=part.stat().st_size)))
'''


def upload(target, package, directory_name, *, resume=False):
    if not re.fullmatch(r"[a-zA-Z0-9_.-]+@[a-zA-Z0-9_.-]+", target):
        raise ValueError("Specify user@host without extra SSH arguments")
    if not re.fullmatch(r"kis-data-[a-zA-Z0-9_-]+", directory_name):
        raise ValueError("Remote directory must be a new kis-data-* directory")
    preflight = remote_python(target, "import json,pathlib,shutil\np=pathlib.Path.home()\n"
        "print(json.dumps(dict(home=str(p),free_bytes=shutil.disk_usage(p).free)))")
    required = package["source_bytes"] + package["archive_bytes"] + 2 * 1024 ** 3
    if preflight["free_bytes"] < required:
        raise RuntimeError("Remote filesystem does not have enough free space")
    destination = str(PurePosixPath(preflight["home"]) / directory_name)
    config = dict(package, destination=destination, resume=resume)
    prepared = remote_python(target, REMOTE_PREPARE.replace("CONFIG", repr(config), 1))
    archive = Path(package["archive"])
    if archive.stat().st_size != package["archive_bytes"] or sha256_file(archive) != package["archive_sha256"]:
        raise RuntimeError("Local archive changed; refusing upload")
    print(json.dumps({"phase": "uploading", **prepared, "archive_bytes": package["archive_bytes"]}), flush=True)
    local = Path(package["archive"]).resolve().as_posix()
    batch = f'put -a "{local}" "{destination}/artifacts.tar.gz.partial"\n'
    subprocess.run(["sftp", "-q", *ssh_options(), "-b", "-", target], input=batch,
                   text=True, encoding="utf-8", check=True)
    print(json.dumps({"phase": "remote_checksum_and_extract", "destination": destination}), flush=True)
    code = REMOTE_VERIFY.replace("CONFIG", repr(dict(package, destination=destination)), 1)
    return remote_python(target, code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", help="Existing authenticated SSH user@host")
    parser.add_argument("--transfer", action="store_true", help="Actually pack, upload and verify")
    parser.add_argument("--resume-package", type=Path, help="Reuse an existing PACKAGE.json without repacking")
    parser.add_argument("--remote-name", help="Existing owned kis-data-* directory for an explicit resume")
    args = parser.parse_args()
    started = time.monotonic()
    if args.resume_package:
        if not args.transfer or not args.remote_name:
            parser.error("Resume requires --transfer and --remote-name")
        if args.output_dir.resolve() != args.resume_package.resolve().parent:
            parser.error("Resume output directory must be the existing package directory")
        if (args.output_dir / "RECEIPT.json").exists():
            parser.error("A completed receipt already exists; refusing to repeat transfer")
        package = json.loads(args.resume_package.read_text(encoding="utf-8"))
        print(json.dumps({"phase": "resume_existing_package", **package}), flush=True)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        plan = inventory(args.root)
        (args.output_dir / "PLAN.json").write_text(json.dumps(plan), encoding="utf-8")
        summary = {k: v for k, v in plan.items() if k not in {"files", "root"}}
        print(json.dumps({"phase": "inventory", **summary}), flush=True)
        if not args.transfer:
            return
        package = pack(plan, args.output_dir / "artifacts.tar.gz")
        (args.output_dir / "PACKAGE.json").write_text(json.dumps(package, indent=2), encoding="utf-8")
        print(json.dumps({"phase": "packed", **package}), flush=True)
    if not args.target:
        parser.error("--transfer requires --target")
    name = args.remote_name or "kis-data-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    receipt = upload(args.target, package, name, resume=bool(args.resume_package))
    receipt["elapsed_seconds"] = round(time.monotonic() - started, 2)
    (args.output_dir / "RECEIPT.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "complete", **receipt}), flush=True)


if __name__ == "__main__":
    main()
