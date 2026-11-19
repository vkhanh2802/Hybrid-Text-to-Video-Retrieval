"""Non-destructive public OCR import: retain ZIPs and exact per-frame bytes.

Run on the school CPU after SHA-256 checked transfer into a NEW import root.
No packages, OCR inference, caption writes or changes to the live worker.
"""

from collections import Counter
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

from kis.public_ocr import checked_path, unique_object
from kis.school_ocr import build_snapshot, digest, mappings, validate_record, validate_groups


def zip_records(archive, group, expected):
    """Validate entire archive before callers write any member, including CRCs."""
    records, seen = [], set()
    with zipfile.ZipFile(checked_path(archive)) as z:
        for info in z.infolist():
            p = PurePosixPath(info.filename)
            if (info.is_dir() or p.is_absolute() or ".." in p.parts or "\\" in info.filename
                    or not re.fullmatch(group + r"/" + group + r"_V[0-9]{3}/[0-9]{3,}\.json", info.filename)
                    or stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1
                    or not 0 < info.file_size <= 1024 * 1024):
                raise ValueError("Unsafe OCR ZIP member")
            vid, n = p.parent.name, int(p.stem)
            if p.name != f"{n:03d}.json" or (vid, n) in seen or (vid, n) not in expected:
                raise ValueError("Duplicate/unmapped OCR ZIP member")
            raw = z.read(info)
            validate_record(json.loads(raw, object_pairs_hook=unique_object), vid, n, expected[vid, n])
            records.append((p.as_posix(), raw))
            seen.add((vid, n))
    if seen != set(expected):
        raise ValueError("Incomplete OCR ZIP coverage")
    return records


def write_new(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def import_data(base, original_dataset, archive_hashes, existing_groups=("L02", "L04")):
    base, original_dataset = checked_path(base), checked_path(original_dataset)
    groups = tuple(sorted((*archive_hashes, *existing_groups)))
    validate_groups(groups)
    # Fail without altering original roots or overwriting an earlier import.
    dataset = base / "dataset"
    dataset.mkdir(exist_ok=False)
    report = {"schema": "school_ocr_import_v1", "groups": {}, "archives": archive_hashes,
              "original_dataset": str(original_dataset), "bytes_preserved": True}
    for group in groups:
        expected, hashes = mappings(original_dataset / "mapping", group, allow_frame_aliases=True)
        for vid, expected_hash in hashes.items():
            matches = list((original_dataset / "mapping" / group).rglob(vid + ".csv"))
            if len(matches) != 1 or digest(matches[0]) != expected_hash:
                raise ValueError("Mapping changed during OCR import")
            write_new(dataset / "mapping" / group / vid / (vid + ".csv"), checked_path(matches[0]).read_bytes())
        if group in archive_hashes:
            archive = base / "archives" / (group + ".zip")
            if digest(archive) != archive_hashes[group]:
                raise ValueError("Transferred OCR ZIP checksum mismatch")
            records = zip_records(archive, group, expected)
        else:
            records = []
            for path in sorted((original_dataset / "ocr" / group).rglob("*.json")):
                records.append((path.relative_to(original_dataset / "ocr").as_posix(), checked_path(path).read_bytes()))
            seen = set()
            for name, raw in records:
                p = PurePosixPath(name)
                vid, n = p.parent.name, int(p.stem)
                if name != f"{group}/{vid}/{n:03d}.json" or (vid, n) in seen or (vid, n) not in expected:
                    raise ValueError("Duplicate/noncanonical existing OCR")
                validate_record(json.loads(raw, object_pairs_hook=unique_object), vid, n, expected[vid, n])
                seen.add((vid, n))
            if seen != set(expected):
                raise ValueError("Incomplete existing OCR coverage")
        states = Counter()
        for name, raw in records:
            target = dataset / "ocr" / name
            write_new(target, raw)
            if target.read_bytes() != raw:
                raise ValueError("OCR write verification failed")
            states[json.loads(raw)[0]["status"]] += 1
        report["groups"][group] = {"records": len(records), "status_counts": dict(states),
                                    "source": "transferred_zip" if group in archive_hashes else "existing_server_copy"}
    report["snapshot"] = build_snapshot(dataset, groups, base / "snapshot")
    with (base / "IMPORT_REPORT.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return report
