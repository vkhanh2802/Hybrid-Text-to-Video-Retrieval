"""Transfer explicitly selected validated OCR ZIPs into a fresh import root."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import zipfile

from kis.school_ocr import mappings, validate_groups
from tools.import_school_ocr import zip_records
from tools.transfer_school_data import ROOT, remote_python, sha256_file, ssh_options


GROUPS = ("L01", "L03", "L05", "L06", "L07", "L08", "L09")


def transfer(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    groups = tuple(getattr(args, 'groups', None) or GROUPS)
    existing_groups = tuple(getattr(args, 'existing_groups', ('L02', 'L04')))
    validate_groups(tuple(sorted((*groups, *existing_groups))))
    zip_dir = Path(getattr(args, 'zip_dir', None) or ROOT / 'dataset/ocr')
    hashes = {}
    for group in groups:
        archive = zip_dir / (group + ".zip")
        expected, _ = mappings(Path(getattr(args, 'mapping_root', None) or ROOT / 'dataset/mapping'), group,
                               allow_frame_aliases=True)
        zip_records(archive, group, expected)
        hashes[group] = sha256_file(archive)
    names = ["kis/__init__.py", "kis/paths.py", "kis/ocr.py", "kis/public_ocr.py", "kis/school_ocr.py",
             "tools/__init__.py", "tools/import_school_ocr.py"]
    archive = output / "import-source.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
        for name in names:
            z.write(ROOT / name, name)
    name = "kis-ocr-import-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    state = remote_python(args.target, "import pathlib,json\n"
        f"base=pathlib.Path.home()/{name!r}\nbase.mkdir(mode=0o700)\n"
        "(base/'archives').mkdir()\nprint(json.dumps({'root':str(base)}))")
    remote = state["root"]
    files = [(archive, remote + "/import-source.zip")]
    files += [(zip_dir / (g + ".zip"), remote + "/archives/" + g + ".zip") for g in groups]
    batch = "".join(f'put "{p.resolve().as_posix()}" "{dest}"\n' for p, dest in files)
    subprocess.run(["sftp", "-q", *ssh_options(), "-b", "-", args.target], input=batch, text=True, check=True)
    cfg = {"root": remote, "source_sha256": sha256_file(archive), "members": names,
            "archive_hashes": hashes, "original_dataset": args.dataset, "existing_groups": existing_groups}
    (output / "TRANSFER.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    script = r'''
import pathlib,json,hashlib,zipfile,sys
cfg=CONFIG
base=pathlib.Path(cfg['root']);archive=base/'import-source.zip'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==cfg['source_sha256']
source=base/'source';source.mkdir()
with zipfile.ZipFile(archive) as z:
    assert z.namelist()==cfg['members']
    for name in cfg['members']:
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(z.read(name))
sys.path.insert(0,str(source))
from tools.import_school_ocr import import_data
report=import_data(base,cfg['original_dataset'],cfg['archive_hashes'],existing_groups=cfg['existing_groups'])
print(json.dumps(report))
'''
    result = remote_python(args.target, script.replace("CONFIG", repr(cfg), 1))
    (output / "IMPORT_REPORT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"root": remote, **result}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--zip-dir', type=Path, help='Validated per-group OCR ZIP directory; defaults to dataset/ocr')
    parser.add_argument('--mapping-root', type=Path, help='Explicit unambiguous local mapping tree for preflight')
    parser.add_argument('--groups', nargs='+', help='Exact new ZIP groups')
    parser.add_argument('--existing-groups', nargs='*', default=['L02', 'L04'],
                        help='Explicit existing server groups to retain; pass empty to import ZIPs only')
    transfer(parser.parse_args())
