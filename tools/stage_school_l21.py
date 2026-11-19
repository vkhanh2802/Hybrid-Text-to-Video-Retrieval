"""Stage canonical new L21 artifacts/images in a fresh composite dataset.

Old data is never overwritten. L01-L20 regular files are hardlinked read-only by
convention into the new dataset; L21 is copied from the user's canonical output.
Do not modify linked files in place. No planner, model inference, web or DRES.
"""
from collections import Counter
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

from kis.caption_schema import normalize_caption_record, validate_unique_caption_records, manifest_data
from kis.config import VISUAL_CONFIG
from kis.objects import validate_object_schema
from tools.append_school_keyframes import preflight
from tools.transfer_school_data import ROOT, CONFIG_FILES, pack, remote_python, sha256_file, ssh_options, REMOTE_VERIFY


def prepare(output):
    import numpy as np
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    mapping_root = output / 'mapping_view'
    # This view deliberately has one CSV per video. Equal legacy duplicates are
    # not deleted; conflicting duplicates still stop the import.
    for group in (f'L{i:02d}' for i in range(1, 22)):
        candidates = {}
        for p in sorted((ROOT / 'dataset/mapping' / group).rglob(group + '_V*.csv')):
            candidates.setdefault(p.stem, []).append(p)
        for vid, paths in candidates.items():
            hashes = {sha256_file(p) for p in paths}
            if len(hashes) != 1:
                raise ValueError('Conflicting mapping copies: ' + vid)
            target = mapping_root / group / (vid + '.csv')
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as f:
                f.write(paths[0].read_bytes())
    files, groups, aliases = [], {}, []
    videos = sorted(p.stem for p in (mapping_root / 'L21').glob('*.csv'))
    if not videos:
        raise ValueError('Missing L21 mapping')
    for vid in videos:
        paths = {}
        for comp, suffix in [('mapping', '.csv'), ('clip-features-32', '.npy'), ('caption_generator', '.json'),
                             ('caption_mapping', '.json'), ('index', '.index')]:
            p = ROOT / 'dataset' / comp / 'L21' / vid / (vid + suffix)
            if not p.is_file():
                raise FileNotFoundError('Canonical new artifact missing: ' + str(p))
            paths[comp] = p
            files.append(p)
        meta_path = paths['index'].with_suffix('.meta.json')
        files.append(meta_path)
        with paths['mapping'].open(encoding='utf-8', newline='') as f:
            rows = list(csv.DictReader(f))
        mapping = {int(r['n']): int(r['frame_idx']) for r in rows}
        if len(mapping) != len(rows) or sorted(mapping) != list(range(1, len(rows) + 1)):
            raise ValueError('Invalid L21 keyframe numbers')
        counts = Counter(mapping.values())
        aliases.extend((vid, frame) for frame, count in counts.items() if count > 1)
        vectors = np.load(paths['clip-features-32'], allow_pickle=False)
        if vectors.dtype != np.float32 or vectors.shape != (len(rows), int(VISUAL_CONFIG['dimension'])) or not np.isfinite(vectors).all():
            raise ValueError('Invalid canonical L21 features')
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-3):
            raise ValueError('L21 features not normalized')
        captions = json.loads(paths['caption_generator'].read_text(encoding='utf-8'))
        normalized = [normalize_caption_record(r, vid, mapping) for r in captions]
        validate_unique_caption_records(normalized, vid)
        if {r['keyframe_n'] for r in normalized} != set(mapping):
            raise ValueError('Incomplete L21 captions')
        if any(r.get('caption_generation', {}).get('backend') != 'internvl_dense_v3' for r in captions):
            raise ValueError('Active caption backend mismatch')
        join = json.loads(paths['caption_mapping'].read_text(encoding='utf-8'))
        if len(join) != len(captions) or any(any(a[k] != b[k] for k in ('video_id', 'frame_id', 'keyframe_n', 'image_path')) for a, b in zip(captions, join)):
            raise ValueError('L21 caption index identity mismatch')
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        if meta != manifest_data(vid, paths['caption_generator'], paths['caption_mapping'], paths['index'], len(rows)):
            raise ValueError('L21 caption/index checksum mismatch')
        for n in mapping:
            obj = ROOT / f'dataset/objects/L21/{vid}/{n:03d}.json'
            validate_object_schema(json.loads(obj.read_text(encoding='utf-8')))
            files.append(obj)
        groups[vid] = len(rows)
    image_inventory = preflight(ROOT / 'dataset/data/L21.zip', mapping_root, 'L21')
    # Validate every caption against the exact source JPEG to detect mismatched
    # export versions before touching the school dataset.
    with zipfile.ZipFile(ROOT / 'dataset/data/L21.zip') as z:
        by_key = {(Path(n).parent.name, int(Path(n).stem)): n for n in z.namelist()
                  if Path(n).stem.isdigit() and Path(n).suffix.lower() in {'.jpg', '.jpeg', '.png'}}
        import hashlib
        for vid in videos:
            captions = json.loads((ROOT / f'dataset/caption_generator/L21/{vid}/{vid}.json').read_text(encoding='utf-8'))
            for r in captions:
                if hashlib.sha256(z.read(by_key[vid, r['keyframe_n']])).hexdigest() != r['image_sha256']:
                    raise ValueError('Caption image source mismatch')
    files += [ROOT / 'config' / n for n in CONFIG_FILES]
    records = [{'path': p.relative_to(ROOT).as_posix(), 'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in sorted(files)]
    plan = {'root': str(ROOT), 'files': records, 'file_count': len(records),
            'bytes': sum(r['size'] for r in records), 'components': dict(Counter(Path(r['path']).parts[1] for r in records))}
    package = pack(plan, output / 'L21-artifacts.tar.gz')
    report = {'status': 'passed', 'videos': len(videos), 'keyframes': sum(groups.values()),
              'canonical_features_selected': True, 'legacy_flat_files_modified': False,
              'ambiguous_frame_identities': aliases, 'images': image_inventory, 'artifact_package': package,
              'mapping_root': str(mapping_root)}
    with (output / 'PREFLIGHT.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    return report


def transfer(output, previous_dataset, target):
    output = Path(output)
    report = json.loads((output / 'PREFLIGHT.json').read_text(encoding='utf-8'))
    source_zip = output / 'source.zip'
    members = [p.relative_to(ROOT).as_posix() for p in (ROOT / 'kis').glob('*.py') if p.name != 'local_images.py']
    members += ['tools/' + n for n in ['__init__.py', 'normalize_school_dataset.py', 'transfer_school_data.py']]
    with zipfile.ZipFile(source_zip, 'x', zipfile.ZIP_DEFLATED) as z:
        for name in members:
            z.write(ROOT / name, name)
    name = 'kis-l21-data-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prepared = remote_python(target, "import pathlib,json,shutil\nbase=pathlib.Path.home()/" + repr(name) +
        "\nassert shutil.disk_usage(base.parent).free>6*1024**3\nbase.mkdir(mode=0o700)\n(base/'archives').mkdir()\n" +
        "\nfor n in ['artifacts.tar.gz.partial','source.zip.partial','archives/L21.zip.partial']:(base/n).open('xb').close()\nprint(json.dumps({'destination':str(base)}))")
    cfg = {**prepared, **report['artifact_package'], 'source_zip_sha256': sha256_file(source_zip),
           'members': members, 'image_sha256': report['images']['sha256'], 'previous_dataset': previous_dataset}
    with (output / 'REMOTE_STATE.json').open('x', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2)
    for p, dest in [(output / 'L21-artifacts.tar.gz', 'artifacts.tar.gz.partial'),
                    (source_zip, 'source.zip.partial'), (ROOT / 'dataset/data/L21.zip', 'archives/L21.zip.partial')]:
        subprocess.run(['sftp', '-q', *ssh_options(), '-b', '-', target],
                       input=f'put -a "{p.resolve().as_posix()}" "{cfg["destination"]}/{dest}"\n', text=True, check=True)
    receipt = remote_python(target, REMOTE_VERIFY.replace('CONFIG', repr(cfg), 1))
    with (output / 'ARTIFACT_TRANSFER_VERIFIED.json').open('x', encoding='utf-8') as f:
        json.dump(receipt, f, indent=2)
    code = r'''
import pathlib,json,hashlib,zipfile,os,sys
cfg=CONFIG;base=pathlib.Path(cfg['destination']);source=base/'source';source.mkdir()
assert hashlib.sha256((base/'source.zip.partial').read_bytes()).hexdigest()==cfg['source_zip_sha256']
with zipfile.ZipFile(base/'source.zip.partial') as z:
 assert set(z.namelist())==set(cfg['members'])
 for n in cfg['members']:
  p=source/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(n))
sys.path.insert(0,str(source))
from tools.normalize_school_dataset import extract_group,validate_images
from tools.transfer_school_data import sha256_file
dataset=base/'dataset';old=pathlib.Path(cfg['previous_dataset']);links=0
for comp in ['mapping','clip-features-32','caption_generator','caption_mapping','index','objects','data']:
 (dataset/comp).mkdir(exist_ok=True)
 for i in range(1,21):
  origin=old/comp/f'L{i:02d}';dest=dataset/comp/origin.name
  if not origin.exists():raise ValueError('Missing previous group '+str(origin))
  for folder,dirs,files in os.walk(origin,followlinks=False):
   assert not pathlib.Path(folder).is_symlink()
   target=dest/pathlib.Path(folder).relative_to(origin);target.mkdir(parents=True,exist_ok=True)
   for n in files:
    inp=pathlib.Path(folder)/n;assert inp.is_file() and not inp.is_symlink()
    os.link(inp,target/n);links+=1
assert sha256_file(base/'archives/L21.zip.partial')==cfg['image_sha256']
(base/'archives/L21.zip.partial').rename(base/'archives/L21.zip')
images=extract_group(base/'archives/L21.zip',dataset/'data','L21')
coverage=validate_images(dataset,'L21',images['files']);assert coverage['status']=='passed'
receipt={'status':'passed','dataset':str(dataset),'image_coverage':coverage,'unchanged_L01_L20_regular_file_hardlinks':links,'old_files_overwritten':False,'L21_images':images['images'],'L21_identical_copy_files':images['identical_zip_duplicates']}
(base/'STAGE_VERIFIED.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
'''
    script = code.replace('CONFIG', repr(cfg), 1)
    wrapper = "import subprocess,os\np=subprocess.run([os.path.expanduser('~/KISBranche/.venv/bin/python'),'-u','-c'," + repr(script) + "],capture_output=True,text=True)\nassert p.returncode==0,p.stderr[-3000:]\nprint(p.stdout)"
    result = remote_python(target, wrapper)
    with (output / 'STAGE_VERIFIED.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--transfer', action='store_true')
    parser.add_argument('--target', required=True, help='user@school-host')
    parser.add_argument('--previous-dataset', required=True)
    args = parser.parse_args()
    if not args.transfer:
        print(json.dumps(prepare(args.output)), flush=True)
    else:
        transfer(args.output, args.previous_dataset, args.target)
