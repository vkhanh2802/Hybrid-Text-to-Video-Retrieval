"""Build KIS artifacts from existing JPEG keyframes + authoritative CSVs only.

Never opens source videos, resamples frames, or guesses frame_idx/FPS/time.
Input bytes are copied unchanged to a writable, hash-checked checkpoint tree.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil
import zipfile

from kis.artifact_io import atomic_write_json
from kis.caption_schema import file_sha256
from tools.prepare_video_data import VideoDatasetBuilder, configure_device


def inspect_inputs(keyframes_root, mapping_root, video_ids=None):
    from PIL import Image

    root, mapping_root = Path(keyframes_root).resolve(), Path(mapping_root).resolve()
    if not root.is_dir() or not mapping_root.is_dir():
        raise FileNotFoundError('Set KEYFRAMES_ROOT and MAPPING_ROOT to existing input folders')
    discovered, ignored_duplicates = {}, {}
    folders = [root, *(p for p in root.rglob('*') if p.is_dir())]
    for folder in folders:
        if not re.fullmatch(r'L\d{2}_V\d{3}', folder.name):
            continue
        images, ignored = {}, []
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            # Download/Explorer copies are not new frame identities. Ignore
            # only numbered-copy names, optionally prefixed by this video ID.
            if re.fullmatch(rf'(?:{re.escape(folder.name)}\s+)?[0-9]+\s*\([1-9][0-9]*\)', path.stem):
                ignored.append(path.name)
                continue
            if not re.fullmatch(r'[0-9]+', path.stem) or int(path.stem) < 1:
                raise ValueError(f'Expected numbered keyframes, not frame_idx filenames: {path}')
            if path.suffix.lower() == '.png':
                raise ValueError(f'Expected JPEG keyframes (no automatic lossy conversion): {path}')
            number = int(path.stem)
            if number in images:
                raise ValueError(f'Duplicate keyframe number: {folder}/{number}')
            images[number] = path
        if images:
            if folder.name in discovered:
                raise ValueError(f'Duplicate video image directory: {folder.name}')
            discovered[folder.name] = images
            ignored_duplicates[folder.name] = ignored
    if video_ids is not None:
        if not video_ids or len(set(video_ids)) != len(video_ids):
            raise ValueError('VIDEO_IDS must be None or a nonempty list of unique IDs')
        missing = set(video_ids) - discovered.keys()
        if missing:
            raise FileNotFoundError(f'Missing keyframe folders: {sorted(missing)}')
        discovered = {v: discovered[v] for v in sorted(video_ids)}
    if not discovered:
        raise FileNotFoundError(f'No Lxx_Vxxx/001.jpg keyframes found under {root}; unzip input archives first')
    plans = {}
    for vid, images in sorted(discovered.items()):
        matches = sorted(mapping_root.rglob(f'{vid}.csv'))
        if len(matches) != 1:
            raise ValueError(f'{vid}: need exactly one original mapping CSV; found {len(matches)}. '
                             'Do not infer frame_idx/timestamps from image numbers.')
        mapping = matches[0]
        with mapping.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ['n', 'pts_time', 'fps', 'frame_idx'] or mapping.read_bytes().startswith(b'\xef\xbb\xbf'):
                raise ValueError(f'{mapping}: required CSV header n,pts_time,fps,frame_idx in this order (UTF-8 without BOM)')
            rows = []
            for raw in reader:
                if any(not re.fullmatch(r'[0-9]+', raw.get(k) or '') for k in ('n', 'frame_idx')):
                    raise ValueError(f'Invalid integer mapping: {mapping}')
                rows.append({'n': int(raw['n']), 'frame_idx': int(raw['frame_idx']),
                             'pts_time': float(raw['pts_time']), 'fps': float(raw['fps'])})
        if not rows or set(images) != {r['n'] for r in rows}:
            raise ValueError(f'{vid}: image/mapping coverage differs; do not renumber missing frames')
        previous_frame, previous_time = -1, -1.0
        frame_to_ns = {}
        for expected, row in enumerate(rows, 1):
            if (row['n'] != expected or row['frame_idx'] < 0 or row['frame_idx'] < previous_frame
                    or not math.isfinite(row['pts_time']) or row['pts_time'] < 0 or row['pts_time'] < previous_time
                    or not math.isfinite(row['fps']) or row['fps'] <= 0):
                raise ValueError(f'{vid}: mapping must have contiguous n, nondecreasing frame_idx/time and positive FPS')
            frame_to_ns.setdefault(row['frame_idx'], []).append(row['n'])
            previous_frame, previous_time = row['frame_idx'], row['pts_time']
        frame_idx_aliases = [{'frame_idx': frame, 'keyframe_ns': ns}
                             for frame, ns in frame_to_ns.items() if len(ns) > 1]
        hashes = {}
        for n, path in images.items():
            with Image.open(path) as image:
                if image.format != 'JPEG':
                    raise ValueError(f'JPEG extension but not JPEG pixels: {path}')
                image.verify()
            hashes[f'{n:03d}.jpg'] = file_sha256(path)
        plans[vid] = {'images': images, 'mapping': mapping, 'rows': rows,
                      'mapping_sha256': file_sha256(mapping), 'image_sha256': hashes,
                      'ignored_duplicate_files': ignored_duplicates[vid],
                      'frame_idx_aliases': frame_idx_aliases}
        if frame_idx_aliases:
            print(f'[{vid}] WARNING: frame_idx aliases preserved: {frame_idx_aliases}. '
                  'Use (video_id, keyframe_n) for identity; original CSV is unchanged.', flush=True)
        if ignored_duplicates[vid]:
            print(f'[{vid}] Ignored {len(ignored_duplicates[vid])} duplicate-copy keyframes: '
                  f'{ignored_duplicates[vid][:5]}', flush=True)
    return plans


def copy_checked(source, target, expected):
    if file_sha256(source) != expected:
        raise ValueError(f'Input changed after preflight: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file() or file_sha256(target) != expected:
            raise FileExistsError(f'Refusing to overwrite different artifact: {target}')
        return
    with source.open('rb') as inp, target.open('xb') as out:
        shutil.copyfileobj(inp, out)
    if file_sha256(target) != expected:
        raise ValueError(f'Copy hash mismatch: {target}')


def register_keyframes(builder, vid, plan):
    paths = builder.artifact_paths(vid)
    expected = {'kind': 'existing_keyframes', 'mapping_sha256': plan['mapping_sha256'],
                'images': plan['image_sha256']}
    state = builder._state(vid)
    if state and state.get('source') != expected:
        raise ValueError(f'{vid}: checkpoint belongs to different inputs; use a new OUTPUT_ROOT')
    if not state and any(paths[k].exists() for k in ('features', 'objects', 'captions', 'caption_mapping', 'index', 'manifest')):
        raise FileExistsError(f'{vid}: generated artifacts have no trusted input checkpoint; use a new OUTPUT_ROOT')
    if paths['images'].exists() and {p.name for p in paths['images'].iterdir()} - set(expected['images']):
        raise ValueError(f'{vid}: unexpected files in output image directory')
    copy_checked(plan['mapping'], paths['mapping'], plan['mapping_sha256'])
    for n, source in plan['images'].items():
        name = f'{n:03d}.jpg'
        copy_checked(source, paths['images'] / name, plan['image_sha256'][name])
    if not state:
        state = {'schema_version': 1, 'video_id': vid, 'source': expected,
                  'keyframes': {'config': {'mode': 'import_existing_no_video_decode'},
                               'mapping_sha256': plan['mapping_sha256'], 'images': plan['image_sha256']}}
    # Enable aliases only on imported, byte-checked inputs. This does not relax
    # validation for checkpoints produced by video extraction.
    state['keyframes']['config']['allow_frame_idx_aliases'] = True
    state['keyframes']['frame_idx_aliases'] = plan['frame_idx_aliases']
    builder._save_state(vid, state)
    builder.mapping_rows(vid)


def archive_artifacts(builder, video_ids, status, include_images=False):
    suffix = '' if status == 'complete' else '-partial'
    archive = builder.root.parent / f'{builder.root.name}-from-keyframes{suffix}.zip'
    manifest = builder.root / 'keyframes_run.json'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as target:
        for vid in video_ids:
            for kind, path in builder.artifact_paths(vid).items():
                if kind == 'images' and not include_images:
                    continue
                for file in sorted(path.rglob('*')) if path.is_dir() else [path]:
                    if file.is_file():
                        target.write(file, file.relative_to(builder.root).as_posix())
            backups = builder.root / 'preparation/caption_backups' / vid
            if backups.exists():
                for file in sorted(backups.rglob('*.json')):
                    target.write(file, file.relative_to(builder.root).as_posix())
        if manifest.exists():
            target.write(manifest, manifest.name)
    return archive


def run(keyframes_root, mapping_root, output_root, video_ids=None, visual_batch=8,
        object_batch=8, caption_batch=16, caption_backend='internvl', caption_4bit=None,
        restart_captions=False, include_images=False, check_only=False):
    output = Path(output_root).resolve()
    for source in (Path('/kaggle/input').resolve(), Path(keyframes_root).resolve(), Path(mapping_root).resolve()):
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('OUTPUT_ROOT must be separate from all read-only input folders')
    if min(visual_batch, object_batch, caption_batch) < 1:
        raise ValueError('Batch sizes must be positive')
    plans = inspect_inputs(keyframes_root, mapping_root, video_ids)
    print(json.dumps({'preflight': 'passed', 'videos': {vid: len(p['rows']) for vid, p in plans.items()},
                      'keyframe_extraction': False, 'check_only': check_only}), flush=True)
    if check_only:
        return plans
    print(configure_device('cuda', threads=4), flush=True)
    from tools.kaggle_prepare import release_models
    from tools.kaggle_keyframes_caption import restart_caption_runtime_if_changed
    from tools.structured_caption_generator import resolve_caption_backend_runtime
    from kis.structured_caption import caption_generation_config
    from tools.audit_artifact_formats import audit_video_formats

    builder = VideoDatasetBuilder(output, device='cuda')
    runtime = resolve_caption_backend_runtime(caption_backend, 'cuda', caption_4bit)
    generation = caption_generation_config(caption_backend, runtime['use_4bit'], runtime['compute_dtype'])
    status = {'status': 'running', 'input_kind': 'existing_keyframes', 'video_extraction': False,
              'video_ids': list(plans), 'caption_generation': generation,
               'archive_includes_images': include_images,
               'identity_policy': 'video_id,keyframe_n',
               'frame_idx_aliases': {vid: p['frame_idx_aliases'] for vid, p in plans.items() if p['frame_idx_aliases']},
              'stages': ['import_keyframes_mapping', 'visual_features', 'objects', 'captions', 'caption_indexes']}
    atomic_write_json(output / 'keyframes_run.json', status)
    try:
        for vid, plan in plans.items():
            register_keyframes(builder, vid, plan)
            restart_caption_runtime_if_changed(builder, vid, generation, force=restart_captions)
        builder.encode_visual_features(plans, batch_size=visual_batch)
        release_models()
        builder.generate_objects(plans, batch_size=object_batch)
        release_models()
        builder.generate_captions(plans, batch_size=caption_batch, backend=caption_backend, use_4bit=runtime['use_4bit'])
        release_models()
        builder.build_caption_indexes(plans, batch_size=32)
        release_models()
        for vid in plans:
            print(builder.validate_video(vid), flush=True)
            audit = audit_video_formats(output, vid)
            if any(r['status'] != 'OK' for r in audit):
                raise ValueError(audit)
    except Exception as exc:
        status.update(status='partial', error=f'{type(exc).__name__}: {str(exc)[:1000]}')
        atomic_write_json(output / 'keyframes_run.json', status)
        try:
            print(f"Partial ZIP: {archive_artifacts(builder, plans, 'partial', include_images)}", flush=True)
        except Exception as archive_error:
            print(f'Could not archive partial output: {archive_error}; checkpoints remain in {output}', flush=True)
        raise
    status['status'] = 'complete'
    atomic_write_json(output / 'keyframes_run.json', status)
    archive = archive_artifacts(builder, plans, 'complete', include_images)
    print(f'Download: {archive}', flush=True)
    return archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keyframes-root', type=Path, required=True)
    parser.add_argument('--mapping-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('/kaggle/working/kis_from_keyframes'))
    parser.add_argument('--video-id', action='append', dest='video_ids')
    parser.add_argument('--visual-batch', type=int, default=8)
    parser.add_argument('--object-batch', type=int, default=8)
    parser.add_argument('--caption-batch', type=int, default=16)
    parser.add_argument('--caption-backend', choices=['internvl', 'qwen'], default='internvl')
    parser.add_argument('--caption-4bit', action='store_true', default=None)
    parser.add_argument('--restart-captions', action='store_true')
    parser.add_argument('--include-images', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    run(**vars(parser.parse_args()))
