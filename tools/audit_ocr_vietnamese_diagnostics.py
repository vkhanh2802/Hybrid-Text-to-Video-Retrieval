"""Read-only diagnostics/public/source-image reconciliation; never loads models."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

from kis.public_ocr import unique_object
from tools.audit_ocr_vietnamese_pair import sha, write
from tools.build_kaggle_ocr_vietnamese_notebook import worker_source


def bounded_members(archive):
    members = {}
    for info in archive.infolist():
        p = PurePosixPath(info.filename)
        if (info.filename in members or p.is_absolute() or '..' in p.parts
                or '\\' in info.filename or stat.S_ISLNK(info.external_attr >> 16)
                or info.flag_bits & 1 or info.is_dir()
                or not 0 < info.file_size <= 16 * 1024 * 1024):
            raise ValueError('Unsafe/duplicate/oversize ZIP member')
        members[info.filename] = info
    return members


def public_projection(record, threshold):
    lines = record['lines'] if record['status'] != 'error' else []
    text = '\n'.join(line['text'] for line in lines if line['text'].strip()
                     and line['confidence'] >= threshold and not line['truncated'])
    video = record['video_id']
    return dict(video_id=video, frame_id=None, keyframe_n=record['keyframe_n'],
                image_path=f'data/{video.split("_V")[0]}/{video}/{PurePosixPath(record["relative_path"]).name}',
                ocr_text=text, retrieval_text=text,
                status='error' if record['status'] == 'error' else 'ok' if text else 'empty')


def run(output, previous):
    output, previous = Path(output), Path(previous)
    output.mkdir(parents=True, exist_ok=False)
    paths = [Path('dataset/ocr/ocr_vietnamese_diagnostics.zip'), Path('dataset/ocr/ocr_vietnamese.zip')]
    hashes = {str(p): sha(p) for p in paths}
    sample = {r['member'] for r in json.loads((previous / 'SAMPLE_PAIRS.json').read_text(encoding='utf-8'))}
    counters, widths, selected, source_checks, records = Counter(), {}, [], [], []
    with zipfile.ZipFile(paths[0]) as technical, zipfile.ZipFile(paths[1]) as public:
        members = bounded_members(technical)
        public_members = bounded_members(public)
        if technical.testzip() is not None or public.testzip() is not None:
            raise ValueError('CRC failure')
        def read(name):
            return json.loads(technical.read(members[name]), object_pairs_hook=unique_object)
        profile, state = read('profile.json'), read('run.json')
        profile_hash = hashlib.sha256(json.dumps(profile, ensure_ascii=False, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
        assert state['profile_sha256'] == profile_hash
        assert state['complete'] is True and state['phase'] == 'finished'
        assert profile['worker_sha256'] == hashlib.sha256(worker_source()[3].encode('utf-8')).hexdigest()
        threshold = profile['options']['public_min_confidence']
        inventory = {i['id']: i for i in state['inventory']}
        assert len(inventory) == state['total'] == len(state['inventory'])
        checkpoints = {name for name in members if name.startswith('checkpoints/')}
        assert len(members) == len(checkpoints) + 2 == state['total'] + 2
        seen, public_seen = set(), set()
        for name in sorted(checkpoints):
            assert re.fullmatch(r'checkpoints/L01/L01_V\d{3}/\d{3,}\.json', name)
            record = read(name)
            ident = record['id']
            assert ident not in seen
            seen.add(ident)
            assert all(record[k] == v for k, v in inventory[ident].items())
            assert record['profile_sha256'] == profile_hash
            assert ident == f'{record["video_id"]}:{record["keyframe_n"]}'
            member = name.removeprefix('checkpoints/')
            assert member in public_members
            public_seen.add(member)
            actual = json.loads(public.read(public_members[member]), object_pairs_hook=unique_object)
            assert actual == [public_projection(record, threshold)]
            counters['records'] += 1
            counters['public_projection_exact'] += 1
            counters['record_status_' + record['status']] += 1
            record_excluded = False
            for line in record['lines']:
                confidence = line['confidence']
                assert math.isfinite(confidence) and 0 <= confidence <= 1
                included = bool(line['text'].strip()) and confidence >= threshold and not line['truncated']
                assert line['public_included'] is included
                assert len(line['polygon_padded']) == 4
                assert all(math.isfinite(v) for pt in line['polygon_padded'] for v in pt)
                counters['lines'] += 1
                counters['public_lines' if included else 'excluded_lines'] += 1
                counters['token_cap_truncated_lines'] += bool(line['truncated'])
                counters['width_capped_lines'] += bool(line['width_capped'])
                counters['low_confidence_lines'] += confidence < threshold
                counters['blank_lines'] += not bool(line['text'].strip())
                record_excluded |= not included
                bucket = '<=512' if line['recognizer_size'][0] <= 512 else '>512'
                counts = widths.setdefault(bucket, Counter())
                counts['lines'] += 1
                counts['included'] += included
                counts['token_cap_truncated'] += line['truncated']
                counts['width_capped'] += line['width_capped']
            counters['records_with_excluded_lines'] += record_excluded
            if member in sample:
                selected.append(dict(member=member, diagnostic=record))
            records.append((member, record))
        assert seen == set(inventory) and public_seen == set(public_members)
    # Hash the actual local JPEG bytes for every checkpoint, not just the sample.
    with zipfile.ZipFile('dataset/data/L01.zip') as images:
        lookup = {}
        for info in images.infolist():
            if info.is_dir() or not info.filename.endswith('.jpg'):
                continue
            p = PurePosixPath(info.filename)
            key = (p.parent.name, p.stem)
            assert key not in lookup
            lookup[key] = info
        for member, record in records:
            key = (record['video_id'], PurePosixPath(record['relative_path']).stem)
            info = lookup[key]
            assert 0 < info.file_size <= 16 * 1024 * 1024
            actual = hashlib.sha256(images.read(info)).hexdigest()
            match = actual == record['source_sha256']
            counters['source_image_hash_matches' if match else 'source_image_hash_mismatches'] += 1
            source_checks.append(dict(member=member, image_member=info.filename,
                                      actual_sha256=actual, checkpoint_sha256=record['source_sha256'], match=match))
    assert hashes == {str(p): sha(p) for p in paths}
    engine = profile['engine']
    report = dict(archives_sha256=hashes, source_archives_unchanged=True, crc='pass',
        profile_sha256=profile_hash, profile_hash_consistent=True, worker_matches_local_v21=True,
        run={k: v for k, v in state.items() if k != 'inventory'}, models=engine['models'],
        recognition_weights_sha256=engine['model_files_sha256']['recognition.pth'],
        recognition_weights_match_expected=engine['model_files_sha256']['recognition.pth'] ==
            '380512193a8b6cbf6fad80deacdc9b6939d10d473d199892fc6408d13775ea59',
        options=profile['options'], stats=dict(counters), width_buckets={k: dict(v) for k,v in widths.items()},
        caveats=['Hashes establish byte consistency with supplied diagnostics, not independent runtime attestation.',
                 'truncated=false means decoder emitted EOS, not that all image text was recognized.',
                 'Confidence is uncalibrated; no whole-corpus OCR accuracy inferred.',
                 'Public record schema and production evidence verification status remain unchanged.'])
    write(output / 'PROFILE.json', profile)
    write(output / 'REPORT.json', report)
    write(output / 'SAMPLE_DIAGNOSTICS.json', selected)
    write(output / 'SOURCE_IMAGE_CHECKS.json', source_checks)
    print(json.dumps(report, ensure_ascii=True))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--previous', default='benchmark_results/ocr_vietnamese_pair_20261115_run01')
    args = parser.parse_args()
    run(args.output, args.previous)
