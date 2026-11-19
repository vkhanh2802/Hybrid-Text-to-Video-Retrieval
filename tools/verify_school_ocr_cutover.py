"""Verify new snapshot evidence against validated ZIPs, CPU smoke, atomic cutover.

Never restart web, retire workers, invoke planner/Qwen/DRES, or alter the corpus.
"""
import json
import os
from pathlib import Path
import zipfile

from kis.school_ocr import mappings
from tools.benchmark_school_ocr import client
from tools.transfer_school_data import sha256_file


def run(settings, preparation, transfer, output):
    settings, preparation, transfer, output = map(Path, (settings, preparation, transfer, output))
    cfg = json.loads(settings.read_text(encoding='utf-8'))
    preflight = json.loads((preparation / 'PREFLIGHT.json').read_text(encoding='utf-8'))
    imported = json.loads((transfer / 'IMPORT_REPORT.json').read_text(encoding='utf-8'))
    active = Path('config/school-runtime.local.json')
    before = active.read_bytes()
    old = json.loads(before)
    assert cfg['vlm_batch_size'] == old['vlm_batch_size'] == 12
    assert cfg['ocr_weight'] == old['ocr_weight']
    assert cfg['ocr_manifest_sha256'] == imported['snapshot']['manifest_sha256']
    worker = client(str(settings), True)
    old_worker = client(str(active), True)
    assert worker.info['models'] == old_worker.info['models']
    assert {k: v for k, v in worker.info['runtime'].items() if k != 'ocr_manifest_sha256'} == {
        k: v for k, v in old_worker.info['runtime'].items() if k != 'ocr_manifest_sha256'}
    scope = worker.info['runtime']['video_groups']
    expected = [r for r in preflight['groups'] if r['group'] in scope]
    assert worker.info['ocr']['covered_groups'] == [r['group'] for r in expected]
    assert worker.info['ocr']['records'] == sum(r['records'] for r in expected)
    assert imported['snapshot']['records'] == preflight['total_records']
    new_runtime = dict(worker.info['runtime'])
    old_public = json.loads(Path('config/pipeline.json').read_text(encoding='utf-8'))
    assert scope == old_public['video_groups']
    samples, checks = [], []
    by_group = {r['group']: r for r in preflight['groups']}
    for row in expected:
        group = row['group']
        archive = Path(preflight['validated_archives']) / (group + '.zip')
        assert sha256_file(archive) == row['import_archive_sha256']
        assert sha256_file(Path('dataset/ocr') / (group + '.zip')) == row['original_archive_sha256']
        mapping, _ = mappings(Path('dataset/mapping'), group)
        with zipfile.ZipFile(archive) as z:
            names = sorted(z.namelist())
            for index in (len(names) // 4, 3 * len(names) // 4):
                record = json.loads(z.read(names[index]))[0]
                samples.append({'video_id': record['video_id'], 'keyframe_n': record['keyframe_n'],
                                'frame_id': mapping[record['video_id'], record['keyframe_n']]})
                checks.append({'group': group, 'text': record['ocr_text'], 'status': record['status']})
    worker.attach_ocr(samples)
    for sample, check in zip(samples, checks):
        assert sample['ocr']['text'] == check['text']
        assert sample['ocr']['status'] == check['status']
        assert sample['ocr']['build_id'] == cfg['ocr_manifest_sha256']
        assert sample['ocr']['verified'] is False
    # Explicitly preserve missing OCR as unknown for in-scope uncovered groups.
    missing = []
    for group in worker.info['ocr']['missing_groups']:
        from kis.caption_schema import load_frame_mapping
        path = next(iter(sorted((Path('dataset/mapping') / group).rglob(group + '_V*.csv'))))
        vid = path.stem
        n, frame = next(iter(load_frame_mapping(vid, Path('dataset/mapping')).items()))
        values = [{'video_id': vid, 'keyframe_n': n, 'frame_id': frame}]
        worker.attach_ocr(values)
        assert values[0]['ocr']['status'] == 'missing' and values[0]['ocr']['text'] == ''
        missing.append(group)
    # CPU functional smoke only, not a quality benchmark. The literal is drawn
    # from newly imported text solely to check FTS wiring; no GT is supplied.
    recent = next((s for s, c in zip(samples, checks) if c['group'] == 'L18' and len(c['text']) > 40), None)
    smoke = None
    if recent:
        line = max(recent['ocr']['text'].splitlines(), key=len).strip()
        literal = line if len(line) <= 120 else line[:120].rsplit(' ', 1)[0]
        result = worker.retrieve({'raw_query': 'Cảnh có dòng chữ: ' + literal, 'needs_ocr': True,
                                  'ocr_text': literal, 'query_variants': [literal], 'entities': []}, temporal=False)
        assert len(result['candidates']) >= 100
        assert len({(r['video_id'], r['frame_id']) for r in result['candidates'][:100]}) == 100
        diag = result['diagnostics']['ocr']
        assert diag['status'] == 'matched' and diag['hits'] > 0
        smoke = {'status': 'passed', 'kind': 'CPU_functional_not_quality',
                 'ocr_hits': diag['hits'], 'distinct_top100': True}
    else:
        raise ValueError('No usable L18 smoke evidence')
    # No secrets in receipts; preserve prior settings as an explicit backup.
    previous = Path('config/school-runtime.before-l01-l19.local.json')
    with previous.open('xb') as stream:
        stream.write(before)
    new_bytes = settings.read_bytes()
    temp = Path('config/school-runtime.l01-l19.tmp.local.json')
    with temp.open('xb') as stream:
        stream.write(new_bytes)
        stream.flush()
        os.fsync(stream.fileno())
    assert active.read_bytes() == before
    os.replace(temp, active)
    final = client(str(active), True)
    assert final.info['runtime'] == new_runtime
    receipt = {'status': 'completed_and_verified', 'worker_pid': cfg['worker_pid'],
               'remote_port': cfg['remote_port'], 'local_port': cfg['local_port'],
               'snapshot_root': cfg['ocr_root'], 'snapshot_pin': cfg['ocr_manifest_sha256'],
               'imported_groups': [r['group'] for r in preflight['groups']],
               'imported_records': preflight['total_records'],
               'active_groups': worker.info['ocr']['covered_groups'], 'active_records': worker.info['ocr']['records'],
               'stored_out_of_scope': [g for g in by_group if g not in scope],
               'missing_unknown_groups_verified': missing, 'exact_evidence_samples': len(samples),
               'cpu_smoke': smoke, 'original_archives_unchanged': True, 'Qwen_run': False,
               'web_restarted': False, 'DRES_calls': False, 'previous_settings': str(previous)}
    with (output / 'FINAL_VERIFICATION.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('settings', 'preparation', 'transfer', 'output'):
        p.add_argument('--' + name, required=True)
    run(**vars(p.parse_args()))
