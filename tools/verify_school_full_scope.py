"""Read-only staged scope verification before editing active pipeline/settings."""
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import zipfile

from PIL import Image

from kis.config import load_runtime_config, MODEL_MANIFEST
from kis.ocr import OCRConfig
from kis.remote_worker import RemoteWorker
from kis.school_ocr import mappings
from tools.benchmark_school_ocr import client


def run(settings, output):
    cfg = json.loads(Path(settings).read_text(encoding='utf-8'))
    groups = tuple(f'L{i:02d}' for i in range(1, 22))
    assert tuple(cfg['video_groups']) == groups
    old = client('config/school-runtime.local.json', True)
    config = replace(load_runtime_config(environ={}), video_groups=groups,
                     ocr_format=cfg['ocr_format'], ocr_root=Path(cfg['ocr_root']),
                     ocr_manifest_sha256=cfg['ocr_manifest_sha256'],
                     ocr=OCRConfig(mode=cfg['ocr_mode'], weight=cfg['ocr_weight']))
    worker = RemoteWorker(cfg['url'], cfg['token'], groups)
    worker.connect(config, MODEL_MANIFEST)
    assert worker.info['models'] == old.info['models']
    assert {k: v for k, v in worker.info['runtime'].items() if k != 'video_groups'} == {
        k: v for k, v in old.info['runtime'].items() if k != 'video_groups'}
    assert worker.info['visual_vectors'] == old.info['visual_vectors'] + 22946
    assert worker.info['caption_vectors'] == old.info['caption_vectors'] + 22946
    assert worker.info['caption_groups'] == list(groups[:-1])
    assert worker.info['ocr']['covered_groups'] == list(groups[:-2])
    assert worker.info['ocr']['records'] == 145750
    samples, originals = [], []
    for group in ('L04', 'L05', 'L06'):
        mapping, _ = mappings(Path('dataset/mapping'), group)
        with zipfile.ZipFile(Path('benchmark_results/ocr_l01_l19_preflight_20261117_run01/validated_archives') / (group + '.zip')) as z:
            names = sorted(z.namelist())
            for index in (len(names) // 4, 3 * len(names) // 4):
                r = json.loads(z.read(names[index]))[0]
                samples.append({'video_id': r['video_id'], 'frame_id': mapping[r['video_id'], r['keyframe_n']],
                                'keyframe_n': r['keyframe_n']})
                originals.append(r)
    worker.attach_ocr(samples)
    for row, original in zip(samples, originals):
        assert row['ocr']['text'] == original['ocr_text']
        assert row['ocr']['status'] == original['status']
    image_checks, retrieval_checks = [], []
    for group in ('L04', 'L05', 'L06'):
        candidates = [r for r in samples if r['video_id'].startswith(group + '_') and r['ocr']['status'] == 'ok']
        assert candidates
        row = candidates[0]
        vid, n = row['video_id'], row['keyframe_n']
        data, mime = worker.read(vid, n)
        with Image.open(BytesIO(data)) as image:
            image.load()
            assert min(image.size) > 0
        with zipfile.ZipFile('dataset/data/' + group + '.zip') as z:
            names = [name for name in z.namelist() if Path(name).parent.name == vid
                     and Path(name).stem.isdigit() and int(Path(name).stem) == n
                     and Path(name).suffix.lower() in ('.jpg', '.jpeg', '.png')]
            assert len(names) == 1 and data == z.read(names[0])
        image_checks.append({'group': group, 'video_id': vid, 'keyframe_n': n, 'original_bytes_match': True, 'mime': mime})
        line = max(row['ocr']['text'].splitlines(), key=len).strip()
        literal = line if len(line) <= 120 else line[:120].rsplit(' ', 1)[0]
        assert literal
        result = worker.retrieve({'raw_query': 'Cảnh có dòng chữ: ' + literal, 'query_variants': [literal],
                                  'needs_ocr': True, 'ocr_text': literal, 'entities': []}, temporal=False)
        diag = result['diagnostics']['ocr']
        assert diag['status'] == 'matched' and diag['hits'] > 0
        assert len(result['candidates']) >= 100
        in_group = sum(r['video_id'].startswith(group + '_') for r in result['candidates'])
        assert in_group > 0
        retrieval_checks.append({'group': group, 'status': 'passed', 'ocr_hits': diag['hits'],
                                 'candidate_pool': len(result['candidates']), 'in_group_candidates': in_group,
                                 'distinct_top100': len({(r['video_id'], r['frame_id']) for r in result['candidates'][:100]}) == 100})
    report = {'status': 'passed', 'kind': 'CPU_functional_scope_smoke_not_accuracy_benchmark',
              'scope': list(groups), 'caption_groups': worker.info['caption_groups'],
              'caption_vectors': worker.info['caption_vectors'], 'visual_vectors': worker.info['visual_vectors'],
              'ocr_groups': worker.info['ocr']['covered_groups'], 'ocr_records': worker.info['ocr']['records'],
              'missing_image_groups': worker.info['missing_image_groups'],
              'exact_ocr_samples': len(samples), 'image_checks': image_checks,
              'retrieval_checks': retrieval_checks, 'active_config_changed': False,
              'planner_called': False, 'Qwen_run': False, 'DRES': False, 'web_restart': False}
    with (Path(output) / 'STAGED_VERIFICATION.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', required=True)
    parser.add_argument('--output', required=True)
    run(**vars(parser.parse_args()))
