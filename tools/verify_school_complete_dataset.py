"""Read-only checks for staged L20 OCR + new L21 artifact/image dataset."""
import json
from pathlib import Path
from io import BytesIO
import zipfile

from PIL import Image
from kis.school_ocr import mappings
from tools.benchmark_school_ocr import client


def verify(settings, preparation, mapping_root, output):
    import numpy as np
    worker = client(settings, True)
    info = worker.info
    baseline = client('config/school-runtime.local.json', True).info
    assert info['models'] == baseline['models']
    assert {k: v for k, v in info['runtime'].items() if k != 'ocr_manifest_sha256'} == {
        k: v for k, v in baseline['runtime'].items() if k != 'ocr_manifest_sha256'}
    assert info['runtime']['video_groups'] == [f'L{i:02d}' for i in range(1, 22)]
    assert info['caption_groups'] == info['runtime']['video_groups']
    assert info['caption_source_vectors'] == 161742
    assert info['caption_vectors'] == info['visual_vectors'] == 161722
    assert info['excluded_ambiguous_caption_rows'] == 20
    assert info['ocr']['records'] == 161742
    assert info['ocr']['bound_records'] == 161722
    assert info['ocr']['excluded_ambiguous_keyframes'] == 20
    assert info['ocr']['covered_groups'] == info['runtime']['video_groups']
    assert info['ocr']['missing_groups'] == info['missing_image_groups'] == []
    checks = []
    for group in ('L20', 'L21'):
        mapping, _ = mappings(Path(mapping_root), group, allow_frame_aliases=True)
        with zipfile.ZipFile(Path(preparation) / 'validated_archives' / (group + '.zip')) as z:
            names = sorted(z.namelist())
            name = names[len(names) // 3]
            original = json.loads(z.read(name))[0]
        vid, n = original['video_id'], original['keyframe_n']
        values = [{'video_id': vid, 'keyframe_n': n, 'frame_id': mapping[vid, n]}]
        worker.attach_ocr(values)
        assert values[0]['ocr']['text'] == original['ocr_text']
        assert values[0]['ocr']['status'] == original['status']
        data, mime = worker.read(vid, n)
        with zipfile.ZipFile('dataset/data/' + group + '.zip') as z:
            names = [p for p in z.namelist() if Path(p).parent.name == vid and Path(p).stem.isdigit()
                     and int(Path(p).stem) == n and Path(p).suffix.lower() in {'.jpg', '.jpeg', '.png'}]
            assert len(names) == 1 and data == z.read(names[0])
        with Image.open(BytesIO(data)) as image:
            image.load()
            if group == 'L21':
                encoded = worker.encode_images([image.convert('RGB')])[0]
                vectors = np.load(Path('dataset/clip-features-32/L21') / vid / (vid + '.npy'), allow_pickle=False)
                expected = vectors[n - 1]
                cosine = float(np.dot(encoded, expected) / (np.linalg.norm(encoded) * np.linalg.norm(expected)))
                assert cosine >= .99, 'New L21 features do not match active visual encoder'
                hits = worker.request('/visual-search', {'vector': expected.tolist(), 'top_k': 100})['results']
                assert any(r['video_id'] == vid and r['keyframe_n'] == n for r in hits)
        literal = max(original['ocr_text'].splitlines(), key=len).strip()
        if len(literal) > 120:
            literal = literal[:120].rsplit(' ', 1)[0]
        result = worker.retrieve({'raw_query': 'Cảnh có dòng chữ: ' + literal, 'ocr_text': literal,
                                  'needs_ocr': True, 'query_variants': [literal], 'entities': []}, temporal=False)
        assert result['diagnostics']['ocr']['status'] == 'matched'
        assert result['diagnostics']['ocr']['hits'] > 0
        assert len({(r['video_id'], r['frame_id']) for r in result['candidates'][:100]}) == 100
        assert any(r['video_id'].startswith(group + '_') for r in result['candidates'])
        assert not any((r['video_id'], r['frame_id']) in {tuple(x) for x in info['ambiguous_frame_ids_excluded']}
                       for r in result['candidates'])
        checks.append({'group': group, 'video_id': vid, 'keyframe_n': n, 'OCR_matches_source': True,
                       'image_bytes_match_source': True, 'ocr_hits': result['diagnostics']['ocr']['hits'],
                       'top100_distinct': True, 'alias_frames_excluded': True,
                       **({'new_feature_encoder_cosine': cosine} if group == 'L21' else {})})
    # Exclusion is a retrieval-only quarantine, not deletion or frame-ID repair.
    preserved_alias_images = []
    with zipfile.ZipFile('dataset/data/L21.zip') as z:
        members = {(Path(p).parent.name, int(Path(p).stem)): p for p in z.namelist()
                   if Path(p).stem.isdigit() and Path(p).suffix.lower() in {'.jpg', '.jpeg', '.png'}}
        mapping, _ = mappings(Path(mapping_root), 'L21', allow_frame_aliases=True)
        ambiguous = {tuple(x) for x in info['ambiguous_frame_ids_excluded']}
        for (vid, n), frame in mapping.items():
            if (vid, frame) in ambiguous:
                data, _ = worker.read(vid, n)
                assert data == z.read(members[vid, n])
                preserved_alias_images.append({'video_id': vid, 'keyframe_n': n, 'frame_id': frame})
    assert len(preserved_alias_images) == 20
    report = {'status': 'passed', 'kind': 'CPU_integration_not_quality_benchmark',
              'worker': {k: info[k] for k in ('caption_vectors', 'caption_source_vectors', 'visual_vectors',
                                             'caption_groups', 'missing_image_groups', 'excluded_ambiguous_caption_rows')},
              'ocr': info['ocr'], 'checks': checks, 'preserved_alias_images': preserved_alias_images,
              'models_and_retrieval_settings_unchanged': True, 'active_config_changed': False,
              'web_restarted': False, 'Qwen_run': False, 'DRES': False}
    with (Path(output) / 'STAGED_WORKER_VERIFIED.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('settings', 'preparation', 'mapping-root', 'output'):
        p.add_argument('--' + name, required=True)
    verify(**vars(p.parse_args()))
