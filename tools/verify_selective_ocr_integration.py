"""Live integration smoke using frozen planner outputs, then one GPU Qwen call.

Not a new accuracy ablation: no live planner call and no benchmark GT sent.
Does not switch configuration or restart any service.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
from time import perf_counter


def save(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def verify(output):
    output = Path(output)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', KIS_VLM_BATCH_SIZE='12',
                      OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    import numpy
    import torch
    from tools.benchmark_school_ocr import client
    from kis.ocr_routing import select_prompt_ocr
    from kis.ranking import get_search_profile
    from kis.solve_kis import rerank_candidates_with_vlm
    from kis.web_app import LazyVLMReranker, serialize_search_results, require_frame_count
    from tools.benchmark_user_five_ocr import load_mapping, validate_identities, ROOT
    worker = client('config/school-ocr-selective.local.json', True)
    settings = json.loads(Path('config/school-ocr-selective.local.json').read_text(encoding='utf-8'))
    assert worker.info['runtime']['ocr_manifest_sha256'] == settings['ocr_manifest_sha256']
    assert {'L01', 'L02'}.issubset(worker.info['ocr']['covered_groups'])
    assert worker.info['ocr']['records'] >= 14955
    mapping, _ = load_mapping(ROOT / 'dataset/mapping', worker.groups)
    reports, packets = [], {}
    for i in range(1, 6):
        name = f'q{i}'
        # Project exactly the structured query, excluding GT and prior results.
        q = json.loads((ROOT / 'benchmark_results/user_five_full_pipeline_20261116_run02' /
                        f'{name}_PLANNER.json').read_text(encoding='utf-8'))['structured']
        start = perf_counter()
        remote = worker.retrieve(q)
        elapsed = perf_counter() - start
        rows = remote['candidates']
        validate_identities(rows, mapping)
        assert len(rows) >= 100 and all('ocr' not in row for row in rows[20:])
        chosen = [select_prompt_ocr(r.get('ocr'), q) for r in rows[:20]]
        assert all(x is None or len(x['text']) <= 384 for x in chosen)
        if name == 'q2':
            assert all(x is None for x in chosen)
        report = {'id': name, 'rpc_seconds': elapsed, 'candidates': len(rows),
                  'ocr': remote['diagnostics']['ocr'],
                  'prompt_candidates': sum(x is not None for x in chosen),
                  'prompt_characters': [len(x['text']) if x else 0 for x in chosen]}
        save(output / f'{name}_SMOKE_RETRIEVAL.json', {'structured': q, 'remote': remote, 'checks': report})
        reports.append(report)
        packets[name] = (q, rows)
        print(json.dumps(report, ensure_ascii=True), flush=True)
    # q5 exercises the newly repaired explicit-text route and selective prompt.
    q, rows = packets['q5']
    lazy = LazyVLMReranker()
    start = perf_counter()
    model = lazy._load()
    torch.cuda.synchronize()
    cold = perf_counter() - start
    assert model.batch_size == 12 and model.device.type == 'cuda'
    profile = get_search_profile(worker.info['runtime']['retrieval_profile'])
    start = perf_counter()
    candidates = deepcopy(rows)
    ranked = rerank_candidates_with_vlm(candidates, q['raw_query'], lazy, worker,
        top_k=20, vlm_weight=.5, verified_first=profile.policy.verified_first,
        image_source=worker, ocr_query=q)[:100]
    elapsed = perf_counter() - start
    serialized = serialize_search_results(ranked, worker, image_resolver=worker.reference)
    require_frame_count(serialized, 100)
    validate_identities(ranked, mapping, 100)
    assert not worker.transport_failed
    stats = lazy.last_batch_stats
    assert stats.get('requested_batch_size') == 12 and stats.get('inputs', 0) > 0
    assert all(a['status'] == 'ok' for a in stats['attempts'])
    assert any(r.get('vlm_ocr_used') is True for r in candidates[:20])
    assert any(r.get('vlm_status') == 'reranked' for r in candidates[:20])
    result = {'status': 'passed', 'kind': 'integration_smoke_not_live_planner_ablation',
              'retrieval_checks': reports, 'gpu_query': 'q5', 'model_load_seconds': cold,
              'gpu_rerank_including_images_seconds': elapsed, 'batch': stats,
              'prompt_candidates_actually_scored': sum(r.get('vlm_ocr_used') is True for r in candidates[:20]),
              'missing_images': sum(r.get('vlm_status') == 'missing_image' for r in candidates[:20]),
              'invalid_or_error': sum(r.get('vlm_status') in ('invalid_score', 'error') for r in candidates[:20]),
              'distinct_mapped_100': True, 'DRES': False, 'web_restarted': False}
    save(output / 'GPU_SMOKE_RESULTS.json', serialized)
    save(output / 'INTEGRATION_VERIFICATION.json', result)
    print(json.dumps(result, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    verify(p.parse_args().output)
