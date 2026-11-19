"""Single-arm production benchmark of frozen image-authored queries; no DRES.

Ground truth is joined only after every query has completed. Sparse exact-frame
labels measure source-image refinding, not exhaustive semantic relevance.
"""
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import statistics
from time import perf_counter

from tools.benchmark_user_five_ocr import (
    ROOT, TimedImages, TimedScorer, load_mapping, quality, save, sha,
    validate_identities,
)


def run(queries, selection, mapping_root, output, settings):
    wall_start = perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4',
                      MKL_NUM_THREADS='4', KIS_VLM_BATCH_SIZE='12')
    import numpy
    import torch
    from tools.benchmark_school_ocr import client
    from kis.structure_query import create_query_structurer, create_structured_query
    from kis.llm import load_query_api_config
    from kis.solve_kis import rerank_candidates_with_vlm
    from kis.web_app import LazyVLMReranker, serialize_search_results, require_frame_count
    from kis.ranking import get_search_profile

    assert torch.cuda.is_available(), 'Full benchmark requires the production CUDA GPU'
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    inputs = json.loads(Path(queries).read_text(encoding='utf-8'))['queries']
    assert inputs and len({q['id'] for q in inputs}) == len(inputs)
    assert all(set(q) == {'id', 'query'} for q in inputs)
    worker = client(settings, True)
    assert worker.info['ocr']['missing_groups'] == worker.info['missing_image_groups'] == []
    mapping, mapping_hashes = load_mapping(Path(mapping_root).resolve(), worker.groups)
    profile = get_search_profile(worker.info['runtime']['retrieval_profile'])
    api = load_query_api_config()
    api.require_ready()
    generator = create_query_structurer(api)
    files = [Path(queries), Path(selection), Path(settings), Path(__file__).resolve(),
             ROOT / 'tools/benchmark_user_five_ocr.py', ROOT / 'tools/benchmark_school_ocr.py',
             ROOT / 'config/models.json', ROOT / 'config/pipeline.json',
             *sorted((ROOT / 'kis').glob('*.py'))]
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in (p.resolve() for p in files)}
    hashes.update(mapping_hashes)
    save(output / 'PROTOCOL.json', {
        'queries': inputs, 'worker': worker.info, 'planner': api.public(),
        'gpu': torch.cuda.get_device_name(0), 'repetitions': 1,
        'qwen_budget': 20, 'batch_size': 12, 'backfill_missing_images': False,
        'ocr_prompt_policy': 'production selective', 'temporal': 'production default',
        'source_sha256': hashes, 'DRES': False, 'web_restart': False,
        'ground_truth': 'selected exact keyframe only, evaluation after execution',
        'timing': 'planner + CPU RPC + image fetch/Qwen rerank + serialization; model load separate; audit writes excluded',
        'excludes': ['browser rendering/grid image loads', 'HTTP web API overhead'],
    })
    save(output / 'QUERIES.json', inputs)
    start = perf_counter()
    model = LazyVLMReranker()._load()
    torch.cuda.synchronize()
    load_seconds = perf_counter() - start
    assert model.device.type == 'cuda' and model.batch_size == 12
    save(output / 'MODEL_LOAD.json', {'seconds': load_seconds, 'device': str(model.device), 'batch_size': model.batch_size})
    print(json.dumps({'phase': 'model_loaded', 'seconds': load_seconds}), flush=True)
    completed = []
    for q in inputs:
        start = perf_counter()
        structured = create_structured_query(q['query'], output_path=None, generator=generator)
        planner_seconds = perf_counter() - start
        assert structured['raw_query'] == q['query']
        save(output / (q['id'] + '_PLANNER.json'), {'structured': structured, 'seconds': planner_seconds})
        start = perf_counter()
        remote = worker.retrieve(deepcopy(structured), temporal=None, ocr_region=None)
        rpc_seconds = perf_counter() - start
        validate_identities(remote['candidates'], mapping)
        save(output / (q['id'] + '_RETRIEVAL.json'), {'rpc_seconds': rpc_seconds, **remote})
        print(json.dumps({'phase': 'retrieved', 'id': q['id'], 'planner_seconds': planner_seconds,
                          'rpc_seconds': rpc_seconds, 'pool': len(remote['candidates'])}), flush=True)
        candidates = deepcopy(remote['candidates'])
        images, scorer = TimedImages(worker, output), TimedScorer(model, torch)
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        ranked = rerank_candidates_with_vlm(
            candidates, structured['raw_query'], scorer, worker, top_k=20,
            vlm_weight=0.5, verified_first=profile.policy.verified_first,
            image_source=images, base_score_field='score', ocr_query=structured)[:100]
        rerank_seconds = perf_counter() - start
        audit_seconds = sum(r.get('audit_write_seconds', 0) for r in images.reads)
        start = perf_counter()
        serialized = serialize_search_results(ranked, worker, image_resolver=images.reference)
        require_frame_count(serialized, 100)
        json.dumps(serialized, allow_nan=False)
        serialization_seconds = perf_counter() - start
        validate_identities(ranked, mapping, 100)
        assert not worker.transport_failed
        assert len(images.reads) == 20
        assert [(r['video_id'], r['keyframe_n']) for r in images.reads] == [
            (r['video_id'], r['keyframe_n']) for r in remote['candidates'][:20]]
        assert scorer.calls and all(c['success'] for c in scorer.calls)
        row = {'id': q['id'], 'query': q['query'], 'results': ranked,
               'serialized': serialized,
               'timing': {'planner': planner_seconds, 'retrieval_rpc': rpc_seconds,
                          'cpu_stages': remote['diagnostics'].get('stage_seconds'),
                          'image_fetch': sum(r['seconds'] for r in images.reads),
                          'qwen': sum(c['seconds'] for c in scorer.calls),
                          'rerank_including_images': rerank_seconds - audit_seconds,
                          'serialization': serialization_seconds,
                          'audit_write_excluded': audit_seconds,
                          'logical_e2e': planner_seconds + rpc_seconds + rerank_seconds - audit_seconds + serialization_seconds},
               'image_reads': images.reads, 'qwen_calls': scorer.calls,
               'top20_status': dict(Counter(r['vlm_status'] for r in candidates[:20])),
               'ocr_prompt_candidates': sum(r.get('vlm_ocr_used') is True for r in candidates[:20]),
               'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
               'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        save(output / (q['id'] + '_RESULT.json'), row)
        completed.append((row, remote['candidates']))
        print(json.dumps({'phase': 'completed_query', 'id': q['id'], 'timing': row['timing'],
                          'status': row['top20_status']}), flush=True)
    assert all(sha(ROOT / p) == h for p, h in hashes.items()), 'Inputs/source changed during benchmark'
    # No reference identities, images or frame numbers enter the planner/ranker.
    truth = {q['id']: q for q in json.loads(Path(selection).read_text(encoding='utf-8'))['queries']}
    assert set(truth) == {q['id'] for q in inputs}
    runs = []
    for row, pool in completed:
        gt = truth[row['id']]
        for n in gt['keyframes']:
            assert mapping[gt['video_id'], n] == gt['frame_id']
        runs.append({k: v for k, v in row.items() if k not in {'results', 'serialized'}} | {
            'ground_truth': gt, 'before': quality(pool[:100], gt),
            'pool': quality(pool, gt), 'after': quality(row['results'], gt),
            'video_hit_at_100': int(any(r['video_id'] == gt['video_id'] for r in row['results'])),
            'top1': {k: row['results'][0].get(k) for k in ('video_id', 'keyframe_n', 'frame_id', 'vlm_raw_score', 'vlm_status')}})
    metrics = ('hit_at_1', 'hit_at_5', 'hit_at_20', 'hit_at_100', 'mrr_at_100')
    aggregate = {stage: {key: statistics.mean(r[stage][key] for r in runs) for key in metrics}
                 for stage in ('before', 'after')}
    aggregate['pool_source_hit_rate'] = statistics.mean(r['pool']['best_rank'] is not None for r in runs)
    aggregate['video_hit_at_100'] = statistics.mean(r['video_hit_at_100'] for r in runs)
    aggregate['timing'] = {key: {name: fn(r['timing'][key] for r in runs)
                               for name, fn in [('median', statistics.median), ('mean', statistics.mean), ('min', min), ('max', max)]}
                           for key in ('planner', 'retrieval_rpc', 'image_fetch', 'qwen', 'logical_e2e')}
    aggregate['top20_status'] = dict(sum((Counter(r['top20_status']) for r in runs), Counter()))
    report = {'status': 'completed', 'runs': runs, 'aggregate': aggregate,
              'model_load_seconds': load_seconds, 'actual_wall_seconds': perf_counter() - wall_start,
              'source_unchanged': True,
              'limitations': ['Five stratified random source images, image-authored queries, single run: exploratory, not independent accuracy estimate.',
                              'One exact source frame per query; semantically valid neighboring/other frames are not exhaustively labeled.',
                              'One randomly selected image is a generic transition graphic; kept without resampling.',
                              'Resident CPU worker, warm OS/model caches possible; first Qwen inference is not warmed up separately.',
                              'Logical end-to-end excludes model loading, audit writes, HTTP API overhead and browser rendering.',
                              'Qwen only reranks original top20, no backfill. Missing/invalid scores remain unknown.']}
    save(output / 'REPORT.json', report)
    print(json.dumps({'status': 'completed', 'aggregate': aggregate, 'model_load_seconds': load_seconds,
                      'actual_wall_seconds': report['actual_wall_seconds']}), flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('queries', 'selection', 'mapping-root', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--settings', default='config/school-runtime.local.json')
    run(**vars(p.parse_args()))
