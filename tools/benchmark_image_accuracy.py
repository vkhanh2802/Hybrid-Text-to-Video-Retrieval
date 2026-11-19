"""Full production search accuracy/timing on frozen image-authored queries.

Only id/query enter planning/retrieval/reranking. Sparse positive image labels
are joined after execution; failures remain in the planned-query denominator.
"""
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import statistics
import time

from tools.benchmark_user_five_ocr import (
    ROOT, TimedImages, TimedScorer, load_mapping, quality, save, sha, validate_identities,
)
from tools.benchmark_groq_planner import retry_delay
from kis.llm import HTTPQueryStructurer, QueryAPIError


class RetryingPlanner(HTTPQueryStructurer):
    """Same production payloads; benchmark-only bounded network/quota retries."""
    def _post(self, route, payload):
        for attempt in range(3):
            start = time.perf_counter()
            row = {'attempt': attempt + 1, 'success': False}
            wait = None
            try:
                response = super()._post(route, payload)
                row.update(success=True, usage=response.get('usage'))
                return response
            except QueryAPIError as exc:
                row.update(http_status=exc.status, error_code=exc.code)
                if exc.status == 429 and attempt < 2:
                    wait = retry_delay(exc.detail, attempt)
                else:
                    raise
            except RuntimeError as exc:
                row['error'] = str(exc)[:400]
                if attempt < 2 and any(s in str(exc) for s in ('Cannot resolve', 'timed out', 'Cannot connect', 'Cannot reach')):
                    wait = 3. * (attempt + 1)
                else:
                    raise
            finally:
                row['http_seconds'] = time.perf_counter() - start
                self.calls.append(row)
            print(json.dumps({'phase': 'planner_retry', 'id': self.query_id,
                              'wait_seconds': wait, 'http_status': row.get('http_status'),
                              'error': row.get('error')}), flush=True)
            start_wait = time.perf_counter()
            time.sleep(wait)  # Rate/network backoff, not background-job polling.
            row['retry_wait_seconds'] = time.perf_counter() - start_wait


def stats(values):
    return {key: fn(values) for key, fn in [('median', statistics.median), ('mean', statistics.mean), ('min', min), ('max', max)]} if values else None


def run(queries, truth, mapping_root, output, settings):
    wall_start = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', KIS_VLM_BATCH_SIZE='12')
    import numpy
    import torch
    from tools.benchmark_school_ocr import client
    from kis.llm import load_query_api_config
    from kis.structure_query import SYSTEM_INSTRUCTION, QUERY_RESPONSE_SCHEMA, QUERY_RESPONSE_EXAMPLE, create_structured_query
    from kis.solve_kis import rerank_candidates_with_vlm
    from kis.web_app import LazyVLMReranker, serialize_search_results, require_frame_count
    from kis.ranking import get_search_profile

    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    inputs = json.loads(Path(queries).read_text(encoding='utf-8'))['queries']
    assert len(inputs) == len({q['id'] for q in inputs}) == 20
    assert all(set(q) == {'id', 'query'} for q in inputs)
    worker = client(settings, True)
    assert worker.info['missing_image_groups'] == worker.info['ocr']['missing_groups'] == []
    mapping, mapping_hashes = load_mapping(Path(mapping_root).resolve(), worker.groups)
    profile = get_search_profile(worker.info['runtime']['retrieval_profile'])
    api = load_query_api_config()
    api.require_ready()
    assert api.provider == 'groq' and api.model == 'openai/gpt-oss-20b'
    planner = RetryingPlanner(api, SYSTEM_INSTRUCTION, QUERY_RESPONSE_SCHEMA, example=QUERY_RESPONSE_EXAMPLE)
    paths = [Path(queries).resolve(), Path(truth).resolve(), Path(settings).resolve(), Path(__file__).resolve(),
             ROOT / 'tools/benchmark_groq_planner.py', ROOT / 'tools/benchmark_user_five_ocr.py',
             ROOT / 'tools/benchmark_school_ocr.py', ROOT / 'config/models.json', ROOT / 'config/pipeline.json',
             *sorted((ROOT / 'kis').glob('*.py'))]
    if (ROOT / 'config/llm.local.json').exists():
        paths.append(ROOT / 'config/llm.local.json')
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    hashes.update(mapping_hashes)
    save(output / 'PROTOCOL.json', {'queries': inputs, 'worker': worker.info, 'planner': api.public(),
        'max_tokens': api.max_tokens, 'temperature': api.temperature, 'gpu': torch.cuda.get_device_name(0),
        'batch_size': 12, 'qwen_budget': 20, 'backfill_missing_images': False,
        'temporal': 'production default', 'ocr': 'production retrieval and selective prompts',
        'repetitions': 1, 'max_transport_attempts': 3, 'network_retries_installed_in_production': False,
        'source_sha256': hashes, 'truth_policy': 'prelabeled source and matching neighbors; evaluation only after execution',
        'timing': 'query wall = planner including retries + retrieval RPC + image fetch/Qwen + serialization; audit writes/model load excluded',
        'retrieval_metrics': 'Hit@1/5/20/100 and MRR@100; planned denominator includes operational failures as zero',
        'semantic_review': 'separate post-run top5 image review, not used to change frozen positive labels',
        'excludes': ['HTTP web API overhead', 'browser grid image load/render'],
        'DRES': False, 'web_restart': False, 'active_config_change': False})
    start = time.perf_counter()
    model = LazyVLMReranker()._load()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    assert model.device.type == 'cuda' and model.batch_size == 12
    save(output / 'MODEL_LOAD.json', {'seconds': load_seconds, 'batch_size': model.batch_size})
    print(json.dumps({'phase': 'model_loaded', 'seconds': load_seconds}), flush=True)
    completed = []
    for q in inputs:
        planner.calls, planner.query_id = [], q['id']
        stage, structured, remote, ranked = 'planner', None, None, []
        timing, error, reads, calls, top20_status = {}, None, [], [], {}
        try:
            start = time.perf_counter()
            try:
                structured = create_structured_query(q['query'], query_id=q['id'], output_path=None, generator=planner)
            finally:
                timing['planner'] = time.perf_counter() - start
            assert structured['raw_query'] == q['query']
            save(output / (q['id'] + '_PLANNER.json'), {'structured': structured, 'seconds': timing['planner'], 'calls': planner.calls})
            stage = 'retrieval'
            start = time.perf_counter()
            remote = worker.retrieve(deepcopy(structured), temporal=None, ocr_region=None)
            timing['retrieval_rpc'] = time.perf_counter() - start
            validate_identities(remote['candidates'], mapping)
            save(output / (q['id'] + '_RETRIEVAL.json'), {'rpc_seconds': timing['retrieval_rpc'], **remote})
            print(json.dumps({'phase': 'retrieved', 'id': q['id'], 'planner_seconds': timing['planner'],
                              'CPU_RPC_seconds': timing['retrieval_rpc'], 'pool': len(remote['candidates'])}), flush=True)
            stage = 'rerank'
            candidates = deepcopy(remote['candidates'])
            images, scorer = TimedImages(worker, output), TimedScorer(model, torch)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            ranked = rerank_candidates_with_vlm(candidates, structured['raw_query'], scorer, worker,
                top_k=20, vlm_weight=.5, verified_first=profile.policy.verified_first,
                image_source=images, base_score_field='score', ocr_query=structured)[:100]
            rerank_wall = time.perf_counter() - start
            reads, calls = images.reads, scorer.calls
            audit = sum(r.get('audit_write_seconds', 0) for r in reads)
            timing.update(image_fetch=sum(r['seconds'] for r in reads), qwen=sum(c['seconds'] for c in calls),
                          rerank_including_images=rerank_wall - audit, audit_write_excluded=audit,
                          cpu_stages=remote['diagnostics'].get('stage_seconds'))
            assert len(reads) == 20
            assert [(r['video_id'], r['keyframe_n']) for r in reads] == [
                (r['video_id'], r['keyframe_n']) for r in remote['candidates'][:20]]
            top20_status = dict(Counter(r['vlm_status'] for r in candidates[:20]))
            stage = 'serialization'
            start = time.perf_counter()
            serialized = serialize_search_results(ranked, worker, image_resolver=images.reference)
            require_frame_count(serialized, 100)
            json.dumps(serialized, allow_nan=False)
            timing['serialization'] = time.perf_counter() - start
            timing['logical_e2e'] = sum(timing[k] for k in ('planner', 'retrieval_rpc', 'rerank_including_images', 'serialization'))
            validate_identities(ranked, mapping, 100)
            assert not worker.transport_failed
        except Exception as exc:
            error = {'stage': stage, 'type': type(exc).__name__, 'message': str(exc)[:1000],
                     'http_status': getattr(exc, 'status', None)}
            ranked = []
        row = {**q, 'completed': error is None, 'error': error, 'results': ranked, 'timing': timing,
               'planner_calls': planner.calls, 'image_reads': reads, 'qwen_calls': calls,
               'top20_status': top20_status, 'GPU_peak_allocated_bytes': torch.cuda.max_memory_allocated()}
        save(output / (q['id'] + '_RESULT.json'), row)
        completed.append((row, remote))
        print(json.dumps({'phase': 'query_complete', 'id': q['id'], 'completed': row['completed'],
                          'timing': timing, 'top20_status': top20_status, 'error': error}), flush=True)
    assert all(sha(ROOT / p) == h for p, h in hashes.items()), 'Frozen inputs/source changed during run'
    # Labels never enter any production inference call.
    labels = {q['id']: q for q in json.loads(Path(truth).read_text(encoding='utf-8'))['queries']}
    assert set(labels) == {q['id'] for q in inputs}
    runs = []
    for row, remote in completed:
        gt = labels[row['id']]
        for positive in gt['positives']:
            assert mapping[gt['video_id'], positive['keyframe_n']] == positive['frame_id']
        pool = remote['candidates'] if remote else []
        exact = dict(gt, keyframes=[gt['source_keyframe_n']])
        runs.append({k: v for k, v in row.items() if k != 'results'} | {
            'ground_truth': gt, 'before': quality(pool[:100], gt), 'pool': quality(pool, gt),
            'after': quality(row['results'], gt), 'exact_source_after': quality(row['results'], exact),
            'video_hit_at_100': int(any(r['video_id'] == gt['video_id'] for r in row['results'])),
            'top1': {k: row['results'][0].get(k) for k in ('video_id', 'keyframe_n', 'frame_id', 'vlm_raw_score', 'vlm_status')} if row['results'] else None})
    keys = ('hit_at_1', 'hit_at_5', 'hit_at_20', 'hit_at_100', 'mrr_at_100')
    def metrics(rows):
        return {stage: {key: statistics.mean(r[stage][key] for r in rows) for key in keys}
                for stage in ('before', 'after', 'exact_source_after')} if rows else None
    success = [r for r in runs if r['completed']]
    report = {'status': 'completed', 'planned': 20, 'completed': len(success), 'failed': 20 - len(success),
        'runs': runs, 'metrics_all_20_queries': metrics(runs), 'metrics_completed_queries': metrics(success),
        'pool_positive_hit_rate': statistics.mean(r['pool']['best_rank'] is not None for r in runs),
        'video_hit_at_100': statistics.mean(r['video_hit_at_100'] for r in runs),
        'successful_timing': {key: stats([r['timing'][key] for r in success])
                              for key in ('planner', 'retrieval_rpc', 'image_fetch', 'qwen', 'logical_e2e')},
        'top20_status': dict(sum((Counter(r['top20_status']) for r in runs), Counter())),
        'model_load_seconds': load_seconds, 'actual_wall_seconds': time.perf_counter() - wall_start,
        'planner_retry_wait_seconds': sum(c.get('retry_wait_seconds', 0) for r in runs for c in r['planner_calls']),
        'source_and_active_settings_unchanged': True,
        'limitations': ['20 self-authored image queries; exploratory refinding, not independent/exhaustive relevance labels.',
                        'Sparse positives fixed before search; matching unlabelled images can be counted as misses.',
                        'Post-run semantic review must be reported separately, without changing frozen-positive metrics.',
                        'Resident CPU worker, single run/query, model/OS caches may be warm.',
                        'No browser or HTTP web API overhead; audit writes and model load excluded from logical E2E.',
                        'Benchmark transport retries are measured but not installed into active web code.']}
    save(output / 'REPORT.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('runs', 'limitations')}), flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('queries', 'truth', 'mapping-root', 'output'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--settings', default='config/school-runtime.local.json')
    run(**vars(p.parse_args()))
