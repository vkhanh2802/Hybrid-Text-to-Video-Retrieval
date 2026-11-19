"""Live shared-planner, production CPU/image/Qwen A/B/C benchmark. No DRES.

GT is read only by preflight joins and post-run evaluation. Execution inputs are
projected to id/query, never supplemented with literals or corrected spellings.
"""
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'benchmarks/user_five_ocr_ablation_20261116.json'
PIN = '0fdb5b786abe23f23a843cd352e7497aba9f076e83d89cd0c5464dad0acc81c6'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def execution_inputs(data):
    return [{'id': q['id'], 'query': q['query']} for q in data['queries']]


def quality(rows, truth):
    ranks = [i for i, r in enumerate(rows, 1)
             if r['video_id'] == truth['video_id'] and r['keyframe_n'] in truth['keyframes']]
    first = min(ranks, default=None)
    return {'best_rank': first, 'positive_ranks': ranks,
            **{f'hit_at_{k}': int(first is not None and first <= k) for k in (1, 5, 20, 100)},
            'mrr_at_100': 1 / first if first is not None and first <= 100 else 0.0}


def strip_ocr(rows):
    rows = deepcopy(rows)
    for row in rows:
        row.pop('ocr', None)
    return rows


def validate_identities(rows, mapping, expected=None):
    if expected is not None and len(rows) != expected:
        raise ValueError('Incorrect result count')
    keys = [(r['video_id'], r['frame_id']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate frame identity')
    for r in rows:
        if mapping.get((r['video_id'], r['keyframe_n'])) != r['frame_id']:
            raise ValueError('Result differs from authoritative keyframe mapping')


def load_mapping(root, groups):
    # Full production CSVs can alias several keyframe numbers to one frame_idx
    # (L21 first frame). This is valid for a forward join; output identities are
    # still strictly deduplicated. Do not use OCR-ingest's bijection requirement.
    result, hashes = {}, {}
    for group in groups:
        paths = sorted((Path(root) / group).rglob(f'{group}_V*.csv'))
        if not paths:
            raise ValueError('Missing production mapping group')
        for path in paths:
            hashes[path.relative_to(ROOT).as_posix()] = sha(path)
            with path.open(encoding='utf-8', newline='') as stream:
                for row in csv.DictReader(stream):
                    key = (path.stem, int(row['n']))
                    if key in result:
                        raise ValueError('Ambiguous keyframe number')
                    result[key] = int(row['frame_idx'])
    return result, hashes


class TimedImages:
    """Delegate every production read; save bytes for later visual review, no cache."""
    def __init__(self, worker, output):
        self.worker, self.output, self.reads = worker, output, []

    def reference(self, *args, **kwargs):
        return self.worker.reference(*args, **kwargs)

    def open_candidate(self, candidate, lookup):
        from contextlib import contextmanager
        from io import BytesIO

        @contextmanager
        def opened():
            start = perf_counter()
            record = {'video_id': candidate['video_id'], 'keyframe_n': candidate['keyframe_n'],
                      'success': False}
            try:
                data, mime = self.worker.read(candidate['video_id'], candidate['keyframe_n'])
                record.update(success=True, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), mime=mime)
            except FileNotFoundError:
                record['error'] = 'image_not_found'
                raise
            finally:
                record['seconds'] = perf_counter() - start
                self.reads.append(record)
            # Audit writing is measured separately, excluded from logical timing.
            start = perf_counter()
            folder = self.output / 'images' / candidate['video_id']
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{candidate['keyframe_n']:03d}.jpg"
            if path.exists():
                assert sha(path) == record['sha256'], 'Source image changed across arms'
            else:
                path.write_bytes(data)
            record['audit_write_seconds'] = perf_counter() - start
            with BytesIO(data) as stream:
                yield stream
        return opened()


class TimedScorer:
    supports_batch = True
    supports_ocr = True

    def __init__(self, model, torch):
        self.model, self.torch, self.calls = model, torch, []

    @property
    def last_batch_errors(self):
        return self.model.last_batch_errors

    def score_batch(self, images, query, captions, **kwargs):
        self.torch.cuda.synchronize()
        start = perf_counter()
        record = {'inputs': len(images), 'ocr_argument': 'ocrs' in kwargs, 'success': False}
        try:
            values = self.model.score_batch(images, query, captions, **kwargs)
            self.torch.cuda.synchronize()
            record.update(success=True, scores=values,
                          diagnostics=deepcopy(self.model.last_batch_stats),
                          errors=deepcopy(self.model.last_batch_errors))
            return values
        finally:
            record['seconds'] = perf_counter() - start
            self.calls.append(record)


def run(output, current_settings='config/school-runtime.local.json',
        baseline_settings='config/school-runtime.ocr-before.local.json', selective=False):
    wall_start = perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', KIS_VLM_BATCH_SIZE='12')
    import numpy  # Windows OpenMP must precede torch.
    import torch
    from tools.benchmark_school_ocr import client
    from kis.structure_query import create_query_structurer, create_structured_query
    from kis.llm import load_query_api_config
    from kis.solve_kis import rerank_candidates_with_vlm
    from kis.web_app import LazyVLMReranker, serialize_search_results, require_frame_count
    from kis.ranking import get_search_profile

    data = json.loads(INPUT.read_text(encoding='utf-8'))
    mapping = {}
    # Validate full production mappings, not only ground truth groups.
    baseline = client(baseline_settings)
    current = client(current_settings, True)
    if not selective:
        raise ValueError('This runner now requires --selective for the integrated prompt policy; historical full-OCR runs are retained')
    expected_pin = json.loads(Path(current_settings).read_text(encoding='utf-8'))['ocr_manifest_sha256']
    assert current.info['runtime']['ocr_manifest_sha256'] == expected_pin
    assert current.info['ocr']['records'] > 0
    mapping, mapping_hashes = load_mapping(ROOT / 'dataset/mapping', current.groups)
    truth_joins = []
    for q in data['queries']:
        truth_joins.append({'id': q['id'], 'supplied': q['supplied_ground_truth'],
                            'positives': [{'video_id': q['video_id'], 'keyframe_n': n,
                                           'frame_id': mapping[q['video_id'], n]} for n in q['keyframes']]})
    save(output / 'GROUND_TRUTH_JOINS.json', truth_joins)
    save(output / 'QUERIES.json', data)
    profile = get_search_profile(current.info['runtime']['retrieval_profile'])
    api = load_query_api_config()
    generator = create_query_structurer(api)
    files = [INPUT, Path(__file__).resolve(), *sorted((ROOT / 'kis').glob('*.py')),
             ROOT / 'config/models.json', ROOT / 'config/pipeline.json',
             ROOT / 'tools/benchmark_school_ocr.py']
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    hashes.update(mapping_hashes)
    protocol = {'arms': {'A': 'OCR-off CPU, no OCR prompt', 'B': 'current OCR CPU, no OCR prompt',
                         'C': 'same B pool, selective relevant OCR excerpt <=384 chars'},
                'planner': api.public(), 'planner_sharing': 'one real production planning operation/query; repair permitted by production',
                'arm_order': ['ABC', 'BCA', 'CAB', 'ABC', 'BCA'], 'repetitions': 1,
                'scope': list(current.groups), 'worker_baseline': baseline.info, 'worker_current': current.info,
                'batch_size': 12, 'qwen_budget': 20, 'backfill_missing_images': False,
                'ocr_prompt_policy': 'selective', 'snapshot_pin': expected_pin,
                'ground_truth_used_for': 'evaluation only; no query repair or manual OCR literals',
                'temporal': 'production default (None)', 'source_sha256': hashes,
                'timing': 'logical E2E = shared measured planner + CPU RPC + production rerank/image fetch + serialization; excludes audit writes/model load; B/C share one CPU call',
                'excludes': ['browser grid image loading/rendering', 'DRES', 'HTTP API server overhead'],
                'web_restart': False, 'DRES_calls': False,
                'gpu': torch.cuda.get_device_name(0)}
    save(output / 'PROTOCOL.json', protocol)
    packets = {}
    for q in execution_inputs(data):
        start = perf_counter()
        structured = create_structured_query(q['query'], output_path=None, generator=generator)
        planner_seconds = perf_counter() - start
        assert structured['raw_query'] == q['query']
        packet = {'id': q['id'], 'structured': structured, 'planner_seconds': planner_seconds, 'retrieval': {}}
        save(output / f"{q['id']}_PLANNER.json", {'structured': structured, 'seconds': planner_seconds})
        for arm, worker in [('A', baseline), ('B', current)]:
            start = perf_counter()
            remote = worker.retrieve(deepcopy(structured), temporal=None, ocr_region=None)
            seconds = perf_counter() - start
            validate_identities(remote['candidates'], mapping)
            packet['retrieval'][arm] = {'rpc_seconds': seconds, **remote}
        packets[q['id']] = packet
        save(output / f"{q['id']}_RETRIEVAL.json", packet)
        print(json.dumps({'phase': 'planning_retrieval', 'query': q['id'], 'planner_seconds': planner_seconds,
                          'rpc_seconds': {a: r['rpc_seconds'] for a, r in packet['retrieval'].items()}}), flush=True)

    # Separate model cold load, not charged to the arbitrarily first arm.
    start = perf_counter()
    lazy = LazyVLMReranker()
    model = lazy._load()
    torch.cuda.synchronize()
    load_seconds = perf_counter() - start
    assert model.device.type == 'cuda' and model.batch_size == 12
    save(output / 'MODEL_LOAD.json', {'seconds': load_seconds, 'device': str(model.device), 'batch_size': model.batch_size})
    results = []
    for index, q in enumerate(execution_inputs(data)):
        packet = packets[q['id']]
        for arm in protocol['arm_order'][index]:
            source = packet['retrieval']['A' if arm == 'A' else 'B']
            candidates = deepcopy(source['candidates']) if arm == 'C' else strip_ocr(source['candidates'])
            before = deepcopy(candidates)
            from kis.ocr_routing import select_prompt_ocr
            selected_ocr = {(r['video_id'], r['keyframe_n']): select_prompt_ocr(r.get('ocr'), packet['structured'])
                            for r in before[:20]}
            worker = baseline if arm == 'A' else current
            images, scorer = TimedImages(worker, output), TimedScorer(model, torch)
            torch.cuda.reset_peak_memory_stats()
            start = perf_counter()
            ranked = rerank_candidates_with_vlm(candidates, packet['structured']['raw_query'], scorer,
                        worker, top_k=20, vlm_weight=0.5, verified_first=profile.policy.verified_first,
                        image_source=images, base_score_field='score', ocr_query=packet['structured'])[:100]
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
            assert [(r['video_id'], r['keyframe_n']) for r in images.reads] == [(r['video_id'], r['keyframe_n']) for r in before[:20]]
            if not scorer.calls or not all(c['success'] for c in scorer.calls):
                raise RuntimeError('Qwen did not execute successfully')
            assert len(scorer.calls) == 1
            assert scorer.calls[0]['inputs'] == sum(r['success'] for r in images.reads)
            expected_ocr = any(r['success'] and selected_ocr[r['video_id'], r['keyframe_n']] is not None for r in images.reads)
            assert scorer.calls[0]['ocr_argument'] == expected_ocr
            assert scorer.calls[0]['diagnostics']['requested_batch_size'] == 12
            row = {'id': q['id'], 'arm': arm, 'results': ranked, 'serialized': serialized,
                   'timing': {'planner': packet['planner_seconds'], 'retrieval_rpc': source['rpc_seconds'],
                              'cpu_stages': source['diagnostics'].get('stage_seconds'),
                              'image_fetch': sum(r['seconds'] for r in images.reads),
                              'qwen': sum(c['seconds'] for c in scorer.calls),
                              'rerank_including_images': rerank_seconds - audit_seconds,
                              'serialization': serialization_seconds,
                              'audit_write_excluded': audit_seconds,
                              'logical_e2e': packet['planner_seconds'] + source['rpc_seconds'] + rerank_seconds - audit_seconds + serialization_seconds},
                   'image_reads': images.reads, 'qwen_calls': scorer.calls,
                   'top20_status': dict(Counter(r['vlm_status'] for r in candidates[:20])),
                   'ocr_prompt_candidates': sum(r.get('vlm_ocr_used') is True for r in candidates[:20]),
                   'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                   'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved()}
            save(output / f"{q['id']}_{arm}.json", row)
            results.append(row)
            print(json.dumps({'phase': 'qwen', 'query': q['id'], 'arm': arm, 'timing': row['timing'],
                              'status': row['top20_status']}), flush=True)
    unchanged = all(sha(ROOT / p) == h for p, h in hashes.items())
    assert unchanged
    report = {'status': 'completed', 'source_unchanged': unchanged, 'model_load_seconds': load_seconds,
              'actual_wall_seconds': perf_counter() - wall_start, 'runs': [], 'aggregate': {}}
    # Only now join ground truth onto completed pipeline outputs.
    truth = {q['id']: q for q in data['queries']}
    for r in results:
        source = packets[r['id']]['retrieval']['A' if r['arm'] == 'A' else 'B']
        report['runs'].append({k: v for k, v in r.items() if k not in ('results', 'serialized')} |
            {'before': quality(source['candidates'][:100], truth[r['id']]),
             'pool': quality(source['candidates'], truth[r['id']]),
             'after': quality(r['results'], truth[r['id']]),
             'top1': {k: r['results'][0].get(k) for k in ('video_id', 'keyframe_n', 'frame_id', 'vlm_raw_score', 'vlm_status')}})
    for arm in 'ABC':
        rows = [r for r in report['runs'] if r['arm'] == arm]
        report['aggregate'][arm] = {
            **{stage: {key: statistics.mean(r[stage][key] for r in rows)
                       for key in ('hit_at_1', 'hit_at_5', 'hit_at_20', 'hit_at_100', 'mrr_at_100')} for stage in ('before', 'after')},
            'timing_median': {key: statistics.median(r['timing'][key] for r in rows)
                              for key in ('planner', 'retrieval_rpc', 'image_fetch', 'qwen', 'logical_e2e')},
            'top20_status': dict(sum((Counter(r['top20_status']) for r in rows), Counter()))}
    report['limitations'] = ['Five user-supplied queries, single run per arm, sparse acceptable keyframes; not exhaustive relevance labels.',
                             'B/C share CPU retrieval; planner shared across all three arms. Logical totals are not three independent wall-time requests.',
                             'Resident school workers, warm OS caches; arm order rotated but no repetition/confidence interval.',
                             'OCR coverage is the worker scope recorded in PROTOCOL; missing OCR is unknown. Some production image groups absent; no backfill.',
                             'No HTTP API overhead or browser grid rendering included. Cold model cost reported separately.']
    save(output / 'REPORT.json', report)
    print(json.dumps({'status': report['status'], 'actual_wall_seconds': report['actual_wall_seconds'],
                      'model_load_seconds': load_seconds, 'aggregate': report['aggregate']}), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--current-settings', default='config/school-runtime.local.json')
    parser.add_argument('--baseline-settings', default='config/school-runtime.ocr-before.local.json')
    parser.add_argument('--selective', action='store_true')
    args = parser.parse_args()
    run(args.output, args.current_settings, args.baseline_settings, args.selective)
