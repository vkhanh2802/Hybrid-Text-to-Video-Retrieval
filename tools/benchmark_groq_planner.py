"""Real Groq-only planner benchmark, unchanged production model/prompt/config.

Bounded retries are benchmark transport policy, not a production code change.
Pacing between queries is separate; retry waits are included in query latency.
"""
from collections import Counter
import json
from pathlib import Path
import re
import statistics
import time

from tools.benchmark_user_five_ocr import ROOT, save, sha


def retry_delay(detail, attempt):
    match = re.search(r'try again in\s+([0-9.]+)\s*(ms|s|m)\b', detail, re.I)
    if match:
        seconds = float(match[1]) * {'ms': .001, 's': 1, 'm': 60}[match[2].lower()]
        return min(60., max(1., seconds + .5))
    return min(60., 15. * (attempt + 1))


def load_queries(spec):
    data = json.loads(Path(spec).read_text(encoding='utf-8'))
    base_spec = ROOT / data['base_spec']
    local = json.loads(base_spec.read_text(encoding='utf-8'))
    base = ROOT / local['base_queries']
    queries = json.loads(base.read_text(encoding='utf-8'))['queries'] + local['probes'] + data['additional_queries']
    if len(queries) != 20 or len({q['id'] for q in queries}) != 20 or any(set(q) != {'id', 'query'} for q in queries):
        raise ValueError('Require exactly 20 unique execution-only queries')
    return queries, [Path(spec).resolve(), base_spec, base]


def evaluate_checks(structured, expected):
    if structured is None:
        return None
    observed = {'needs_ocr': structured['needs_ocr'], 'needs_asr': structured['needs_asr'],
                'has_sequence': bool(structured['temporal_constraints']),
                'has_question': structured['question'] is not None}
    checks = {key: observed[key] == expected[key] for key in observed if key in expected}
    if 'minimum_events' in expected:
        checks['minimum_events'] = len(structured['events']) >= expected['minimum_events']
    if 'ocr_literal' in expected:
        # Do not count raw_query/query_variants, which normalizer copies verbatim.
        generated_text = json.dumps({key: structured[key] for key in ('entities', 'visual_description', 'stream_queries')}, ensure_ascii=False)
        checks['ocr_literal_preserved'] = expected['ocr_literal'] in generated_text
    return {'checks': checks, 'all_pass': all(checks.values()), 'observed': observed}


def run(output, pace_seconds=10):
    from kis.llm import HTTPQueryStructurer, QueryAPIError, load_query_api_config, parse_query_json
    from kis.structure_query import (
        SYSTEM_INSTRUCTION, QUERY_RESPONSE_SCHEMA, QUERY_RESPONSE_EXAMPLE,
        canonicalize_model_query, create_structured_query,
    )
    import jsonschema
    wall_start = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    spec = ROOT / 'benchmarks/groq_planner_20_20261118.json'
    expected_path = ROOT / 'benchmarks/groq_planner_20_20261118.expectations.json'
    queries, files = load_queries(spec)
    files += [expected_path, Path(__file__).resolve(), ROOT / 'kis/llm.py', ROOT / 'kis/structure_query.py',
              ROOT / 'config/models.json', ROOT / 'config/pipeline.json', ROOT / 'config/school-runtime.local.json']
    if (ROOT / 'config/llm.local.json').exists():
        files.append(ROOT / 'config/llm.local.json')
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in files}
    config = load_query_api_config()
    config.require_ready()
    assert config.provider == 'groq' and config.model == 'openai/gpt-oss-20b'

    class TracedGenerator(HTTPQueryStructurer):
        def _post(self, route, payload):
            for attempt in range(3):
                start = time.perf_counter()
                record = {'route': route, 'transport_attempt': attempt + 1, 'success': False,
                          'max_tokens': payload.get(config.token_parameter),
                          'response_format': payload.get('response_format', {}).get('type')}
                wait = None
                try:
                    response = super()._post(route, payload)
                    record.update(success=True, usage=response.get('usage'))
                    for choice in response.get('choices', []):
                        self.raw_outputs.append({'content': choice.get('message', {}).get('content'),
                                                 'finish_reason': choice.get('finish_reason')})
                    return response
                except QueryAPIError as exc:
                    record.update(http_status=exc.status, error_code=exc.code)
                    if exc.failed_generation:
                        self.raw_outputs.append({'content': exc.failed_generation, 'provider_rejected': True})
                    if exc.status == 429 and attempt < 2:
                        wait = retry_delay(exc.detail, attempt)
                    else:
                        raise
                except RuntimeError as exc:
                    message = str(exc)
                    record['error'] = message[:400]
                    if attempt < 2 and any(s in message for s in ('Cannot resolve', 'timed out', 'Cannot connect', 'Cannot reach')):
                        wait = 3. * (attempt + 1)
                    else:
                        raise
                finally:
                    record['http_seconds'] = time.perf_counter() - start
                    self.calls.append(record)
                if wait is not None:
                    print(json.dumps({'phase': 'transport_retry', 'id': self.query_id,
                                      'wait_seconds': wait, 'http_status': record.get('http_status'),
                                      'error': record.get('error')}), flush=True)
                    start_wait = time.perf_counter()
                    time.sleep(wait)  # Provider/network backoff, not background-job polling.
                    record['retry_wait_seconds'] = time.perf_counter() - start_wait

    generator = TracedGenerator(config, SYSTEM_INSTRUCTION, QUERY_RESPONSE_SCHEMA, example=QUERY_RESPONSE_EXAMPLE)
    save(output / 'PROTOCOL.json', {'planner': config.public(), 'max_tokens': config.max_tokens,
        'temperature': config.temperature, 'queries': queries, 'repetitions': 1,
        'between_query_pace_seconds': pace_seconds, 'max_transport_attempts': 3,
        'production_schema_repairs': 'unchanged; maximum two production generation operations',
        'timing': 'query wall includes bounded transport retries/waits and production repairs; between-query pacing separate',
        'scope': 'planner_only_no_retrieval_no_reranker', 'source_sha256': hashes,
        'DRES': False, 'active_settings_changed': False, 'web_restarted': False,
        'ground_truth': 'expectations loaded only after execution, never in model inputs',
        'fail_fast': 'stop after three consecutive infrastructure-blocked queries'})
    save(output / 'QUERIES.json', queries)
    rows, pacing_waits = [], []
    consecutive_infra = 0
    for i, q in enumerate(queries):
        if i and pace_seconds:
            start = time.perf_counter()
            time.sleep(pace_seconds)
            pacing_waits.append(time.perf_counter() - start)
        generator.calls, generator.raw_outputs, generator.query_id = [], [], q['id']
        structured, error = None, None
        start = time.perf_counter()
        try:
            structured = create_structured_query(q['query'], query_id=q['id'], output_path=None, generator=generator)
        except (RuntimeError, ValueError) as exc:
            error = {'type': type(exc).__name__, 'message': str(exc)[:1000],
                     'http_status': getattr(exc, 'status', None)}
        seconds = time.perf_counter() - start
        schema_checks = []
        for raw in generator.raw_outputs:
            try:
                parsed = parse_query_json(raw['content'])
                canonical = canonicalize_model_query(parsed)
                jsonschema.validate(canonical, QUERY_RESPONSE_SCHEMA)
                schema_checks.append({'canonical_schema_valid': True})
            except (RuntimeError, ValueError, jsonschema.ValidationError) as exc:
                schema_checks.append({'canonical_schema_valid': False, 'error': str(exc).splitlines()[0][:500]})
        retry_wait = sum(c.get('retry_wait_seconds', 0) for c in generator.calls)
        row = {**q, 'success': structured is not None, 'error': error, 'structured': structured,
               'timing': {'query_wall_seconds': seconds, 'http_seconds': sum(c['http_seconds'] for c in generator.calls),
                          'retry_wait_seconds': retry_wait, 'query_without_retry_wait_seconds': seconds - retry_wait},
               'calls': generator.calls, 'raw_outputs': generator.raw_outputs, 'schema_checks': schema_checks}
        save(output / (q['id'] + '.json'), row)
        rows.append(row)
        print(json.dumps({'phase': 'query_complete', 'id': q['id'], 'success': row['success'],
                          'timing': row['timing'], 'transport_calls': len(generator.calls), 'error': error}), flush=True)
        infra = error and (error['http_status'] == 429 or any(s in error['message'] for s in ('Cannot resolve', 'timed out', 'Cannot connect', 'Cannot reach')))
        consecutive_infra = consecutive_infra + 1 if infra else 0
        if consecutive_infra >= 3:
            break
    # Only now load expectation labels; no labels influenced query execution.
    expected = json.loads(expected_path.read_text(encoding='utf-8'))['queries']
    for row in rows:
        row['evaluation'] = evaluate_checks(row['structured'], expected[row['id']])
    assert all(sha(ROOT / p) == h for p, h in hashes.items())
    success = [r for r in rows if r['success']]
    def stats(values):
        return {k: f(values) for k, f in [('median', statistics.median), ('mean', statistics.mean), ('min', min), ('max', max)]} if values else None
    calls = [c for r in rows for c in r['calls']]
    report = {'status': 'completed' if len(rows) == 20 else 'partial_infrastructure_blocked',
        'planned': 20, 'attempted': len(rows), 'successful': len(success), 'failed': len(rows) - len(success),
        'not_attempted': [q['id'] for q in queries[len(rows):]], 'runs': rows,
        'success_query_wall_seconds': stats([r['timing']['query_wall_seconds'] for r in success]),
        'success_query_without_retry_wait_seconds': stats([r['timing']['query_without_retry_wait_seconds'] for r in success]),
        'all_attempted_query_wall_seconds': stats([r['timing']['query_wall_seconds'] for r in rows]),
        'all_automatic_checks_pass': sum(r['evaluation']['all_pass'] for r in success),
        'check_counts': {key: {'passed': sum(r['evaluation']['checks'].get(key) is True for r in success),
                               'evaluated': sum(key in r['evaluation']['checks'] for r in success)}
                         for key in sorted({k for r in success for k in r['evaluation']['checks']})},
        'transport_calls': len(calls), 'http_status_counts': dict(Counter(str(c.get('http_status', 200 if c['success'] else 'network_error')) for c in calls)),
        'total_retry_wait_seconds': sum(r['timing']['retry_wait_seconds'] for r in rows),
        'total_between_query_pacing_seconds': sum(pacing_waits), 'actual_wall_seconds': time.perf_counter() - wall_start,
        'usage': {key: sum((c.get('usage') or {}).get(key, 0) for c in calls) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')},
        'source_and_settings_unchanged': True,
        'limitations': ['20 self-authored/frozen queries, one run each; not independent accuracy evaluation.',
                        'Automatic flag/literal checks are not full semantic correctness; manual review required.',
                        'Benchmark retries/pacing are not installed in production. Account quota may be shared.',
                        'Planner only: no retrieval accuracy or full-pipeline latency claim.']}
    save(output / 'REPORT.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'runs'}), flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--pace-seconds', type=float, default=10)
    args = p.parse_args()
    if args.pace_seconds < 0:
        p.error('Pacing must be nonnegative')
    run(args.output, args.pace_seconds)
