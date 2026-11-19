"""Real GPU cost sweep over frozen OCR-retrieval pools; no production limit change.

Calls the production scorer in contiguous batches <=12 (its public input cap is
20). Every budget is measured independently, fetching real school images anew.
No backfill, OCR prompts, score reuse, planner/retrieval calls, or DRES.
"""
from collections import Counter
from copy import deepcopy
from io import BytesIO
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from time import perf_counter

BUDGETS = (20, 40, 60, 80, 100)
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'benchmark_results/user_five_selective_ocr_20261116_run01'


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def chunks(values, size=12):
    return [values[i:i + size] for i in range(0, len(values), size)]


def score_status(value, error):
    if error:
        return 'error'
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 10:
        return 'invalid_score'
    return 'valid'


def summarize(rows, budgets=None):
    budgets = tuple(budgets or BUDGETS)
    summary = {}
    for budget in budgets:
        selected = [r for r in rows if r['budget'] == budget]
        summary[str(budget)] = {
            'queries': len(selected),
            **{key: {'median': statistics.median(r[key] for r in selected),
                     'mean': statistics.mean(r[key] for r in selected),
                     'min': min(r[key] for r in selected), 'max': max(r[key] for r in selected)}
               for key in ('image_fetch_seconds', 'qwen_seconds', 'measured_total_seconds')},
            'images_scored': sum(r['images_scored'] for r in selected),
            'valid_scores': sum(r['statuses'].get('valid', 0) for r in selected),
            'missing_images': sum(r['statuses'].get('missing_image', 0) for r in selected),
            'invalid_or_error': sum(r['statuses'].get('invalid_score', 0) + r['statuses'].get('error', 0) for r in selected),
            'oom_splits': sum(r['oom_splits'] for r in selected),
            'peak_allocated_MiB': max(r['peak_allocated_bytes'] for r in selected) / 1024**2,
            'peak_reserved_MiB': max(r['peak_reserved_bytes'] for r in selected) / 1024**2}
    base = summary.get('20', {}).get('measured_total_seconds', {}).get('median')
    for value in summary.values():
        value['total_median_ratio_to_20'] = value['measured_total_seconds']['median'] / base if base else None
    return summary


def write_report(output, report):
    budgets = report.get('budgets', BUDGETS)
    lines = ['# Chi phí Qwen top ' + '/'.join(map(str, budgets)), '',
             '5 query, cùng frozen OCR-retrieval pool mỗi query. Mỗi mức tải ảnh/chấm mới, batch12, không OCR prompt; không backfill ảnh thiếu.',
             'Đây là cost sweep riêng, không đổi production top20. Thời gian gồm lấy ảnh CPU→laptop và production Qwen scorer. Planner/retrieval không chạy lại và không tính vào số đo.', '',
             '| TopK | Median lấy ảnh (s) | Median Qwen (s) | Median tổng (s) | So với20 | Peak allocated VRAM (MiB) | Ảnh thiếu | Score invalid/error |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for budget in budgets:
        s = report['summary'][str(budget)]
        ratio = f"{s['total_median_ratio_to_20']:.2f}x" if s['total_median_ratio_to_20'] is not None else '—'
        lines.append(f"| {budget} | {s['image_fetch_seconds']['median']:.2f} | {s['qwen_seconds']['median']:.2f} | {s['measured_total_seconds']['median']:.2f} | {ratio} | {s['peak_allocated_MiB']:.0f} | {s['missing_images']} | {s['invalid_or_error']} |")
    lines += ['', '## Tổng thời gian từng query (giây)', '', '| Query | ' + ' | '.join(f'Top{b}' for b in budgets) + ' |', '|---|' + '---:|' * len(budgets)]
    for q in ['q1', 'q2', 'q3', 'q4', 'q5']:
        rows = {r['budget']: r for r in report['runs'] if r['id'] == q}
        lines.append('| ' + q + ' | ' + ' | '.join(f"{rows[b]['measured_total_seconds']:.2f}" for b in budgets) + ' |')
    lines += ['', f"- Nạp model cold riêng: **{report['model_load_seconds']:.2f}s**.",
              f"- Wall time chương trình: **{report['actual_wall_seconds']:.2f}s** (gồm imports/preflight/load/artifact writes).",
              '- Query/budget chạy xoay vòng; không warm-up bằng query khác. First inference được đánh dấu riêng trong JSON.',
              '- Một lần mỗi query/mức, không có khoảng tin cậy; cache hệ điều hành/server và nhiệt GPU có thể ảnh hưởng.',
              '- VRAM là peak tensor allocated/reserved của PyTorch, không phải toàn bộ VRAM các process. Không đo điện năng hoặc quy đổi chi phí tiền.',
              '- score_batch gọi nhiều lần tối đa12ảnh để giữ API cap20. Không dùng production reranker (bị chặn top20), không đưa ra metric chất lượng ranking giả cho top100.',
              '- GT chỉ đối chiếu sau đo để báo coverage, không đưa vào scorer. Việc ảnh GT được chấm không đồng nghĩa kết quả cuối đúng.', '',
              '## GT có cơ hội được chấm (coverage của pool cố định)', '',
              '| Query | Rank GT trước Qwen | Các budget có ít nhất 1 GT được chấm hợp lệ |', '|---|---:|---|']
    for r in report['gt_coverage']:
        lines.append(f"| {r['id']} | {r['best_pool_rank']} | {', '.join(map(str, r['budgets_valid_gt'])) or 'Không'} |")
    (output / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    fields = ['id', 'budget', 'image_fetch_seconds', 'qwen_seconds', 'measured_total_seconds', 'images_scored', 'peak_allocated_bytes', 'peak_reserved_bytes', 'oom_splits', 'first_inference']
    with (output / 'COST.csv').open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in report['runs'])


def run(output, budgets=BUDGETS):
    budgets = tuple(budgets)
    if not budgets or len(set(budgets)) != len(budgets) or any(type(b) is not int or not 1 <= b <= 100 for b in budgets):
        raise ValueError('Budgets must be unique integers from 1 to 100')
    wall = perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
                      OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', KIS_VLM_BATCH_SIZE='12')
    import numpy
    import torch
    from tools.benchmark_school_ocr import client
    from tools.benchmark_user_five_ocr import load_mapping, validate_identities
    from kis.web_app import LazyVLMReranker
    worker = client('config/school-runtime.local.json', True)
    old_protocol = json.loads((SOURCE / 'PROTOCOL.json').read_text(encoding='utf-8'))
    assert worker.info['runtime'] == old_protocol['worker_current']['runtime']
    mapping, _ = load_mapping(ROOT / 'dataset/mapping', worker.groups)
    packets, inputs = {}, {}
    for i in range(1, 6):
        p = SOURCE / f'q{i}_RETRIEVAL.json'
        inputs[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
        packet = json.loads(p.read_text(encoding='utf-8'))
        rows = packet['retrieval']['B']['candidates'][:100]
        validate_identities(rows, mapping, 100)
        packets[f'q{i}'] = {'query': packet['structured']['raw_query'], 'rows': rows}
    source_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [Path(__file__).resolve(), *sorted((ROOT / 'kis').glob('*.py'))]}
    schedule = []
    # Each budget occurs once in each position. Interleave queries to avoid
    # evaluating all large budgets only at the end of a hot GPU session.
    for round_index in range(len(budgets)):
        for query_index in range(5):
            schedule.append((f'q{query_index + 1}', budgets[(round_index + query_index) % len(budgets)]))
    save(output / 'PROTOCOL.json', {'budgets': budgets, 'batch_size': 12, 'schedule': schedule,
         'frozen_pool_inputs': inputs, 'source_sha256': source_hashes,
         'runtime': worker.info['runtime'], 'gpu': torch.cuda.get_device_name(0),
         'policy': 'OCR retrieval frozen; no OCR prompt; each budget fresh image fetch and inference',
         'GT_in_scoring': False, 'score_reuse': False, 'missing_backfill': False,
         'new_planner_calls': 0, 'new_retrieval_calls': 0, 'production_top20_unchanged': True,
         'DRES': False, 'web_restart': False})
    lazy = LazyVLMReranker()
    start = perf_counter()
    model = lazy._load()
    torch.cuda.synchronize()
    load_seconds = perf_counter() - start
    assert model.device.type == 'cuda' and model.batch_size == 12
    save(output / 'MODEL_LOAD.json', {'seconds': load_seconds, 'batch_size': 12})
    results, image_hashes = [], {}
    for sequence, (qid, budget) in enumerate(schedule):
        packet = packets[qid]
        torch.cuda.reset_peak_memory_stats()
        started = perf_counter()
        reads, pending, scored, calls = [], [], [], []
        try:
            for rank, candidate in enumerate(packet['rows'][:budget], 1):
                key = (candidate['video_id'], candidate['keyframe_n'])
                rec = {'rank': rank, 'video_id': key[0], 'keyframe_n': key[1], 'frame_id': candidate['frame_id']}
                start = perf_counter()
                try:
                    data, mime = worker.read(*key)
                    digest = hashlib.sha256(data).hexdigest()
                    assert image_hashes.setdefault(key, digest) == digest, 'Source pixels changed'
                    rec.update(status='loaded', bytes=len(data), sha256=digest)
                    pending.append((rec, BytesIO(data), candidate.get('retrieval_text', '')))
                except FileNotFoundError:
                    rec.update(status='missing_image', score=None)
                    scored.append(dict(rec))
                finally:
                    rec['seconds'] = perf_counter() - start
                    reads.append(rec)
            for batch in chunks(pending):
                torch.cuda.synchronize()
                start = perf_counter()
                values = model.score_batch([x[1] for x in batch], packet['query'], [x[2] for x in batch], batch_size=12)
                torch.cuda.synchronize()
                seconds = perf_counter() - start
                assert len(values) == len(batch)
                errors = model.last_batch_errors
                stats = deepcopy(model.last_batch_stats)
                calls.append({'seconds': seconds, 'diagnostics': stats})
                for (rec, _, _), value, error in zip(batch, values, errors):
                    status = score_status(value, error)
                    scored.append({**rec, 'status': status, 'score': value if status == 'valid' else None, 'error': error})
            total = perf_counter() - started
            assert len(scored) == budget and not worker.transport_failed
            row = {'id': qid, 'budget': budget, 'sequence': sequence + 1, 'first_inference': sequence == 0,
                   'image_fetch_seconds': sum(r['seconds'] for r in reads),
                   'qwen_seconds': sum(c['seconds'] for c in calls), 'measured_total_seconds': total,
                   'images_scored': len(pending), 'statuses': dict(Counter(r['status'] for r in scored)),
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                   'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                   'oom_splits': sum(c['diagnostics']['oom_splits'] for c in calls),
                   'calls': calls, 'scores': sorted(scored, key=lambda r: r['rank']), 'image_reads': reads}
            save(output / f'{qid}_top{budget}.json', row)
            results.append(row)
            print(json.dumps({k: row[k] for k in ('id', 'budget', 'image_fetch_seconds', 'qwen_seconds', 'measured_total_seconds', 'images_scored', 'statuses', 'oom_splits')}), flush=True)
        finally:
            for _, stream, _ in pending:
                stream.close()
    assert all(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value for name, value in {**inputs, **source_hashes}.items())
    # Evaluation-only GT join after all scoring finished.
    truth = json.loads((SOURCE / 'QUERIES.json').read_text(encoding='utf-8'))['queries']
    coverage = []
    for q in truth:
        matches = lambda r: r['video_id'] == q['video_id'] and r['keyframe_n'] in q['keyframes']
        rank = next((i for i, r in enumerate(packets[q['id']]['rows'], 1) if matches(r)), None)
        valid = [r['budget'] for r in results if r['id'] == q['id'] and any(matches(s) and s['status'] == 'valid' for s in r['scores'])]
        coverage.append({'id': q['id'], 'best_pool_rank': rank, 'budgets_valid_gt': sorted(valid)})
    report = {'status': 'completed', 'model_load_seconds': load_seconds, 'actual_wall_seconds': perf_counter() - wall,
              'runs': [{k: v for k, v in r.items() if k not in ('scores', 'image_reads')} for r in results],
              'budgets': budgets, 'summary': summarize(results, budgets), 'gt_coverage': coverage, 'source_and_inputs_unchanged': True}
    save(output / 'REPORT.json', report)
    write_report(output, report)
    print(json.dumps({'status': 'completed', 'summary': report['summary'], 'actual_wall_seconds': report['actual_wall_seconds']}), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--budgets', nargs='+', type=int, default=BUDGETS)
    args = parser.parse_args()
    run(args.output, args.budgets)
