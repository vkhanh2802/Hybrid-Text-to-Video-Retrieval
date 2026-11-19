"""Read-only post-run integrity, Markdown tables and visual-review contact sheet."""
from collections import Counter
import json
from pathlib import Path
import statistics
import zipfile

from tools.benchmark_user_five_ocr import INPUT, ROOT, save, sha


def report(output):
    output = Path(output)
    result = json.loads((output / 'REPORT.json').read_text(encoding='utf-8'))
    data = json.loads((output / 'QUERIES.json').read_text(encoding='utf-8'))
    protocol = json.loads((output / 'PROTOCOL.json').read_text(encoding='utf-8'))
    packets = {q['id']: json.loads((output / f"{q['id']}_RETRIEVAL.json").read_text(encoding='utf-8')) for q in data['queries']}
    rows = {(q['id'], arm): json.loads((output / f"{q['id']}_{arm}.json").read_text(encoding='utf-8'))
            for q in data['queries'] for arm in 'ABC'}
    checks, per_query = [], []
    for q in data['queries']:
        packet = packets[q['id']]
        bpool = packet['retrieval']['B']['candidates']
        apool = packet['retrieval']['A']['candidates']
        a20 = {(r['video_id'], r['frame_id']) for r in apool[:20]}
        b20 = {(r['video_id'], r['frame_id']) for r in bpool[:20]}
        bc = [rows[q['id'], arm] for arm in 'BC']
        assert bc[0]['qwen_calls'][0]['ocr_argument'] is False
        assert bc[1]['qwen_calls'][0]['ocr_argument'] == (bc[1].get('ocr_prompt_candidates', 1) > 0)
        signatures = [[(r['video_id'], r['keyframe_n'], r.get('sha256'), r['success']) for r in v['image_reads']] for v in bc]
        assert signatures[0] == signatures[1]
        per_query.append({'id': q['id'], 'A_B_top20_intersection': len(a20 & b20),
                          'A_B_top100_intersection': len({(r['video_id'], r['frame_id']) for r in apool[:100]} &
                                                       {(r['video_id'], r['frame_id']) for r in bpool[:100]}),
                          'B_top20_ocr_status': dict(Counter(r.get('ocr', {}).get('status', 'absent') for r in bpool[:20])),
                          'ocr_retrieval_diagnostics': packet['retrieval']['B']['diagnostics'].get('ocr'),
                          'planner_needs_ocr': packet['structured'].get('needs_ocr'),
                          'planner_ocr_text': packet['structured'].get('ocr_text'),
                          'B_C_input_images_identical': True})
        for arm in 'ABC':
            row = rows[q['id'], arm]
            requested = [(r['video_id'], r['keyframe_n']) for r in (apool if arm == 'A' else bpool)[:20]]
            actual = [(r['video_id'], r['keyframe_n']) for r in row['image_reads']]
            assert actual == requested
            assert all(c['diagnostics']['requested_batch_size'] == 12 for c in row['qwen_calls'])
            checks.append({'id': q['id'], 'arm': arm, 'distinct_100': len({(r['video_id'], r['frame_id']) for r in row['results']}) == 100,
                           'original_top20_only': True, 'requested_batch12': True,
                           'effective_batches': row['qwen_calls'][0]['diagnostics']['effective_batches'],
                           'oom_splits': row['qwen_calls'][0]['diagnostics']['oom_splits']})
    assert all(c['distinct_100'] for c in checks)
    assert all(sha(ROOT / p) == h for p, h in protocol['source_sha256'].items())
    audit = {'status': 'passed', 'checks': checks, 'per_query': per_query,
             'source_unchanged': True, 'query_input_matches_frozen': sha(INPUT) == protocol['source_sha256'][INPUT.relative_to(ROOT).as_posix()]}
    save(output / 'POSTRUN_AUDIT.json', audit)

    text = ['# Benchmark 5 query người dùng — full pipeline OCR ablation', '',
            '## Phạm vi', '',
            '- A: OCR off; B: OCR retrieval, không OCR trong prompt; C: cùng pool B, OCR prompt theo policy ghi trong PROTOCOL.',
            '- Planner Groq thật một operation/query (production có thể repair), không đưa GT vào pipeline. Query giữ nguyên lỗi gõ.',
            f"- CPU corpus đầy đủ {len(protocol['scope'])} nhóm; OCR active {protocol['worker_current']['ocr']['covered_groups']}, {protocol['worker_current']['ocr']['records']} record. Qwen GPU batch12, top20 gốc, không backfill.",
            '- Gọi các hàm production trực tiếp, không HTTP web server; không restart web, không DRES.', '',
            '## Chất lượng theo ground truth cung cấp', '',
            'Rank tốt nhất trong các keyframe được chấp nhận; `—` = không có trong top100. Các frame chưa gán nhãn không được kết luận sai.', '',
            '| Query | A trước → sau Qwen | B trước → sau Qwen | C trước → sau Qwen |',
            '|---|---:|---:|---:|']
    bykey = {(r['id'], r['arm']): r for r in result['runs']}
    fmt = lambda x: str(x) if x is not None and x <= 100 else '—'
    for q in data['queries']:
        values = [f"{fmt(bykey[q['id'], a]['before']['best_rank'])} → {fmt(bykey[q['id'], a]['after']['best_rank'])}" for a in 'ABC']
        text.append(f"| {q['id']} | " + ' | '.join(values) + ' |')
    text += ['', '| Sau Qwen | Hit@1 | Hit@5 | Hit@20 | Hit@100 | MRR@100 |', '|---|---:|---:|---:|---:|---:|']
    for a in 'ABC':
        v = result['aggregate'][a]['after']
        text.append(f"| {a} | " + ' | '.join(f"{v[k]*100:.0f}%" for k in ('hit_at_1', 'hit_at_5', 'hit_at_20', 'hit_at_100')) + f" | {v['mrr_at_100']:.4f} |")
    text += ['', '## Thời gian theo query/nhánh (giây)', '',
             'Tổng logic = planner thực đo + RPC retrieval + rerank (gồm lấy ảnh) + serialize. Planner chia sẻ A/B/C; retrieval B chia sẻ C. Không cộng các tổng logic thành wall time.', '',
             '| Query | Nhánh | Planner | CPU RPC | Lấy ảnh | Qwen | Tổng logic | Ảnh thiếu / score invalid-error |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    for q in data['queries']:
        for a in 'ABC':
            row = bykey[q['id'], a]
            t, s = row['timing'], row['top20_status']
            text.append(f"| {q['id']} | {a} | " + ' | '.join(f"{t[k]:.2f}" for k in ('planner', 'retrieval_rpc', 'image_fetch', 'qwen', 'logical_e2e')) +
                        f" | {s.get('missing_image',0)} / {s.get('invalid_score',0)+s.get('error',0)} |")
    text += ['', f"- Nạp Qwen cold, tách riêng: **{result['model_load_seconds']:.2f}s**.",
             f"- Wall time toàn bộ chương trình (imports/preflight, 5 planner, 10 retrieval, nạp model, 15 Qwen, audit writes): **{result['actual_wall_seconds']:.2f}s**.",
             '- Không bao gồm tải/render grid 100 ảnh trên browser; không đo HTTP endpoint overhead.', '',
             '| Median theo nhánh | CPU RPC | Lấy ảnh | Qwen | Tổng logic |', '|---|---:|---:|---:|---:|']
    for a in 'ABC':
        t = result['aggregate'][a]['timing_median']
        text.append(f"| {a} | " + ' | '.join(f"{t[k]:.2f}" for k in ('retrieval_rpc', 'image_fetch', 'qwen', 'logical_e2e')) + ' |')
    text += ['', '## Router/OCR và kiểm chứng', '']
    for q in per_query:
        text += [f"- {q['id']}: planner needs_ocr=`{q['planner_needs_ocr']}`, ocr_text={json.dumps(q['planner_ocr_text'],ensure_ascii=False)}; "
                 f"A/B trùng {q['A_B_top20_intersection']}/20 và {q['A_B_top100_intersection']}/100. "
                 f"OCR B top20: `{q['B_top20_ocr_status']}`."]
    text += ['', '- Mỗi nhánh trả đúng 100 frame identity distinct và đã đối chiếu CSV mapping; ảnh B/C giống hash, không backfill; source hash không đổi.',
             f"- Tổng OOM split: {sum(c['oom_splits'] for c in checks)}; effective batches chi tiết: `POSTRUN_AUDIT.json`.", '',
             '## Giới hạn', '', *['- ' + s for s in result['limitations']], '',
             '## Query và GT đã đóng băng', '']
    for q in data['queries']:
        text += [f"### {q['id']}", q['query'], '', f"GT supplied: `{q['supplied_ground_truth']}`; chuẩn hóa `{q['video_id']}` keyframe {q['keyframes']}. Kết quả mapping xem `GROUND_TRUTH_JOINS.json`.", '']
    (output / 'REPORT.md').write_text('\n'.join(text), encoding='utf-8')

    # Contact sheet: GT reference beside A/B/C top1. Not fed into any model.
    from PIL import Image, ImageDraw
    from io import BytesIO
    width, height = 400, 275
    sheet = Image.new('RGB', (4 * width, len(data['queries']) * height), '#eeeeee')
    draw = ImageDraw.Draw(sheet)
    for i, q in enumerate(data['queries']):
        refs = [(q['video_id'], q['keyframes'][0], 'GT')]
        refs += [(rows[q['id'], a]['results'][0]['video_id'], rows[q['id'], a]['results'][0]['keyframe_n'], a) for a in 'ABC']
        for j, (vid, n, label) in enumerate(refs):
            path = output / 'images' / vid / f'{n:03d}.jpg'
            image = None
            if path.is_file():
                image = Image.open(path).convert('RGB')
            else:
                # Local reference ZIP only. Never replace missing production images for scoring.
                zpath = ROOT / 'dataset/data' / f"{vid.split('_')[0]}.zip"
                if label == 'GT' and zpath.is_file():
                    with zipfile.ZipFile(zpath) as z:
                        matches = [name for name in z.namelist() if name.endswith(f'/{vid}/{n:03d}.jpg')]
                        if len(matches) == 1:
                            image = Image.open(BytesIO(z.read(matches[0]))).convert('RGB')
            if image:
                image.thumbnail((width, height - 35))
                sheet.paste(image, (j * width, i * height + 35))
            draw.text((j * width + 5, i * height + 6), f"{q['id']} {label}: {vid}/{n:03d}" + ('' if image else ' MISSING'), fill='black')
    sheet.save(output / 'VISUAL_REVIEW.jpg', quality=92)
    print(json.dumps({'audit': 'passed', 'report': str(output / 'REPORT.md'), 'visual_review': str(output / 'VISUAL_REVIEW.jpg')}, ensure_ascii=True))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    report(p.parse_args().output)
