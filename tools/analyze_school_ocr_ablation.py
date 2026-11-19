"""Analyze a frozen A/B/C packet after GPU run, without rerunning any model."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from tools.school_ocr_accuracy import metrics, save, sha


def key(row):
    return row['video_id'],row['keyframe_n']


def valid(value):
    return type(value) in (int,float) and math.isfinite(value) and 0<=value<=10


def analyze(output):
    output=Path(output)
    packet=json.loads((output/'PACKET.json').read_text(encoding='utf-8'))
    report=json.loads((output/'REPORT.json').read_text(encoding='utf-8'))
    assert sha(output/'QUERIES.json')==packet['queries_sha256']
    assert sha(output/'PROTOCOL.json')==packet['protocol_sha256']
    assert len(packet['queries'])==len(report['records'])
    rows=[];status_changes=Counter();bc_pairs=[]
    for original,record in zip(packet['queries'],report['records']):
        assert original['query']==record['query']
        q=record['query'];arms=record['arms'];row=dict(query_id=q['id'],literal=q.get('literal'),arms={})
        for arm in ['A','B','C']:
            source=original['retrieval']['A' if arm=='A' else 'B']['pool'];info=arms[arm]
            expected=[r for r in source[:20] if original['images'][r['video_id']+'/'+str(r['keyframe_n'])]['status']=='ready']
            assert info['inputs']==len(expected)==len(info['scores'])<=20
            batch=info['batch'];assert batch['requested_batch_size']==12 and batch['oom_splits']==0
            assert all(a['status']=='ok' for a in batch['attempts'])
            assert batch['effective_batches']==[min(12,len(expected)-i) for i in range(0,len(expected),12)]
            final=info['final_ranking'];assert len(final)==len({key(r) for r in final})==100
            allowed={key(r) for r in source};first20={key(r) for r in source[:20]}
            assert all(key(r) in allowed for r in final)
            assert all(r['vlm_status'] in ['unscored','missing_image'] or key(r) in first20 for r in final)
            row['arms'][arm]=dict(pre=metrics(source,q),post=info['post_qwen'],
                inputs=len(expected),invalid_scores=sum(not valid(s) for s in info['scores']),
                qwen_seconds=info['seconds'],cpu_rpc_seconds=original['retrieval']['A' if arm=='A' else 'B']['rpc_seconds'],
                effective_batches=batch['effective_batches'],top1_identity=list(key(final[0])),
                top1_score=final[0].get('vlm_raw_score'),top1_ocr_status=(final[0].get('ocr') or {}).get('status','off'))
        b_pool=original['retrieval']['B']['pool']
        expected=[r for r in b_pool[:20] if original['images'][r['video_id']+'/'+str(r['keyframe_n'])]['status']=='ready']
        for candidate,b,c in zip(expected,arms['B']['scores'],arms['C']['scores']):
            status=(candidate.get('ocr') or {}).get('status','missing')
            if valid(b) and valid(c):
                changed=b!=c
                if changed:status_changes[status]+=1
                bc_pairs.append(dict(query_id=q['id'],video_id=candidate['video_id'],keyframe_n=candidate['keyframe_n'],
                    ocr_status=status,B=b,C=c,delta=c-b,changed=changed))
        row['ocr_hits']=original['retrieval']['B']['diagnostics']['ocr']['hits']
        row['same_A_B_top20_set']={key(r) for r in original['retrieval']['A']['pool'][:20]}=={key(r) for r in b_pool[:20]}
        rows.append(row)
    summary={}
    for arm in ['A','B','C']:
        values=[r['arms'][arm] for r in rows]
        summary[arm]=dict(queries=len(values),inputs=sum(r['inputs'] for r in values),
            invalid_scores=sum(r['invalid_scores'] for r in values),
            known_positive_post_metrics={m:sum(r['post'][m] for r in values)/len(values)
                for m in values[0]['post'] if m!='first_known_positive_rank'},
            qwen_median_seconds=statistics.median(r['qwen_seconds'] for r in values),
            qwen_mean_seconds=statistics.mean(r['qwen_seconds'] for r in values),
            qwen_median_excluding_first_query=statistics.median(r['qwen_seconds'] for r in values[1:]) if len(values)>1 else None,
            cpu_rpc_median_seconds=statistics.median(r['cpu_rpc_seconds'] for r in values),
            top1_without_valid_qwen_score=sum(not valid(r['top1_score']) for r in values))
    result=dict(schema='school_ocr_abc_analysis_v1',status='checked',rows=rows,summary=summary,
        B_C_valid_paired_scores=len(bc_pairs),B_C_changed_scores=sum(r['changed'] for r in bc_pairs),
        B_C_changed_by_ocr_status=dict(status_changes),same_A_B_top20_set_queries=sum(r['same_A_B_top20_set'] for r in rows),
        output_validation=f'All{len(rows)*3} arm/query outputs have100 distinct supplied real frame identities; Qwen only original top20, batch12, no OOM split or backfill.',
        limitations=['Small previously OCR-reviewed development pilot, not human ground truth or heldout.',
            'Known-positive anchors sparse; unjudged/neighbor frames not negatives; not precision/accuracy.',
            'Same B/C pool/order/images; A/B ordering can differ even when top20 sets equal.',
            'CPU RPC and Qwen inference only; no live planner/video/image-transfer/render timing or general end-to-end speedup.',
            'First inference may include warmup; excluding first query is descriptive, not a controlled latency benchmark.',
            'Legacy runner REPORT.json has a static six-query limitation sentence; this packet actually contains8 queries as validated here.',
            'Top1 visual judgments must be written before decoding arm labels for supplementary relevance assessment.'])
    save(output/'ANALYSIS.json',result);save(output/'B_C_PAIRED_SCORES.json',bc_pairs)
    print(json.dumps({'summary':summary,'B_C_changed_scores':result['B_C_changed_scores'],
        'B_C_changed_by_ocr_status':dict(status_changes),'same_A_B_top20_set_queries':result['same_A_B_top20_set_queries']},ensure_ascii=True))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    analyze(parser.parse_args().output)
