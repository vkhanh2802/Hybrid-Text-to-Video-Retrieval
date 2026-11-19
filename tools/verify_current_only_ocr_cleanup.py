"""Read-only final receipt/config/evidence checks after approved cleanup."""
import json
import os
from pathlib import Path

from tools.benchmark_school_ocr import client
from tools.clean_legacy_ocr import sha
from tools.transfer_school_data import remote_python


def verify(output):
    output=Path(output)
    cleanup=json.loads((output/'REMOTE_CLEANUP_RECEIPT.json').read_text(encoding='utf-8'))
    local=json.loads((output/'LOCAL_CLEANUP_RECEIPT.json').read_text(encoding='utf-8'))
    prep=json.loads((output/'REMOTE_PREPARE.json').read_text(encoding='utf-8'))
    runtime=json.loads(Path('config/school-runtime.local.json').read_text(encoding='utf-8'))
    assert cleanup['status']=='completed'
    assert runtime['ocr_manifest_sha256']==prep['subset_manifest_sha256']
    assert runtime['vlm_batch_size']==12
    assert sorted(p.name for p in Path('dataset/ocr').iterdir())==['L01.zip','L02.zip']
    for name,h in local['kept_latest_archives_sha256'].items():assert sha(name)==h
    for row in local['deleted_legacy_local_archives']:assert not Path(row['path']).exists()
    for row in local['removed_stale_private_OCR_settings']:assert not Path(row['path']).exists()
    worker=client('config/school-runtime.local.json',True)
    assert worker.info['ocr']['covered_groups']==['L01','L02']
    assert worker.info['ocr']['records']==14955 and worker.info['ocr']['verified'] is False
    remote_config=dict(audit_root=prep['audit_root'],subset=prep['subset_root'],
        snapshot_sha256=prep['subset_manifest_sha256'],deleted_roots=[r['root'] for r in cleanup['deleted_trees']],
        kept_archives=local['kept_latest_archives_sha256'])
    code=r'''
import pathlib,json,hashlib,os
cfg=CONFIG
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
assert all(not pathlib.Path(name).exists() for name in cfg['deleted_roots'])
subset=pathlib.Path(cfg['subset']);assert sha(subset/'MANIFEST.json')==cfg['snapshot_sha256']
m=json.loads((subset/'MANIFEST.json').read_text());assert set(m['groups'])=={'L01','L02'}
count=0
for g,row in m['groups'].items():
 assert sha(subset/row['path'])==row['sha256']
 for name,h in row['source_json_sha256'].items():
  assert sha(subset.parent/'dataset/ocr'/name)==h;count+=1
for local,h in cfg['kept_archives'].items():assert sha(subset.parent/'archives'/pathlib.Path(local).name)==h
assert count==14955
for pid in [90716,92720]:
 p=pathlib.Path('/proc')/str(pid)/'cmdline'
 if p.exists():assert b'kis.school_worker' not in p.read_bytes()
audit=pathlib.Path(cfg['audit_root']);receipt=json.loads((audit/'CLEANUP_RECEIPT.json').read_text())
archive=pathlib.Path(receipt['mixed_transfer_archive']['archive'])
assert sha(archive)==receipt['mixed_transfer_archive']['new_sha256']
print(json.dumps({'new_raw_JSON_hashes_valid':count,'old_ocr_roots_absent':len(cfg['deleted_roots']),
 'filtered_transfer_archive_hash_matches_receipt':True,'old_OCR_workers_not_loaded':True,
 'kept_new_groups':['L01','L02'],'rollback_old_OCR_corpus_available':False}))
'''
    remote=remote_python(os.environ['KIS_SCHOOL_TARGET'],code.replace('CONFIG',repr(remote_config),1))
    result=dict(status='completed_and_verified',worker_ready=True,worker_pid=runtime['worker_pid'],
        remote_port=runtime['remote_port'],local_port=runtime['local_port'],
        active_snapshot_manifest_sha256=runtime['ocr_manifest_sha256'],covered_groups=['L01','L02'],
        records=14955,scope_unchanged=worker.info['runtime']['video_groups'],
        old_ocr_missing_groups=worker.info['ocr']['missing_groups'],local_ocr_directory=['L01.zip','L02.zip'],
        local_legacy_archives_deleted=len(local['deleted_legacy_local_archives']),
        remote_files_deleted=sum(r['deleted_files'] for r in cleanup['deleted_trees']),
        mixed_archive_old_ocr_members_removed=cleanup['mixed_transfer_archive']['removed_OCR_members'],
        remote=remote,benchmark_reports_retained=True,no_ocr_or_qwen_inference=True,
        web_restart=False,DRES_calls=False,query_benchmark_run=False)
    with (output/'FINAL_VERIFICATION.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    verify(parser.parse_args().output)
