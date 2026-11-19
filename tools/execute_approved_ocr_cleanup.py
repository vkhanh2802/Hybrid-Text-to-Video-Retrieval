"""Execute this session's exact, user-approved OCR-only deletion plan.

Not a broad recursive cleaner. Requires a hash-pinned plan, current-only worker
already validated/active, and the existing audited helper on the school CPU.
"""
import json
import os
from pathlib import Path

from tools.transfer_school_data import remote_python


REMOTE = r'''
import pathlib,json,hashlib,importlib.util,urllib.request
cfg=CONFIG;audit=pathlib.Path(cfg['audit_root']);helper=audit/'clean_legacy_ocr.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest()==cfg['helper_sha256']
spec=importlib.util.spec_from_file_location('cleanup',helper);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
plan_path=audit/'DELETE_PLAN.json';assert mod.sha(plan_path)==cfg['plan_sha256']
plan=json.loads(plan_path.read_text())
assert plan['helper_sha256']==cfg['helper_sha256']
for name,h in plan['protected_files'].items():assert mod.sha(pathlib.Path(name))==h
for tree in plan['trees']:assert mod.inventory_tree(tree['root'],plan['allowed_roots'])==tree
token=pathlib.Path(cfg['worker_root'])/'worker.token'
req=urllib.request.Request('http://127.0.0.1:8017/status',headers={'Authorization':'Bearer '+token.read_text().strip()})
with urllib.request.urlopen(req,timeout=20) as response:status=json.load(response)
assert status['state']=='ready' and status['ocr']['covered_groups']==['L01','L02']
assert status['ocr']['records']==14955 and status['ocr']['build_id']==plan['subset_manifest_sha256']
for pid in [90716,92720]:
 p=pathlib.Path('/proc')/str(pid)/'cmdline'
 if p.exists():assert b'kis.school_worker' not in p.read_bytes(),'obsolete OCR worker still loaded'
archive=pathlib.Path(plan['mixed_archive']['path']);assert mod.sha(archive)==plan['mixed_archive']['sha256']
filtered=mod.strip_legacy_ocr_from_transfer(archive,audit/'mixed-archive-filter')
deleted=[]
for tree in plan['trees']:
 receipt=mod.delete_inventoried_tree(tree,plan['allowed_roots']);deleted.append(receipt)
 with (audit/'DELETED_TREES.jsonl').open('a') as stream:stream.write(json.dumps(receipt)+'\n')
for name,h in plan['protected_files'].items():assert mod.sha(pathlib.Path(name))==h
subset=pathlib.Path(plan['subset_root']);manifest=json.loads((subset/'MANIFEST.json').read_text())
assert set(manifest['groups'])=={'L01','L02'}
checked=0
for group,row in manifest['groups'].items():
 for name,h in row['source_json_sha256'].items():
  assert mod.sha(pathlib.Path(plan['protected_raw_root'])/name)==h;checked+=1
assert checked==14955
with urllib.request.urlopen(req,timeout=20) as response:after=json.load(response)
assert after['state']=='ready' and after['ocr']['records']==14955
receipt={'status':'completed','deleted_trees':deleted,'mixed_transfer_archive':filtered,
 'protected_new_raw_JSON_records_checked':checked,'new_snapshot_manifest_sha256':plan['subset_manifest_sha256'],
 'new_worker_ready':True,'old_OCR_rollback_removed':True,'reports_captions_mapping_features_images_models_retained':True,
 'web_restart':False,'DRES_calls':False,'query_benchmark_run':False}
with (audit/'CLEANUP_RECEIPT.json').open('x') as stream:json.dump(receipt,stream,indent=2)
print(json.dumps(receipt))
'''


def run(output):
    output=Path(output)
    prep=json.loads((output/'REMOTE_PREPARE.json').read_text(encoding='utf-8'))
    plan=json.loads((output/'REMOTE_DELETE_PLAN_SUMMARY.json').read_text(encoding='utf-8'))
    runtime=json.loads(Path('config/school-runtime.local.json').read_text(encoding='utf-8'))
    from tools.clean_legacy_ocr import sha
    cfg=dict(audit_root=prep['audit_root'],plan_sha256=plan['plan_sha256'],
             helper_sha256=sha('tools/clean_legacy_ocr.py'),worker_root=runtime['remote_root'])
    assert runtime['remote_port']==8017
    assert runtime['ocr_manifest_sha256']==prep['subset_manifest_sha256']
    result=remote_python(os.environ['KIS_SCHOOL_TARGET'],REMOTE.replace('CONFIG',repr(cfg),1))
    with (output/'REMOTE_CLEANUP_RECEIPT.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
