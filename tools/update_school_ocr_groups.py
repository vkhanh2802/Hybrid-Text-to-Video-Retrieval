"""Install only new L01/L02 into a derived snapshot; preserve every old asset.

No OCR inference/package installation, no DRES/web restart, no worker retirement.
Creates a new CPU worker from the previously deployed immutable source.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess

from kis.school_ocr import mappings
from tools.import_school_ocr import zip_records
from tools.transfer_school_data import remote_python, ssh_options


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


REMOTE_INSTALL = r'''
import pathlib,json,hashlib,sys,shutil,os,socket,subprocess,copy
cfg=CONFIG
base=pathlib.Path(cfg['root']);previous=pathlib.Path(cfg['previous_snapshot'])
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(previous/'MANIFEST.json')==cfg['previous_manifest_sha256']
old_manifest=json.loads((previous/'MANIFEST.json').read_text())
old_hashes={'MANIFEST.json':sha(previous/'MANIFEST.json')}
for group,row in old_manifest['groups'].items():
 assert row['path']==group+'.json'
 assert sha(previous/row['path'])==row['sha256']
 old_hashes[row['path']]=row['sha256']
for name,expected in cfg['source_sha256'].items():
 assert sha(pathlib.Path(cfg['previous_worker_source'])/name)==expected,name
sys.path.insert(0,cfg['previous_worker_source'])
from kis.school_ocr import mappings,build_snapshot
from tools_placeholder import unused
'''

# Keep the remote source standalone: use the audited existing importer source
# uploaded as bytes, never execute code from supplied OCR ZIPs.
REMOTE_INSTALL = REMOTE_INSTALL.replace('from tools_placeholder import unused', r'''
import importlib.util
spec=importlib.util.spec_from_file_location('checked_importer',base/'import_school_ocr.py')
assert sha(base/'import_school_ocr.py')==cfg['importer_sha256']
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
dataset=base/'dataset';dataset.mkdir()
group_counts={}
for group,expected_archive in cfg['archive_sha256'].items():
 archive=base/'archives'/(group+'.zip');assert sha(archive)==expected_archive
 expected,mapping_hashes=mappings(pathlib.Path(cfg['dataset'])/'mapping',group)
 assert mapping_hashes==cfg['mapping_sha256'][group]
 records=module.zip_records(archive,group,expected)
 for name,raw in records:
  out=dataset/'ocr'/name;out.parent.mkdir(parents=True,exist_ok=True)
  with out.open('xb') as f:f.write(raw)
  assert out.read_bytes()==raw
 for video,h in mapping_hashes.items():
  candidates=list((pathlib.Path(cfg['dataset'])/'mapping'/group).rglob(video+'.csv'))
  assert len(candidates)==1 and sha(candidates[0])==h
  out=dataset/'mapping'/group/video/(video+'.csv');out.parent.mkdir(parents=True,exist_ok=True)
  with out.open('xb') as f:f.write(candidates[0].read_bytes())
 group_counts[group]=len(records)
subset=base/'updated-groups-snapshot'
build_snapshot(dataset,tuple(cfg['archive_sha256']),subset)
updated=json.loads((subset/'MANIFEST.json').read_text())
manifest=copy.deepcopy(old_manifest);snapshot=base/'snapshot';snapshot.mkdir()
for group,row in manifest['groups'].items():
 if group in updated['groups']:
  row=updated['groups'][group];manifest['groups'][group]=row;source=subset/row['path']
 else:source=previous/row['path']
 with (snapshot/row['path']).open('xb') as f:f.write(source.read_bytes())
 assert sha(snapshot/row['path'])==row['sha256']
with (snapshot/'MANIFEST.json').open('x') as f:json.dump(manifest,f,indent=2)
assert old_hashes=={name:sha(previous/name) for name in old_hashes}
report={'root':str(base),'snapshot':str(snapshot),'manifest_sha256':sha(snapshot/'MANIFEST.json'),
 'updated_groups':group_counts,'all_groups':{g:{k:v for k,v in row.items() if k in ['sha256','records','status_counts']} for g,row in manifest['groups'].items()},
 'previous_snapshot_unchanged':True,'unchanged_groups_byte_identical':[g for g in manifest['groups'] if g not in updated['groups']],
 'original_json_bytes_preserved':True,'ocr_inference_or_install':False}
with (base/'IMPORT_REPORT.json').open('x') as f:json.dump(report,f,indent=2)
# New worker, same public source/config/models, new snapshot pin only.
s=socket.socket();s.bind(('127.0.0.1',cfg['remote_port']));s.close()
worker=base/'worker';worker.mkdir();source=worker/'source';source.mkdir()
for name in cfg['worker_members']:
 origin=pathlib.Path(cfg['previous_worker_source'])/name
 out=source/name;out.parent.mkdir(parents=True,exist_ok=True)
 with out.open('xb') as f:f.write(origin.read_bytes())
 assert sha(out)==cfg['source_sha256'][name]
(source/'dataset').symlink_to(pathlib.Path(cfg['dataset']),target_is_directory=True)
token=worker/'worker.token';fd=os.open(token,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
with os.fdopen(fd,'w') as f:f.write(cfg['token'])
env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',FAISS_OPT_LEVEL='generic')
for name in list(env):
 if name.startswith(('KIS_OCR_','GROQ_','GEMINI_','OPENAI_','GOOGLE_APPLICATION_','KIS_LLM_','KIS_WORKER_')):env.pop(name)
env.update(KIS_OCR_FORMAT='school_snapshot',KIS_OCR_MODE='retrieve',KIS_OCR_ROOT=str(snapshot),
 KIS_OCR_MANIFEST_SHA256=report['manifest_sha256'],KIS_OCR_WEIGHT=str(cfg['ocr_weight']),
 KIS_VIDEO_GROUPS=','.join(cfg['video_groups']),KIS_RETRIEVAL_PROFILE=cfg['retrieval_profile'])
read_fd,write_fd=os.pipe()
with (worker/'worker.log').open('x') as log:
 process=subprocess.Popen([os.path.expanduser('~/KISBranche/.venv/bin/python'),'-u','-m','kis.school_worker',
 '--dataset',cfg['dataset'],'--token-file',str(token),'--port',str(cfg['remote_port']),'--ready-fd',str(write_fd)],
 cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
 start_new_session=True,pass_fds=(write_fd,))
os.close(write_fd)
with os.fdopen(read_fd) as f:line=f.readline()
if not line:raise RuntimeError('New worker failed startup; previous workers untouched. Inspect '+str(worker/'worker.log'))
ready=json.loads(line);assert ready.get('state')=='ready',ready
with (worker/'WORKER_READY.json').open('x') as f:json.dump(ready,f,indent=2)
report['worker']=ready;report['worker_root']=str(worker)
print(json.dumps(report))
''')


def install(output, settings, target=None, remote_port=8016, local_port=18006):
    output,settings=Path(output),Path(settings)
    if settings.exists():raise FileExistsError('Existing private staging settings retained')
    if not target:
        raise ValueError("install() needs target='user@school-host'")
    output.mkdir(parents=True,exist_ok=False)
    old=json.loads(Path('config/school-runtime.local.json').read_text(encoding='utf-8'))
    pipeline=json.loads(Path('config/pipeline.json').read_text(encoding='utf-8'))
    archive_hashes={};mapping_hashes={}
    for group in ['L01','L02']:
        archive=Path('dataset/ocr')/(group+'.zip')
        expected,mapping_hashes[group]=mappings(Path('dataset/mapping'),group)
        zip_records(archive,group,expected)
        archive_hashes[group]=digest(archive)
    members=[p.as_posix() for p in Path('kis').glob('*.py') if p.name!='local_images.py']
    # Clone only the previously deployed allowlist; local hashes gate equality.
    members+=['config/models.json','config/pipeline.json','config/object_labels.json']
    source_hashes={name:digest(name) for name in members}
    name='kis-ocr-l01-l02-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prepared=remote_python(target,"import pathlib,json\nbase=pathlib.Path.home()/"+repr(name)+
        "\nbase.mkdir(mode=0o700);(base/'archives').mkdir()\nprint(json.dumps({'root':str(base)}))")
    config=dict(prepared,previous_snapshot=old['ocr_root'],previous_manifest_sha256=old['ocr_manifest_sha256'],
        previous_worker_source=old['remote_root']+'/source',source_sha256=source_hashes,worker_members=members,
        archive_sha256=archive_hashes,mapping_sha256=mapping_hashes,importer_sha256=digest('tools/import_school_ocr.py'),
        dataset=os.environ['KIS_SCHOOL_DATASET'],remote_port=remote_port,
        token=secrets.token_urlsafe(48),ocr_weight=old['ocr_weight'],video_groups=pipeline['video_groups'],
        retrieval_profile=pipeline['retrieval_profile'])
    batch=''.join(f'put "{Path("dataset/ocr",g+".zip").resolve().as_posix()}" "{config["root"]}/archives/{g}.zip"\n' for g in archive_hashes)
    batch+=f'put "{Path("tools/import_school_ocr.py").resolve().as_posix()}" "{config["root"]}/import_school_ocr.py"\n'
    subprocess.run(['sftp','-q',*ssh_options(),'-b','-',target],input=batch,text=True,check=True)
    # Preserve exact remote program and public inputs for audit, never token.
    with (output/'REMOTE_INSTALL.py').open('x',encoding='utf-8') as f:f.write(REMOTE_INSTALL)
    with (output/'INPUTS.json').open('x',encoding='utf-8') as f:json.dump({k:v for k,v in config.items() if k!='token'},f,indent=2)
    result=remote_python(target,REMOTE_INSTALL.replace('CONFIG',repr(config),1))
    runtime=dict(url=f'http://127.0.0.1:{local_port}',token=config['token'],target=target,
        remote_port=remote_port,local_port=local_port,remote_root=result['worker_root'],
        worker_pid=result['worker']['pid'],vlm_batch_size=12,ocr_format='school_snapshot',ocr_mode='retrieve',
        ocr_root=result['snapshot'],ocr_manifest_sha256=result['manifest_sha256'],ocr_weight=old['ocr_weight'])
    with settings.open('x',encoding='utf-8') as f:json.dump(runtime,f,indent=2)
    with (output/'INSTALLED.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    assert archive_hashes=={g:digest(Path('dataset/ocr')/(g+'.zip')) for g in archive_hashes}
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True);parser.add_argument('--settings',required=True)
    args=parser.parse_args();install(args.output,args.settings)
