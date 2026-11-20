"""Create an isolated, resident CPU worker without packages or credential copies."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import subprocess
import zipfile

from tools.legacy.transfer_school_data import ROOT, CONFIG_FILES, remote_python, sha256_file, ssh_options


BOOTSTRAP = r'''
import pathlib,json,hashlib,zipfile,os,subprocess,stat
cfg=CONFIG
base=pathlib.Path(cfg['destination']); archive=base/'source.zip.partial'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==cfg['source_sha256']
archive.rename(base/'source.zip'); source=base/'source'; source.mkdir(mode=0o700)
with zipfile.ZipFile(base/'source.zip') as z:
    assert set(z.namelist())==set(cfg['members'])
    for member in z.infolist():
        p=pathlib.PurePosixPath(member.filename)
        assert not p.is_absolute() and '..' not in p.parts and not stat.S_ISLNK(member.external_attr>>16)
        out=source.joinpath(*p.parts);out.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        with out.open('xb') as stream:stream.write(z.read(member))
(source/'dataset').symlink_to(pathlib.Path(cfg['dataset']),target_is_directory=True)
token_file=base/'worker.token'
fd=os.open(token_file,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as stream:stream.write(cfg['token'])
env=dict(os.environ,PYTHONPATH=str(source),CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',FAISS_OPT_LEVEL='generic')
for key in list(env):
    if key.startswith('KIS_OCR_'):env.pop(key)
env.update(cfg.get('runtime_env',{}))
for name in list(env):
    if name.startswith(('GROQ_','GEMINI_','OPENAI_','GOOGLE_APPLICATION_','KIS_LLM_','KIS_WORKER_')):env.pop(name)
read_fd,write_fd=os.pipe()
with (base/'worker.log').open('x') as log:
    process=subprocess.Popen([os.path.expanduser('~/KISBranche/.venv/bin/python'),'-u','-m','kis.school_worker','--dataset',cfg['dataset'],'--token-file',str(token_file),'--port',str(cfg['port']),'--ready-fd',str(write_fd)],cwd=source,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(write_fd,))
os.close(write_fd)
with os.fdopen(read_fd) as stream:line=stream.readline()
if not line:raise RuntimeError('Worker failed before startup; inspect isolated worker.log')
ready=json.loads(line)
if ready.get('state')!='ready':raise RuntimeError('Worker startup failed: '+str(ready))
(base/'WORKER_READY.json').write_text(json.dumps(ready,indent=2))
print(json.dumps({'destination':str(base),'worker':ready}))
'''


def deploy(args):
    output = Path(args.output)
    settings_path = Path(args.settings) if args.settings else ROOT / "config/school-runtime.local.json"
    if settings_path.exists():
        raise FileExistsError("Existing school runtime settings retained; do not overwrite an active worker")
    output.mkdir(parents=True, exist_ok=False)
    sources = [p for p in (ROOT / "kis").glob("*.py") if p.name != "local_images.py"]
    sources += [ROOT / "config" / name for name in CONFIG_FILES]
    archive = output / "source.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
        for path in sources:
            z.write(path, path.relative_to(ROOT).as_posix())
    name = "kis-worker-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    state = remote_python(args.target, "import pathlib,json,socket\n"
        f"s=socket.socket();s.bind(('127.0.0.1',{args.port}));s.close()\n"
        f"base=pathlib.Path.home()/{name!r}\nbase.mkdir(mode=0o700)\n"
        "(base/'source.zip.partial').open('xb').close()\nprint(json.dumps({'destination':str(base)}))")
    cfg = dict(state, source_sha256=sha256_file(archive), members=[p.relative_to(ROOT).as_posix() for p in sources],
               dataset=args.dataset, port=args.port, token=secrets.token_urlsafe(48))
    if args.ocr_root:
        if not args.ocr_manifest_sha256:
            raise ValueError("Snapshot deployment requires explicit manifest pin")
        cfg["runtime_env"] = {"KIS_OCR_FORMAT": "school_snapshot", "KIS_OCR_MODE": "retrieve",
                              "KIS_OCR_ROOT": args.ocr_root, "KIS_OCR_MANIFEST_SHA256": args.ocr_manifest_sha256,
                              "KIS_OCR_WEIGHT": str(args.ocr_weight)}
    selected_groups = getattr(args, 'video_groups', None)
    if selected_groups:
        from kis.school_ocr import validate_groups
        validate_groups(selected_groups)
        cfg.setdefault('runtime_env', {})['KIS_VIDEO_GROUPS'] = ','.join(selected_groups)
    (output / "STATE.json").write_text(json.dumps({k: v for k, v in cfg.items() if k != "token"}, indent=2), encoding="utf-8")
    batch = f'put -a "{archive.resolve().as_posix()}" "{cfg["destination"]}/source.zip.partial"\n'
    subprocess.run(["sftp", "-q", *ssh_options(), "-b", "-", args.target], input=batch, text=True, check=True)
    result = remote_python(args.target, BOOTSTRAP.replace("CONFIG", repr(cfg), 1))
    runtime = {"url": f"http://127.0.0.1:{args.local_port}", "token": cfg["token"],
               "target": args.target, "remote_port": args.port, "local_port": args.local_port,
                "remote_root": result["destination"], "worker_pid": result["worker"]["pid"], "vlm_batch_size": 12}
    runtime['dataset'] = args.dataset
    if args.ocr_root:
        runtime.update(ocr_format="school_snapshot", ocr_mode="retrieve", ocr_root=args.ocr_root,
                        ocr_manifest_sha256=args.ocr_manifest_sha256, ocr_weight=args.ocr_weight)
    if selected_groups:
        runtime['video_groups'] = selected_groups
    with settings_path.open("x", encoding="utf-8") as stream:
        json.dump(runtime, stream, indent=2)
    (output / "WORKER_READY.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8014)
    parser.add_argument("--local-port", type=int, default=18004)
    parser.add_argument("--settings", type=Path, help="New private staging config; existing files are never replaced")
    parser.add_argument("--ocr-root")
    parser.add_argument("--ocr-manifest-sha256")
    parser.add_argument("--ocr-weight", type=float, default=0.1)
    parser.add_argument('--video-groups', nargs='+', help='Explicit staged worker scope; does not edit pipeline.json')
    deploy(parser.parse_args())


if __name__ == "__main__":
    main()
