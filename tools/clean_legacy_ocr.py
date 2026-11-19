"""Narrow deletion helpers for explicitly approved legacy OCR cleanup.

No model/network calls. Callers supply exact allowed roots and expected hashes.
Mixed transfer archives retain every non-OCR payload byte, with a new manifest.
"""
import copy
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import uuid


def stream_sha(stream):
    result=hashlib.sha256()
    for block in iter(lambda:stream.read(1024*1024),b''):
        result.update(block)
    return result.hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return stream_sha(stream)


def checked_root(root,allowed):
    root=Path(root)
    if not root.is_absolute() or str(root) not in {str(Path(p)) for p in allowed}:
        raise ValueError('Deletion root not explicitly allowlisted')
    if root.is_symlink() or root.resolve()!=root or not root.is_dir():
        raise ValueError('Deletion root must be an existing nonsymlink canonical directory')
    return root


def inventory_tree(root,allowed):
    root=checked_root(root,allowed);rows=[]
    for base,dirs,files in os.walk(root,followlinks=False):
        for name in dirs+files:
            p=Path(base)/name
            if p.is_symlink():raise ValueError('Symlink inside deletion root')
        for name in sorted(files):
            p=Path(base)/name
            if not p.is_file() or p.suffix not in {'.json','.zip','.sqlite','.sqlite3'}:
                raise ValueError('Unexpected non-OCR file type in deletion root')
            rows.append(dict(relative_path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p)))
    return dict(root=str(root),files=sorted(rows,key=lambda r:r['relative_path']))


def delete_inventoried_tree(plan,allowed):
    root=checked_root(plan['root'],allowed)
    if inventory_tree(root,allowed)!=plan:
        raise ValueError('Deletion tree changed since inventory; refusing removal')
    for row in plan['files']:
        (root/row['relative_path']).unlink()
    for base,dirs,files in os.walk(root,topdown=False,followlinks=False):
        Path(base).rmdir()  # Fails on unplanned new content, never broad rmtree.
    return dict(root=str(root),deleted_files=len(plan['files']),deleted_bytes=sum(r['bytes'] for r in plan['files']))


class HashReader:
    def __init__(self,stream):self.stream=stream;self.digest=hashlib.sha256()
    def read(self,size=-1):
        raw=self.stream.read(size);self.digest.update(raw);return raw


def strip_legacy_ocr_from_transfer(archive,audit_directory):
    """Replace a known mixed transfer archive only after content verification.

    Kept payloads are copied and checked against the original transfer manifest.
    Only dataset/ocr/* is removed. Reports live outside the data archive.
    """
    archive=Path(archive);audit=Path(audit_directory)
    if archive.is_symlink() or not archive.is_file():raise ValueError('Expected nonsymlink transfer archive')
    audit.mkdir(parents=True,exist_ok=False)
    original_hash=sha(archive);before=archive.stat()
    temporary=archive.with_name('.no-legacy-ocr-'+uuid.uuid4().hex+'.tar.gz')
    seen=set();kept={};removed=[];original_manifest=None;new_manifest=None
    try:
        with tarfile.open(archive,'r|gz') as source, tarfile.open(temporary,'x:gz',compresslevel=1) as target:
            for member in source:
                p=PurePosixPath(member.name)
                if (not member.isfile() or p.is_absolute() or '..' in p.parts or '\\' in member.name
                        or member.name in seen):raise ValueError('Unsafe or duplicate transfer archive member')
                seen.add(member.name)
                if member.name=='TRANSFER_MANIFEST.json':
                    original_manifest=json.load(source.extractfile(member));continue
                if member.name.startswith('dataset/ocr/'):
                    removed.append(member.name);continue
                if p.parts[0] not in {'dataset','config'}:raise ValueError('Unexpected transfer payload root')
                with source.extractfile(member) as stream:
                    reader=HashReader(stream);target.addfile(member,reader)
                    kept[member.name]=dict(bytes=member.size,sha256=reader.digest.hexdigest())
            if original_manifest is None:raise ValueError('Missing transfer manifest')
            files=original_manifest['files']
            if len({r['path'] for r in files})!=len(files):raise ValueError('Duplicate manifest entries')
            original_paths={r['path'] for r in files}
            if seen!=original_paths|{'TRANSFER_MANIFEST.json'}:raise ValueError('Source archive manifest mismatch')
            remaining=[r for r in files if not r['path'].startswith('dataset/ocr/')]
            if set(kept)!={r['path'] for r in remaining}:raise ValueError('Unexpected kept payload set')
            for row in remaining:
                if kept[row['path']]!={'bytes':row['size'],'sha256':row['sha256']}:
                    raise ValueError('Source non-OCR payload hash differs from transfer manifest')
            new_manifest=copy.deepcopy(original_manifest)
            new_manifest.update(files=remaining,file_count=len(remaining),bytes=sum(r['size'] for r in remaining),
                legacy_ocr_removed=True,derived_from_archive_sha256=original_hash)
            new_manifest['components'].pop('ocr',None)
            raw=json.dumps(new_manifest,ensure_ascii=True,separators=(',',':')).encode('utf-8')
            info=tarfile.TarInfo('TRANSFER_MANIFEST.json');info.size=len(raw);info.mode=0o600
            target.addfile(info,io.BytesIO(raw))
        checked=set()
        with tarfile.open(temporary,'r|gz') as check:
            for member in check:
                if member.name in checked or member.name.startswith('dataset/ocr/'):
                    raise ValueError('Filtered archive contains duplicate/legacy OCR')
                checked.add(member.name)
                if member.name=='TRANSFER_MANIFEST.json':
                    if json.load(check.extractfile(member))!=new_manifest:raise ValueError('Filtered manifest mismatch')
                else:
                    with check.extractfile(member) as stream:h=stream_sha(stream)
                    if kept.get(member.name)!={'bytes':member.size,'sha256':h}:
                        raise ValueError('Filtered non-OCR payload changed')
        if checked!=set(kept)|{'TRANSFER_MANIFEST.json'}:raise ValueError('Filtered archive incomplete')
        if (archive.stat().st_size,archive.stat().st_mtime_ns)!=(before.st_size,before.st_mtime_ns):
            raise ValueError('Original archive changed during filtering')
        with (audit/'ORIGINAL_TRANSFER_MANIFEST.json').open('x',encoding='utf-8') as f:
            json.dump(original_manifest,f,ensure_ascii=True,separators=(',',':'))
        receipt=dict(archive=str(archive),original_sha256=original_hash,new_sha256=sha(temporary),
            removed_OCR_members=len(removed),retained_non_OCR_payloads=len(kept),
            all_retained_payload_bytes_sha256_match=True,old_archive_retained=False)
        os.replace(temporary,archive)  # Explicitly authorized removal of old OCR, including packed copies.
        with (audit/'RECEIPT.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
        return receipt
    finally:
        temporary.unlink(missing_ok=True)  # Only this invocation's own temporary file.
