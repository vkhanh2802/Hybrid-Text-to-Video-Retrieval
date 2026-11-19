"""Read-only supplied public ZIP audit; no deployment, network or inference."""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

from kis.school_ocr import mappings
from tools.import_school_ocr import zip_records
from tools.audit_ocr_vietnamese_pair import sha, write


def metrics(rows):
    texts = [r['ocr_text'] for r in rows.values()]
    chars = Counter(''.join(texts))
    return dict(records=len(rows), videos=len({r['video_id'] for r in rows.values()}),
        statuses=dict(Counter(r['status'] for r in rows.values())),
        characters=sum(map(len,texts)), lines=sum(len(t.splitlines()) for t in texts),
        replacement_characters=chars['\ufffd'],
        exact_HTV9_records=sum(bool(re.search(r'(?<![A-Za-z0-9])HTV9(?![A-Za-z0-9])',t)) for t in texts),
        probe_counts={c:chars[c] for c in 'ảấứừếộđĐ'})


def comparisons(new, old):
    assert set(new) == set(old)
    pairs=[dict(member=name,old_text=old[name]['ocr_text'],new_text=new[name]['ocr_text'],
                old_status=old[name]['status'],new_status=new[name]['status']) for name in sorted(new)]
    return dict(paired=len(pairs),identical_text=sum(p['old_text']==p['new_text'] for p in pairs),
        new_shorter=sum(len(p['new_text'])<len(p['old_text']) for p in pairs),
        new_longer=sum(len(p['new_text'])>len(p['old_text']) for p in pairs),
        old_nonempty_new_empty=sum(bool(p['old_text']) and not p['new_text'] for p in pairs)), pairs


def run(output):
    from PIL import Image, ImageDraw
    from io import BytesIO
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    previous=Path('benchmark_results/ocr_vietnamese_pair_20261115_run01')
    old_pairs=json.loads((previous/'PAIRS.json').read_text(encoding='utf-8'))
    previously_reviewed={r['member'] for r in json.loads((previous/'SAMPLE_PAIRS.json').read_text(encoding='utf-8'))}
    l01_anchors=['L01/L01_V005/069.json','L01/L01_V013/170.json','L01/L01_V015/196.json',
                 'L01/L01_V020/029.json','L01/L01_V030/067.json']
    l02_anchors=['L02/L02_V015/181.json']
    hashes={str(p):sha(p) for p in [Path('dataset/ocr/L01.zip'),Path('dataset/ocr/L02.zip'),
        Path('dataset/ocr/ocr_vietnamese.zip'),Path('dataset/ocr/ocr_vietnamese_diagnostics.zip')]}
    selected={};expected_by_group={};mapping_hashes={}
    # Identity-only sample selection, before opening any NEW OCR member text.
    for group, anchors in [('L01',l01_anchors),('L02',l02_anchors)]:
        with zipfile.ZipFile(f'dataset/ocr/{group}.zip') as z:
            infos=z.infolist();names=[i.filename for i in infos]
            assert len(set(names))==len(names)
            for info in infos:
                p=PurePosixPath(info.filename)
                if (not re.fullmatch(group+r'/'+group+r'_V\d{3}/\d{3,}\.json',info.filename)
                    or p.is_absolute() or '..' in p.parts or '\\' in info.filename
                    or stat.S_ISLNK(info.external_attr>>16) or info.flag_bits & 1
                    or not 0<info.file_size<=1024*1024):
                    raise ValueError('Unsafe ZIP path/member')
            excluded=set(anchors) | (previously_reviewed if group=='L01' else set())
            candidates=[n for n in names if n not in excluded]
            candidates.sort(key=lambda n:hashlib.sha256(('release-review-v1:'+n).encode()).hexdigest())
            selected[group]=dict(novel=candidates[:16],known_anchors=anchors)
        expected_by_group[group],mapping_hashes[group]=mappings(Path('dataset/mapping'),group)
    write(output/'FROZEN_SAMPLE.json',dict(archives_sha256=hashes,samples=selected,
        rule='First16 SHA256(release-review-v1:member) sorted identities per group; exclude known review/tuning frames. Anchors supplemental, not holdout.'))
    all_rows={};report=dict(archives_sha256=hashes,groups={},mapping_sha256=mapping_hashes)
    for group in ['L01','L02']:
        raw=dict(zip_records(Path(f'dataset/ocr/{group}.zip'),group,expected_by_group[group]))
        rows={n:json.loads(v)[0] for n,v in raw.items()};all_rows[group]=rows
        report['groups'][group]=dict(structure='CRC/schema/identity/mapping/exact full coverage pass',new=metrics(rows))
    old_l01={r['member']:dict(video_id=r['video_id'],ocr_text=r['old_text'],status=r['old_status']) for r in old_pairs}
    result,pairs=comparisons(all_rows['L01'],old_l01)
    report['groups']['L01']['vs_original_latin']=dict(result,old=metrics(old_l01),
        baseline='Preserved PAIRS.json from earlier full audit of original L01 ZIP SHA256 89a1dc89914c5a400a8cc15231ea25e66c8aa9d610016fa555ea54ffcfab31eb; not re-reading missing original archive.')
    write(output/'L01_VS_ORIGINAL_LATIN.json',pairs)
    with zipfile.ZipFile('dataset/ocr/ocr_vietnamese.zip') as z:
        prior={n:json.loads(z.read(n))[0] for n in z.namelist()}
    result,pairs=comparisons(all_rows['L01'],prior)
    report['groups']['L01']['vs_previous_vietnamese_scalar']=dict(result,old=metrics(prior))
    write(output/'L01_VS_PREVIOUS_VIETNAMESE.json',pairs)
    audit=json.loads(Path('benchmark_results/school_ocr_accuracy_20261115_run01/POST_FREEZE_SOURCE_OCR_AUDIT.json').read_text(encoding='utf-8'))
    old_anchor=next(r['source_ocr_text'] for r in audit['rows'] if r['id']=='q05_formol_literal')
    report['groups']['L02']['baseline_limit']='No full original L02 OCR copy locally; school SSH timed out. Cannot compute full paired improvement.'
    write(output/'L02_OLD_ANCHOR.json',dict(member=l02_anchors[0],old_text=old_anchor,
        new_text=all_rows['L02'][l02_anchors[0]]['ocr_text'],
        source='Earlier preserved school source OCR audit, q05_formol_literal'))
    sample_rows=[];manifest=[];image_dir=output/'images';image_dir.mkdir()
    for group in ['L01','L02']:
        members=selected[group]['novel']+selected[group]['known_anchors']
        with zipfile.ZipFile(f'dataset/data/{group}.zip') as z:
            image_lookup={}
            for info in z.infolist():
                if info.is_dir() or not info.filename.lower().endswith('.jpg'):continue
                p=PurePosixPath(info.filename);key=(p.parent.name,p.stem)
                if key in image_lookup:raise ValueError('Duplicate image join')
                image_lookup[key]=info
            for index,member in enumerate(members):
                row=all_rows[group][member];key=(row['video_id'],f'{row["keyframe_n"]:03d}')
                info=image_lookup[key]
                assert 0<info.file_size<=16*1024*1024
                data=z.read(info);path=image_dir/f'{group}_{index+1:02d}_{key[0]}_{key[1]}.jpg'
                with path.open('xb') as f:f.write(data)
                record=dict(group=group,index=index+1,member=member,kind='novel_fixed' if index<16 else 'known_anchor',
                    frame_idx=expected_by_group[group][row['video_id'],row['keyframe_n']],
                    new_text=row['ocr_text'],new_status=row['status'])
                if group=='L01':
                    record['old_latin_text']=old_l01[member]['ocr_text'];record['old_vietnamese_text']=prior[member]['ocr_text']
                elif member==l02_anchors[0]:record['old_latin_text']=old_anchor
                sample_rows.append(record)
                manifest.append(dict(group=group,index=index+1,member=member,file=str(path),
                    source=f'dataset/data/{group}.zip',image_member=info.filename,
                    sha256=hashlib.sha256(data).hexdigest()))
        for batch in range(4):
            canvas=Image.new('RGB',(1280,770),'white');draw=ImageDraw.Draw(canvas)
            for slot,index in enumerate(range(batch*4,batch*4+4)):
                image=Image.open(image_dir/f'{group}_{index+1:02d}_{all_rows[group][members[index]]["video_id"]}_{all_rows[group][members[index]]["keyframe_n"]:03d}.jpg')
                image.thumbnail((640,350));x=slot%2*640;y=slot//2*385
                draw.text((x+6,y+5),f'{index+1:02d} {members[index]}',fill='black');canvas.paste(image,(x,y+30))
            canvas.save(output/f'{group}_panel_{batch+1}.jpg',quality=95)
        group_images=[r for r in manifest if r['group']==group][:16]
        for batch in range(2):
            canvas=Image.new('RGB',(1280,880),'white');draw=ImageDraw.Draw(canvas)
            for slot,row in enumerate(group_images[batch*8:(batch+1)*8]):
                im=Image.open(row['file']);w,h=im.size;crop=im.crop((0,int(h*.89),w,h));crop.thumbnail((1280,85))
                y=slot*110;draw.text((6,y+2),f'{row["index"]:02d} {row["member"]}',fill='black');canvas.paste(crop,(0,y+20))
            canvas.save(output/f'{group}_tickers_{batch+1}.png')
    write(output/'SAMPLE_TEXTS.json',sample_rows);write(output/'IMAGE_MANIFEST.json',manifest)
    assert hashes=={p:sha(p) for p in hashes}
    report['source_archives_unchanged']=True
    report['provenance']='Public-only L01/L02 ZIPs cannot establish engine/batch/image-byte provenance. Existing diagnostics ZIP belongs to previous scalar L01 run, not assumed to describe new archives.'
    write(output/'STRUCTURAL_REPORT.json',report)
    print(json.dumps(report,ensure_ascii=True))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    run(parser.parse_args().output)
