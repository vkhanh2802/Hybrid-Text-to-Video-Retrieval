"""Read-only paired supplied-ZIP audit. No inference, import, or live mutations."""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat

from kis.public_ocr import unique_object
from kis.school_ocr import mappings
from tools.import_school_ocr import zip_records


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    import zipfile
    new_path, old_path = Path('dataset/ocr/ocr_vietnamese.zip'), Path('dataset/ocr/L01.zip')
    initial_hashes = {str(p): sha(p) for p in (new_path, old_path)}
    expected, mapping_hashes = mappings(Path('dataset/mapping'), 'L01')
    # Freeze deterministic image sample from identities BEFORE reading OCR text.
    with zipfile.ZipFile(new_path) as archive:
        infos = archive.infolist()
        names = sorted(i.filename for i in infos)
        if len(set(names)) != len(names):
            raise ValueError('Duplicate ZIP names')
        for info in infos:
            p = PurePosixPath(info.filename)
            if (not re.fullmatch(r'L01/L01_V\d{3}/\d{3,}\.json', info.filename)
                    or p.is_absolute() or '..' in p.parts or '\\' in info.filename
                    or stat.S_ISLNK(info.external_attr >> 16) or info.flag_bits & 1
                    or not 0 < info.file_size <= 1024 * 1024):
                raise ValueError('Unsafe/unexpected member')
        selected = [names[i * (len(names) - 1) // 15] for i in range(16)]
        write(output / 'FROZEN_SAMPLE.json', {
            'rule': '16 evenly spaced sorted member identities, including endpoints; frozen before paired text review',
            'archive_sha256': initial_hashes[str(new_path)], 'members': selected})
        if archive.testzip() is not None:
            raise ValueError('CRC failure')
    new_raw = dict(zip_records(new_path, 'L01', expected))
    old_raw = dict(zip_records(old_path, 'L01', expected))
    new = {k: json.loads(v, object_pairs_hook=unique_object)[0] for k, v in new_raw.items()}
    old = {k: json.loads(v, object_pairs_hook=unique_object)[0] for k, v in old_raw.items()}
    def metrics(rows):
        texts = [r['ocr_text'] for r in rows.values()]
        chars = Counter(''.join(texts))
        return dict(records=len(rows), videos=len({r['video_id'] for r in rows.values()}),
                    status_counts=dict(Counter(r['status'] for r in rows.values())),
                    characters=sum(map(len, texts)), lines=sum(len(t.splitlines()) for t in texts),
                    replacement_characters=chars['\ufffd'],
                    records_with_replacement=sum('\ufffd' in t for t in texts),
                    vietnamese_probe_counts={c: chars[c] for c in 'ảấứừếộắằẳẵặậẩẫầậềểễệốồổỗợớờởỡựửữỳỷỹỵđĐ'})
    pairs = []
    for name in sorted(new):
        a, b = old[name], new[name]
        pairs.append(dict(member=name, video_id=b['video_id'], keyframe_n=b['keyframe_n'],
                          frame_idx=expected[b['video_id'], b['keyframe_n']],
                          old_text=a['ocr_text'], new_text=b['ocr_text'],
                          old_status=a['status'], new_status=b['status']))
    report = dict(archives_sha256=initial_hashes, new_zip_crc='pass',
                  schema_and_mapping='pass: seven fields, exact full L01 coverage, no duplicate identities',
                  mapping_sha256=mapping_hashes, old=metrics(old), new=metrics(new),
                  paired_records=len(pairs), identical_text=sum(r['old_text']==r['new_text'] for r in pairs),
                  new_shorter=sum(len(r['new_text'])<len(r['old_text']) for r in pairs),
                  new_longer=sum(len(r['new_text'])>len(r['old_text']) for r in pairs),
                  new_empty_old_nonempty=sum(bool(r['old_text']) and not r['new_text'] for r in pairs),
                  diagnostics_found=False,
                  limitations=['Public OCR has no image hash/model provenance/confidences/geometry.',
                               'Character coverage and output length are not accuracy measurements.',
                               'Image sample is small and not human-annotated corpus ground truth.'])
    write(output / 'PAIRS.json', pairs)
    write(output / 'SAMPLE_PAIRS.json', [r for r in pairs if r['member'] in selected])
    assert initial_hashes == {str(p): sha(p) for p in (new_path, old_path)}
    report['source_archives_unchanged'] = True
    write(output / 'STRUCTURAL_REPORT.json', report)
    print(json.dumps(report, ensure_ascii=True))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    run(parser.parse_args().output)
