"""Stage original OCR ZIPs or their single main nested ZIP, without rewriting JSON.

Diagnostics ZIPs remain in the original local source; they are never search data.
All records must pass the existing strict public schema/mapping/full-coverage gate.
"""
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile

from tools.import_school_ocr import zip_records
from kis.school_ocr import mappings
from tools.transfer_school_data import sha256_file


def prepare(source, mapping_root, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    staged = output / 'validated_archives'
    staged.mkdir()
    rows = []
    for archive in sorted(source.glob('L*.zip')):
        group = archive.stem
        if not re.fullmatch(r'L(?:0[1-9]|1[0-9]|2[01])', group):
            raise ValueError('Unexpected OCR archive name')
        original_hash = sha256_file(archive)
        nested_name, diagnostics = None, []
        with zipfile.ZipFile(archive) as z:
            files = [i for i in z.infolist() if not i.is_dir()]
            for member in z.infolist():
                p = PurePosixPath(member.filename)
                if (p.is_absolute() or '..' in p.parts or '\\' in member.filename
                        or stat.S_ISLNK(member.external_attr >> 16) or member.flag_bits & 1):
                    raise ValueError('Unsafe outer OCR ZIP member')
            if any(i.filename.endswith('.zip') for i in files):
                expected_prefix = group + '_OCR/'
                main = [i for i in files if i.filename == expected_prefix + 'ocr_vietnamese_v3_batch32.zip']
                diagnostics = [i.filename for i in files if i.filename == expected_prefix + 'ocr_vietnamese_v3_batch32_diagnostics.zip']
                allowed = {i.filename for i in main} | set(diagnostics)
                if len(main) != 1 or len(allowed) != len(files) or not 0 < main[0].file_size <= 128 * 1024**2:
                    raise ValueError('Expected exactly one known main OCR ZIP plus optional diagnostics')
                nested_name = main[0].filename
                with (staged / archive.name).open('xb') as stream:
                    stream.write(z.read(main[0]))  # CRC validated, exact inner ZIP bytes.
            else:
                with archive.open('rb') as inp, (staged / archive.name).open('xb') as out:
                    shutil.copyfileobj(inp, out)
        expected, hashes = mappings(mapping_root, group, allow_frame_aliases=True)
        records = zip_records(staged / archive.name, group, expected)
        assert sha256_file(archive) == original_hash
        row = {'group': group, 'original_archive_sha256': original_hash,
               'import_archive_sha256': sha256_file(staged / archive.name),
               'nested_member': nested_name, 'diagnostics_not_imported': diagnostics,
               'records': len(records), 'videos': len(hashes),
               'status_counts': dict(Counter(json.loads(raw)[0]['status'] for _, raw in records)),
               'mapping_sha256': hashes}
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != 'mapping_sha256'}), flush=True)
    if not rows:
        raise ValueError('No OCR ZIPs')
    report = {'status': 'passed', 'groups': rows, 'total_records': sum(r['records'] for r in rows),
              'source_archives_modified': False, 'JSON_bytes_rewritten': False,
              'validated_archives': str(staged)}
    with (output / 'PREFLIGHT.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('dataset/ocr'))
    parser.add_argument('--mapping-root', type=Path, default=Path('dataset/mapping'))
    parser.add_argument('--output', type=Path, required=True)
    prepare(**vars(parser.parse_args()))
