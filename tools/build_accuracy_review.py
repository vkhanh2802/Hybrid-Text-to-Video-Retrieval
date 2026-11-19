"""Build source-vs-top5 sheets after a completed accuracy benchmark. No inference."""
from io import BytesIO
import json
from pathlib import Path
import textwrap
import zipfile

from PIL import Image, ImageDraw, ImageFont, ImageOps


def build(root):
    root = Path(root)
    report = json.loads((root / 'live/REPORT.json').read_text(encoding='utf-8'))
    output = root / 'review'
    output.mkdir(exist_ok=False)
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 17)
    index = []
    for run in report['runs']:
        qid = run['id']
        results = json.loads((root / 'live' / (qid + '_RESULT.json')).read_text(encoding='utf-8'))['results']
        gt = run['ground_truth']
        source = next(p for p in gt['positives'] if p['keyframe_n'] == gt['source_keyframe_n'])
        cells = [{'label': 'SOURCE', 'image': source['image']}]
        for rank, r in enumerate(results[:5], 1):
            p = root / 'live/images' / r['video_id'] / f"{r['keyframe_n']:03d}.jpg"
            origin = 'measured_CPU_fetch_audit'
            if not p.exists():
                # Review-only fallback; never substitutes images into measured inference.
                with zipfile.ZipFile(Path('dataset/data') / (r['video_id'].split('_')[0] + '.zip')) as z:
                    names = [n for n in z.namelist() if Path(n).parent.name == r['video_id']
                             and Path(n).stem.isdigit() and int(Path(n).stem) == r['keyframe_n']
                             and Path(n).suffix.lower() in {'.jpg', '.jpeg', '.png'}]
                    assert len(names) == 1
                    p = output / f"{qid}_rank{rank}.jpg"
                    p.write_bytes(z.read(names[0]))
                origin = 'local_source_ZIP_postrun_review_only'
            cells.append({'label': f"TOP{rank} {r['video_id']} #{r['keyframe_n']}",
                          'image': str(p), 'origin': origin, 'rank': rank,
                          'video_id': r['video_id'], 'keyframe_n': r['keyframe_n']})
        sheet = Image.new('RGB', (1200, 650), '#15191f')
        draw = ImageDraw.Draw(sheet)
        draw.multiline_text((8, 5), '\n'.join(textwrap.wrap(qid + ': ' + run['query'], width=126)),
                            font=font, fill='white', spacing=3)
        for i, cell in enumerate(cells):
            x, y = (i % 3) * 400, 115 + (i // 3) * 265
            draw.text((x + 5, y), cell['label'], font=font, fill='white')
            with Image.open(cell['image']) as image:
                tile = ImageOps.contain(image.convert('RGB'), (395, 230))
            sheet.paste(tile, (x + (400 - tile.width) // 2, y + 28))
        sheet_path = output / (qid + '.jpg')
        sheet.save(sheet_path, quality=94)
        index.append({'id': qid, 'query': run['query'], 'completed': run['completed'],
                      'sheet': str(sheet_path), 'cells': cells})
    with (output / 'INDEX.json').open('x', encoding='utf-8') as stream:
        json.dump(index, stream, ensure_ascii=False, indent=2)
    print(json.dumps({'queries': len(index), 'review_images': sum(len(r['cells']) - 1 for r in index),
                      'output': str(output)}))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    build(p.parse_args().root)
