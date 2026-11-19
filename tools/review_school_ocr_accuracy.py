"""Prepare arm/rank-blinded top1 images for visual inspection after frozen run."""

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import textwrap

from tools.benchmark_school_ocr import client


def prepare(output):
    from PIL import Image, ImageDraw
    output = Path(output)
    report = json.loads((output / "REPORT.json").read_text(encoding="utf-8"))
    directory = output / "blind_top1"
    directory.mkdir(exist_ok=False)
    worker = client("config/school-runtime.local.json", True)
    key = {}
    for record in report["records"]:
        q = record["query"]
        identities = {(arm["final_ranking"][0]["video_id"], arm["final_ranking"][0]["keyframe_n"])
                      for arm in record["arms"].values()}
        order = sorted(identities, key=lambda identity: sha256((q["id"] + repr(identity)).encode()).hexdigest())
        sheet = Image.new("RGB", (1200, 120 + 385 * ((len(order) + 1) // 2)), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 10), q["id"], fill="black")
        for i, line in enumerate(textwrap.wrap(q["text"].encode("ascii", "backslashreplace").decode(), 145)):
            draw.text((10, 30 + 15 * i), line, fill="black")
        key[q["id"]] = {}
        for i, identity in enumerate(order):
            label = f"item{i + 1}"
            matching_arms = [a for a, r in record["arms"].items()
                            if (r["final_ranking"][0]["video_id"], r["final_ranking"][0]["keyframe_n"]) == identity]
            x, y = (i % 2) * 600 + 10, (i // 2) * 385 + 120
            draw.text((x, y), label, fill="black")
            try:
                raw, mime = worker.read(*identity)
            except FileNotFoundError:
                key[q["id"]][label] = {"video_id": identity[0], "keyframe_n": identity[1],
                                        "status": "missing_image", "arms": matching_arms}
                draw.text((x, y + 30), "Image unavailable: do not invent relevance label", fill="black")
                continue
            suffix = ".png" if mime == "image/png" else ".jpg"
            name = q["id"] + "_" + label + suffix
            with (directory / name).open("xb") as stream:
                stream.write(raw)
            with Image.open(BytesIO(raw)) as im:
                image = im.convert("RGB")
                image.thumbnail((580, 330))
            sheet.paste(image, (x, y + 25))
            key[q["id"]][label] = {"video_id": identity[0], "keyframe_n": identity[1],
                "status": "ready", "image_sha256": sha256(raw).hexdigest(), "path": "blind_top1/" + name,
                "arms": matching_arms}
        sheet.save(directory / (q["id"] + "_sheet.jpg"), quality=95)
    with (output / "BLIND_REVIEW_KEY.json").open("x", encoding="utf-8") as stream:
        json.dump(key, stream, indent=2)
    print(json.dumps({"phase": "blind_top1_images_ready", "queries": len(key),
                      "images": sum(len(v) for v in key.values())}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    prepare(parser.parse_args().output)
