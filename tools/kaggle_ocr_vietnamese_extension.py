"""Embedded into the standalone Vietnamese notebook AFTER legacy IO helpers.

This file is a notebook source fragment, not a standalone module. Heavy imports
are lazy. Existing datasets, notebook, venv and published OCR remain untouched.
"""

VI_REVISION = 'fe8c3a7fc714aec57ab81cec844eb3adf0c1636c'
VI_BLOBS = {
    'LICENSE': '261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64',
    'config/base.yml': '08a9fca6233a19fa1da42aed6214958e4284ea97',
    'config/vgg-transformer.yml': 'ca7dbbca06177d02e472821b114a751554833294',
    'vietocr/__init__.py': 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391',
    'vietocr/model/__init__.py': 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391',
    'vietocr/model/backbone/__init__.py': 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391',
    'vietocr/model/backbone/cnn.py': '1af65da7bf0c6105e0cefe36302e67f05745b88a',
    'vietocr/model/backbone/vgg.py': 'b164b9c783c2e4106b2e0701d16f0cc181924970',
    'vietocr/model/backbone/resnet.py': 'b2aff464b44015f5941e8711e12b8a585026b786',
    'vietocr/model/seqmodel/__init__.py': 'e69de29bb2d1d6434b8b29ae775ad8c2e48c5391',
    'vietocr/model/seqmodel/transformer.py': 'a7d4ed5313cf4031eb60012677cdbe2a8553eba2',
    'vietocr/model/seqmodel/seq2seq.py': '31960ed313d6dccf3635fc45371e12d8ef448032',
    'vietocr/model/seqmodel/convseq2seq.py': '46b980b3033ec13b1b17d457df037de183162636',
    'vietocr/model/transformerocr.py': '7c88cd9860592c3a699652397a9ba4ec60bd2926',
    'vietocr/model/vocab.py': 'a4fab4059ad3098e57388d83ea49be290e6d7b9d',
}
WEIGHTS_URL = ('https://huggingface.co/doanhm/vgg_transformer/resolve/'
               '4de5b0429954fdac3dee3173978c062b3bc4d534/vgg_transformer.pth')
WEIGHTS_SHA256 = '380512193a8b6cbf6fad80deacdc9b6939d10d473d199892fc6408d13775ea59'
DET_BASE = ('https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det/resolve/'
            'ca867c897ecbca8873081573a802ad70d499cb94/')
DET_ASSETS = {
    'inference.json': ('git', '7b582d63397b09eea2b3ee8c7c7e4efae5d0ccff'),
    'inference.yml': ('git', 'c4f16313e7e2c567da8e9fa8f4aeba14a307786e'),
    'inference.pdiparams': ('sha256', '183146fe9d9910352f68482f623bcbbb9fa7b9e8fa1463b9ad288cef00524d2d'),
}
VI_LOWER = ('aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩị'
            'oòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵđ')
VI_REQUIRED = VI_LOWER + VI_LOWER.upper()
PUBLIC_MIN_CONFIDENCE = float(os.environ.get('KIS_VI_OCR_MIN_CONFIDENCE', '0.5'))
if not math.isfinite(PUBLIC_MIN_CONFIDENCE) or not 0 <= PUBLIC_MIN_CONFIDENCE <= 1:
    raise ValueError('Public confidence threshold must be in [0,1]')
OPTIONS = dict(detector_limit_side_len=1600, detector_limit_type='max',
               detector_thresh=0.3, detector_box_thresh=0.6, detector_unclip_ratio=1.5,
               recognition_image_height=32, recognition_max_width=2048,
               recognition_max_tokens=256, recognition_batch_size=16,
               recognition_frame_window=16, recognition_batch_policy='exact_width_no_padding_fp32_v1',
               line_segment_trigger_width=512, line_segment_max_width=480,
               line_segment_min_gap_height_ratio=0.18, line_segment_max_count=32,
               line_segment_method='otsu_word_gaps_no_overlap_v1',
               public_min_confidence=PUBLIC_MIN_CONFIDENCE, use_super_resolution=False)
MODELS = dict(detection='PP-OCRv5_server_det', recognition='VietOCR/vgg_transformer')


def git_blob_digest(raw):
    return hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()


def fetch_asset(url, path, kind, expected, max_bytes):
    """Known HTTPS assets only, verify cached bytes too; no unpickling here."""
    from urllib.request import urlopen
    path = safe_path(path)
    def checksum(p):
        return git_blob_digest(p.read_bytes()) if kind == 'git' else file_digest(p)
    if path.exists():
        if not path.is_file() or path.stat().st_size > max_bytes or checksum(path) != expected:
            raise ValueError('Cached asset changed; preserve it and use a fresh owned runtime: ' + str(path))
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    try:
        with urlopen(url, timeout=180) as response, temporary.open('xb') as stream:
            if not response.geturl().startswith('https://'):
                raise ValueError('Asset redirect must remain HTTPS')
            total = 0
            while True:
                data = response.read(1024 * 1024)
                if not data:
                    break
                total += len(data)
                if total > max_bytes:
                    raise ValueError('Asset exceeds bounded download size')
                stream.write(data)
        if checksum(temporary) != expected:
            raise ValueError('Pinned model/source asset checksum mismatch: ' + path.name)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def validate_alphabet(vocab):
    missing = sorted(set(VI_REQUIRED) - set(vocab.chars))
    if missing:
        raise RuntimeError('Recognizer cannot represent Vietnamese: ' + ''.join(missing))
    if vocab.decode(vocab.encode(VI_REQUIRED)) != VI_REQUIRED:
        raise RuntimeError('Vietnamese vocabulary roundtrip failed')
    return dict(required=len(set(VI_REQUIRED)), missing=[], roundtrip=True)


def public_record(record):
    # Raw lines/probabilities remain in technical checkpoints. This configurable
    # filter is not a calibrated accuracy guarantee or negative image evidence.
    lines = record['lines'] if record['status'] != 'error' else []
    text = '\n'.join(line['text'] for line in lines if line['text'].strip()
                     and line['confidence'] >= PUBLIC_MIN_CONFIDENCE
                     and not line.get('truncated', False) and line.get('public_eligible', True))
    video = record['video_id']
    return dict(video_id=video, frame_id=None, keyframe_n=record['keyframe_n'],
                image_path=f'data/{video.split("_V")[0]}/{video}/{Path(record["relative_path"]).name}',
                ocr_text=text, retrieval_text=text,
                status='error' if record['status'] == 'error' else 'ok' if text else 'empty')


def recognition_width(width, height):
    if min(width, height) < 1:
        raise ValueError('Empty detector crop')
    return max(32, math.ceil((32 * width / height) / 10) * 10)


def plan_line_segments(bgr):
    """Partition only at sufficiently wide background gaps; never cut by text guesses.

    Ranges are adjacent, nonoverlapping, and cover every original crop column.
    No arbitrary glyph cut, OCR-string overlap deduplication, or image synthesis.
    This geometric heuristic is not proof that a boundary is a word space.
    """
    import numpy as np
    value = np.asarray(bgr)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
        raise ValueError('Expected uint8 BGR crop')
    h, w = value.shape[:2]
    if recognition_width(w, h) <= OPTIONS['line_segment_trigger_width']:
        return dict(status='short', ranges=[[0, w]])
    # Otsu luminance threshold, no OpenCV/Paddle import in the Torch process.
    gray = np.rint(value @ np.array([.114, .587, .299])).astype(np.uint8)
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    weight = np.cumsum(hist)
    moment = np.cumsum(hist * np.arange(256))
    denominator = weight * (weight[-1] - weight)
    score = np.zeros(256)
    np.divide((moment[-1] * weight - moment * weight[-1]) ** 2,
              denominator, out=score, where=denominator > 0)
    threshold = int(score.argmax())
    if not score.max():
        return dict(status='no_safe_gap', ranges=[], threshold=threshold)
    # The median border luminance estimates background polarity. Both light
    # ticker text on dark and dark print on light backgrounds are supported.
    border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    light_text = bool(np.median(border) <= threshold)
    ink = gray > threshold if light_text else gray <= threshold
    empty_columns = ~ink.any(axis=0)
    minimum_gap = max(2, math.ceil(h * OPTIONS['line_segment_min_gap_height_ratio']))
    edges = np.diff(np.r_[False, empty_columns, False].astype(np.int8))
    gaps = [(int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
            if b - a >= minimum_gap and a > 0 and b < w]
    # Round down to a multiple of 10 so recognition_width's rounding cannot
    # push a planned segment beyond the configured normalized-width target.
    max_pixels = int((OPTIONS['line_segment_max_width'] // 10 * 10) * h // 32)
    ranges, start = [], 0
    while w - start > max_pixels:
        candidates = [(a + b) // 2 for a, b in gaps if start < (a + b) // 2 <= start + max_pixels]
        if not candidates or len(ranges) >= OPTIONS['line_segment_max_count'] - 1:
            return dict(status='no_safe_gap', ranges=[], threshold=threshold, light_text=light_text)
        end = max(candidates)
        ranges.append([start, end])
        start = end
    ranges.append([start, w])
    return dict(status='split', ranges=ranges, threshold=threshold, light_text=light_text,
                boundary_kind='background_gap_heuristic', overlap_px=0)


def pack_array(array):
    import base64
    import numpy as np
    value = np.asarray(array)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3 or value.nbytes > 48 * 1024 ** 2:
        raise ValueError('Expected bounded uint8 BGR image/crop')
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return base64.b64encode(stream.getvalue()).decode('ascii')


def unpack_array(encoded_array):
    import base64
    import numpy as np
    if not isinstance(encoded_array, str) or len(encoded_array) > 68 * 1024 ** 2:
        raise ValueError('Detector image payload exceeds limit')
    value = np.load(io.BytesIO(base64.b64decode(encoded_array, validate=True)), allow_pickle=False)
    if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3 or min(value.shape) < 1 or value.nbytes > 48 * 1024 ** 2:
        raise ValueError('Invalid detector image/crop')
    return value


def detector_worker():
    """Paddle-only child. Redirect ALL backend stdout to logs; JSON RPC on pipe."""
    from contextlib import redirect_stdout
    # Duplicate the protocol pipe first, then redirect FD1 as well as Python
    # stdout. Native Paddle/CUDA prints must not corrupt JSON replies.
    wire = os.fdopen(os.dup(sys.stdout.fileno()), 'w', encoding='utf-8', buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    with redirect_stdout(sys.stderr):
        import numpy as np
        import paddle
        from paddleocr import TextDetection
        from paddlex.inference.pipelines.components import CropByPolys, SortQuadBoxes
        gpu = require_gpu(paddle)
        versions = {name: importlib.metadata.version(name) for name in PADDLE_PINS}
        if versions != PADDLE_PINS:
            raise RuntimeError('Paddle child core version mismatch: ' + str(versions))
        root = safe_path(Path.home() / 'models-vietocr-pinned/detector')
        for name, (kind, expected) in DET_ASSETS.items():
            fetch_asset(DET_BASE + name, root / name, kind, expected, 100 * 1024 ** 2)
        detector = TextDetection(model_name=MODELS['detection'], model_dir=str(root), device='gpu:0',
            enable_hpi=False, use_tensorrt=False, limit_side_len=OPTIONS['detector_limit_side_len'],
            limit_type='max', thresh=.3, box_thresh=.6, unclip_ratio=1.5)
        actual = detector.paddlex_predictor
        if actual.model_name != MODELS['detection'] or actual.pp_option.device_type != 'gpu' or actual.pp_option.device_id != 0:
            raise RuntimeError('Detector model/device mismatch')
        crop, sort = CropByPolys(det_box_type='quad'), SortQuadBoxes()
        profile = dict(gpu, core_versions=versions,
                       model_files_sha256={name: file_digest(root / name) for name in DET_ASSETS})
        print(json.dumps(dict(status='ready', profile=profile)), file=wire, flush=True)
        for line in sys.stdin:
            if len(line) > 68 * 1024 ** 2:
                raise ValueError('Oversized detector request')
            request = json.loads(line)
            if request.get('action') != 'detect' or type(request.get('id')) is not int:
                raise ValueError('Invalid detector request')
            image = unpack_array(request['image'])
            results = list(detector.predict(input=image))
            if len(results) != 1 or 'error' in results[0]:
                raise RuntimeError('Unexpected detector response')
            # PaddleX SortQuadBoxes returns a Python list in this API. Normalize
            # either supported form once, preserving order for crops and JSON.
            polygons = np.asarray(sort(np.asarray(results[0]['dt_polys'])))
            crops = list(crop(image, polygons)) if len(polygons) else []
            if len(crops) != len(polygons):
                raise RuntimeError('Detector/crop alignment mismatch')
            print(json.dumps(dict(status='ok', id=request['id'], polygons=polygons.tolist(),
                                  crops=[pack_array(c) for c in crops])), file=wire, flush=True)


class DetectorProcess:
    """Persistent child: no GPU DLLs or Python package paths shared with Torch."""
    def __init__(self):
        import atexit
        import subprocess
        python = safe_path(os.environ['KIS_VI_PADDLE_PYTHON'])
        launcher = safe_path(os.environ['KIS_VI_LAUNCHER'])
        self.process = subprocess.Popen([str(python), '-I', '-S', str(launcher), 'worker',
            str(safe_path(__file__)), 'detector-worker'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, text=True, encoding='utf-8', bufsize=1)
        self.sequence = 0
        try:
            reply = self.read()
            if reply.get('status') != 'ready':
                raise RuntimeError('Detector child did not become ready')
            self.profile = reply['profile']
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def read(self):
        line = self.process.stdout.readline(68 * 1024 ** 2 + 1)
        if not line or len(line) > 68 * 1024 ** 2:
            raise RuntimeError('Paddle child failed or sent invalid response; inspect its stderr above')
        return json.loads(line)

    def detect(self, image):
        self.sequence += 1
        self.process.stdin.write(json.dumps(dict(action='detect', id=self.sequence, image=pack_array(image))) + '\n')
        self.process.stdin.flush()
        reply = self.read()
        if reply.get('status') != 'ok' or reply.get('id') != self.sequence:
            raise RuntimeError('Detector response identity/status mismatch')
        polygons, encoded_crops = reply['polygons'], reply['crops']
        if not isinstance(polygons, list) or not isinstance(encoded_crops, list) or len(polygons) != len(encoded_crops):
            raise RuntimeError('Detector crop identity/count mismatch')
        return polygons, [unpack_array(c) for c in encoded_crops]

    def close(self):
        import subprocess
        process = self.process
        if process.stdin and not process.stdin.closed:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()  # Only this adapter's child, never another runtime.
            process.wait(timeout=15)
        if process.stdout:
            process.stdout.close()


class Adapter:
    def __init__(self):
        import numpy as np
        versions = {name: importlib.metadata.version(name) for name in PINS}
        if versions != PINS:
            raise RuntimeError(f'Core version mismatch: {versions}; expected {PINS}')
        import torch
        import yaml
        from PIL import Image
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('VietOCR requires the same visible GPU0, no CPU fallback')
        smoke = torch.tensor([[2.]], device='cuda:0')
        if float((smoke @ smoke).item()) != 4.:
            raise RuntimeError('Torch CUDA smoke failed')
        torch.set_num_threads(2)
        root = safe_path(Path.home() / 'models-vietocr-pinned')
        root.mkdir(parents=True, exist_ok=True)
        source = root / 'source'
        for name, expected in VI_BLOBS.items():
            fetch_asset(f'https://raw.githubusercontent.com/pbcquoc/vietocr/{VI_REVISION}/{name}',
                        source / name, 'git', expected, 1024 * 1024)
        weights = fetch_asset(WEIGHTS_URL, root / 'vgg_transformer.pth', 'sha256', WEIGHTS_SHA256, 200 * 1024 ** 2)
        detector = root / 'detector'
        for name, (kind, expected) in DET_ASSETS.items():
            fetch_asset(DET_BASE + name, detector / name, kind, expected, 100 * 1024 ** 2)
        sys.path.insert(0, str(source))
        from vietocr.model.transformerocr import VietOCR
        from vietocr.model.vocab import Vocab
        config = yaml.safe_load((source / 'config/base.yml').read_text(encoding='utf-8'))
        config.update(yaml.safe_load((source / 'config/vgg-transformer.yml').read_text(encoding='utf-8')))
        config['cnn']['pretrained'] = False  # NO unrelated ImageNet download.
        if config['backbone'] != 'vgg19_bn' or config['seq_modeling'] != 'transformer':
            raise RuntimeError('Unexpected VietOCR architecture')
        self.vocab = Vocab(config['vocab'])
        alphabet = validate_alphabet(self.vocab)
        model = VietOCR(len(self.vocab), config['backbone'], config['cnn'], config['transformer'], 'transformer')
        # Pinned tensor-only state, strict keys/shapes/vocabulary. NEVER unsafe
        # weights_only=False fallback for an incompatible legacy pickle.
        state = torch.load(weights, map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        if model.transformer.fc.out_features != len(self.vocab):
            raise RuntimeError('Checkpoint/vocabulary head mismatch')
        del state
        self.model = model.to('cuda:0').eval()
        self.detector = DetectorProcess()
        self.config = config
        self.engine = dict(device='gpu:0', gpu_name=torch.cuda.get_device_name(0), models=MODELS,
            core_versions=versions, alphabet=alphabet, detector_process=self.detector.profile,
            runtime_isolation='separate venvs/processes; incompatible NVIDIA dependencies never coinstalled',
            source_revision=VI_REVISION, weights_source=WEIGHTS_URL,
            model_files_sha256={'recognition.pth': file_digest(weights),
                **{'detector/' + n: file_digest(detector / n) for n in DET_ASSETS}},
            source_files_sha256={name: file_digest(source / name) for name in VI_BLOBS},
            recognition_config=config,
            confidence='single crop: mean character softmax; segmented line: minimum segment mean; both uncalibrated',
            warnings=['Alphabet support and CUDA smoke do not establish OCR accuracy.',
                      'VietOCR can hallucinate on blurred/background crops. Inspect preview before bulk.',
                      'Weights mirrored by doanhm; origin/model compatibility checked by pinned digest and strict state load.'])
        print(json.dumps(self.engine, ensure_ascii=False, indent=2), flush=True)

    def recognize_crop(self, bgr):
        return self.recognize_crops([bgr])[0]

    def recognize_crops(self, crops):
        """Real tensor microbatches, exact resized-width buckets, original order.

        No extra width padding or mixed precision: preserve v3 input pixels.
        Width buckets and final tails may contain fewer than16 real crops.
        """
        import numpy as np
        import torch
        from PIL import Image
        batch_size = OPTIONS['recognition_batch_size']
        if type(batch_size) is not int or not 1 <= batch_size <= 16:
            raise ValueError('Recognition batch must be an integer from1 to16')
        buckets, shapes = {}, []
        for i, crop in enumerate(crops):
            value = np.asarray(crop)
            if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
                raise ValueError('Expected uint8 BGR recognition crop')
            h, w = value.shape[:2]
            desired = recognition_width(w, h)
            width = min(OPTIONS['recognition_max_width'], desired)
            shapes.append((w, h, width, desired))
            buckets.setdefault(width, []).append(i)
        results = [None] * len(crops)
        if not hasattr(self, 'batch_stats'):
            self.batch_stats = dict(forward_batches=0, crops=0, batch_size_histogram={})
        for width, indices in buckets.items():
            for start in range(0, len(indices), batch_size):
                chosen = indices[start:start + batch_size]
                tensors = []
                for index in chosen:
                    image = Image.fromarray(crops[index][:, :, ::-1]).resize((width, 32), Image.Resampling.LANCZOS)
                    tensors.append(torch.from_numpy(np.asarray(image).transpose(2, 0, 1).copy()).float().div(255))
                try:
                    tensor = torch.stack(tensors).to('cuda:0')
                    decoded = self.decode_batch(tensor)
                except torch.cuda.OutOfMemoryError as exc:
                    raise RuntimeError('VietOCR batch16 CUDA OOM; run stopped, checkpoints preserved. '
                                       'No silent smaller-batch/CPU fallback; inspect GPU memory before retry.') from exc
                histogram = self.batch_stats['batch_size_histogram']
                histogram[str(len(chosen))] = histogram.get(str(len(chosen)), 0) + 1
                self.batch_stats['forward_batches'] += 1
                self.batch_stats['crops'] += len(chosen)
                for index, row in zip(chosen, decoded):
                    w, h, width, desired = shapes[index]
                    results[index] = dict(row, width_capped=desired > width, crop_size=[w, h],
                        recognizer_size=[width, 32], probability_kind='uncalibrated_mean_character_softmax',
                        completeness='not_verified', inference_batch_size=len(chosen),
                        inference_batch_width=width, inference_batch_policy=OPTIONS['recognition_batch_policy'])
        return results

    def decode_batch(self, tensor):
        import torch
        count = tensor.shape[0]
        ids = [[self.vocab.go] for _ in range(count)]
        probabilities = [[] for _ in range(count)]
        finished = [False] * count
        active = list(range(count))
        with torch.inference_mode():
            memory = self.model.transformer.forward_encoder(self.model.cnn(tensor))
            for _ in range(OPTIONS['recognition_max_tokens']):
                tokens = torch.tensor([ids[i] for i in active], dtype=torch.long, device='cuda:0').transpose(0, 1)
                logits, _ = self.model.transformer.forward_decoder(tokens, memory)
                distribution = torch.softmax(logits[:, -1, :], dim=-1)
                if distribution.shape[0] != len(active):
                    raise RuntimeError('VietOCR decoder batch alignment mismatch')
                if not bool(torch.isfinite(distribution).all().item()):
                    raise RuntimeError('Non-finite recognizer output')
                scores, choices = distribution.max(dim=-1)
                # One batched host transfer per vector, not a .item() sync for
                # every crop. Finished rows are removed from memory and tokens.
                keep, remaining = [], []
                for column, (identity, token, score) in enumerate(zip(active, choices.tolist(), scores.tolist())):
                    ids[identity].append(token)
                    if token == self.vocab.eos:
                        finished[identity] = True
                    else:
                        if token < 4:
                            raise RuntimeError('Unexpected non-text token; no guessed substitution')
                        probabilities[identity].append(score)
                        keep.append(column)
                        remaining.append(identity)
                if not remaining:
                    break
                if len(keep) != len(active):
                    memory = memory[:, keep, :]
                active = remaining
        return [dict(text=self.vocab.decode(ids[i]),
                     confidence=sum(probabilities[i]) / len(probabilities[i]) if probabilities[i] else 0.,
                     truncated=not finished[i], decoder_eos=finished[i], decoded_tokens=len(probabilities[i]))
                for i in range(count)]

    def recognize_line(self, bgr):
        return self.recognize_lines([bgr])[0]

    def recognize_lines(self, crops):
        jobs, descriptors = [], []
        for bgr in crops:
            plan = plan_line_segments(bgr)
            full_index = len(jobs)
            jobs.append(bgr)
            segment_indices = []
            if plan['status'] == 'split':
                for start, end in plan['ranges']:
                    segment_indices.append((len(jobs), start, end))
                    jobs.append(bgr[:, start:end].copy())
            descriptors.append((bgr, plan, full_index, segment_indices))
        raw = self.recognize_crops(jobs)
        if len(raw) != len(jobs):
            raise RuntimeError('Recognition job count mismatch')
        results = []
        for bgr, plan, full_index, indices in descriptors:
            segments = [dict(raw[index], x_range_crop=[start, end],
                             crop_bgr_sha256=digest(jobs[index].tobytes())) for index, start, end in indices]
            results.append(self.assemble_line(bgr, raw[full_index], plan, segments))
        return results

    def assemble_line(self, bgr, full, plan, segments):
        common = dict(segmentation=plan, completeness='not_verified')
        if plan['status'] == 'short':
            return dict(full, **common, recognition_strategy='full_line', public_eligible=True)
        if plan['status'] != 'split':
            # Keep raw baseline for inspection, but do not silently publish a
            # known-risk long prefix when no safe geometric partition exists.
            return dict(full, **common, full_line=full, recognition_strategy='full_line_unsegmented',
                        public_eligible=False, review_reasons=['long_line_no_safe_gap'])
        reasons = []
        if any(not s['text'].strip() for s in segments):
            reasons.append('segment_empty')
        if any(s['confidence'] < PUBLIC_MIN_CONFIDENCE for s in segments):
            reasons.append('segment_low_confidence')
        if any(s['truncated'] for s in segments):
            reasons.append('segment_token_cap')
        if any(s['width_capped'] for s in segments):
            reasons.append('segment_width_cap')
        # No language correction or fuzzy overlap merge: real repeated words
        # survive. Choice is geometric, never based on greater confidence.
        return dict(text=' '.join(s['text'].strip() for s in segments),
                    confidence=min(s['confidence'] for s in segments),
                    truncated=any(s['truncated'] for s in segments),
                    width_capped=any(s['width_capped'] for s in segments),
                    crop_size=list(reversed(bgr.shape[:2])), recognizer_size=full['recognizer_size'],
                    probability_kind='minimum_of_uncalibrated_segment_means',
                    decoder_eos=all(not s['truncated'] for s in segments),
                    recognition_strategy='word_gap_segments', public_eligible=not reasons,
                    review_reasons=reasons, full_line=full, segments=segments, **common)

    def recognize(self, padded):
        return self.recognize_many([padded])[0]

    def recognize_many(self, padded_images):
        import numpy as np
        jobs, frames = [], []
        for padded in padded_images:
            bgr = np.asarray(padded)[:, :, ::-1].copy()
            polygons, crops = self.detector.detect(bgr)
            if len(crops) != len(polygons):
                raise RuntimeError('Detector/crop alignment mismatch')
            frames.append((len(jobs), polygons))
            jobs.extend(crops)
        recognized = self.recognize_lines(jobs)
        return [[dict(recognized[start + i], polygon_padded=polygon) for i, polygon in enumerate(polygons)]
                for start, polygons in frames]


_legacy_record_for = record_for
def record_for(item, sha, profile_hash, decoded, error, adapter):
    # Reuse geometry/identity validation, preserve extra per-line diagnostics.
    if error:
        return _legacy_record_for(item, sha, profile_hash, decoded, error, adapter)
    captured = []
    class Capture:
        def recognize(self, image):
            captured.extend(adapter.recognize(image))
            return captured
    record = _legacy_record_for(item, sha, profile_hash, decoded, None, Capture())
    for target, raw in zip(record['lines'], captured):
        for name in ('truncated', 'width_capped', 'crop_size', 'recognizer_size', 'probability_kind',
                     'decoder_eos', 'decoded_tokens', 'completeness', 'recognition_strategy',
                     'inference_batch_size', 'inference_batch_width', 'inference_batch_policy',
                     'public_eligible', 'review_reasons', 'segmentation', 'segments', 'full_line'):
            if name in raw:
                target[name] = raw[name]
        target['public_included'] = (bool(target['text'].strip()) and target['confidence'] >= PUBLIC_MIN_CONFIDENCE
                                     and not target.get('truncated', False) and target.get('public_eligible', True))
    return record


def records_for_batch(rows, profile_hash, adapter):
    good = [i for i, row in enumerate(rows) if row[3] is None]
    padded = [rows[i][2][0] for i in good]
    if hasattr(adapter, 'recognize_many'):
        values = adapter.recognize_many(padded) if padded else []
    else:
        values = [adapter.recognize(image) for image in padded]
    if len(values) != len(good):
        raise RuntimeError('Frame batch reply count mismatch')
    by_index = dict(zip(good, values))
    class PreparedFrame:
        def __init__(self, lines):
            self.lines = lines
        def recognize(self, image):
            return self.lines
    return [record_for(item, sha, profile_hash, decoded, error,
                       PreparedFrame(by_index.get(i, [])))
            for i, (item, sha, decoded, error) in enumerate(rows)]


_legacy_run = run
def run(root, output, env_root, run_id, adapter=None, progress=None):
    """Buffer up to16 uncached frames to fill width-bucketed crop batches.

    Detector still runs one image/request. Checkpoints retain individual image
    hashes and order; no previous profile is reused when batching changes.
    """
    # Preserve the lightweight test/custom adapter interface. Production Adapter
    # always uses recognize_many and never silently switches to scalar inference.
    if adapter is not None and not hasattr(adapter, 'recognize_many'):
        return _legacy_run(root, output, env_root, run_id, adapter, progress)
    root, output, env_root = validate_paths(root, output, env_root)
    claim_output(root, output)
    state = dict(run_id=run_id, phase='running', complete=False)
    invalidate_ready(output)
    atomic_json(output / 'run.json', state)
    try:
        items = inventory(root)
        adapter = adapter if adapter is not None else Adapter()
        profile, profile_hash = make_profile(adapter)
        smoke = None
        for item in items:
            raw = safe_path(root / item['relative_path']).read_bytes()
            decoded, error = decode_or_error(raw)
            if not error:
                smoke = record_for(item, digest(raw), profile_hash, decoded, None, adapter)
                break
        if smoke is None:
            raise RuntimeError('No decodable real image for mandatory GPU OCR smoke')
        print(f'Real image OCR smoke OK: {smoke["relative_path"]}; profile={profile_hash}', flush=True)
        atomic_json(output / 'profile.json', profile)
        if progress is None:
            from tqdm import tqdm
            progress = lambda rows: tqdm(rows, total=len(rows), unit='image',
                desc='OCR batch16 (including resume checks)', file=sys.stdout, mininterval=1, smoothing=.1)
        window = OPTIONS['recognition_frame_window']
        if type(window) is not int or not 1 <= window <= 16:
            raise ValueError('Recognition frame window must be from1 to16')
        pending = []
        def flush():
            if not pending:
                return
            records = records_for_batch([(item, sha, decoded, None) for item, sha, decoded in pending],
                                        profile_hash, adapter)
            for record in records:
                save_checkpoint(output, record)
            pending.clear()
        errors, skipped = 0, 0
        for item in progress(items):
            raw = safe_path(root / item['relative_path']).read_bytes()
            sha = digest(raw)
            old = cached(output, item, sha, profile_hash)
            if old and old['status'] in {'ok', 'empty'}:
                skipped += 1
                continue
            if item['id'] == smoke['id'] and sha == smoke['source_sha256']:
                save_checkpoint(output, smoke)
                continue
            decoded, error = decode_or_error(raw)
            if error:
                record = record_for(item, sha, profile_hash, None, error, adapter)
                save_checkpoint(output, record)
                errors += 1
                print(f'Image decode error: {item["relative_path"]}: {record["error"]}', flush=True)
            else:
                pending.append((item, sha, decoded))
                if len(pending) >= window:
                    flush()
        flush()
        state.update(phase='finished', complete=errors == 0, errors=errors, skipped=skipped,
                     total=len(items), profile_sha256=profile_hash, inventory=items,
                     recognition_batches=getattr(adapter, 'batch_stats', None))
        atomic_json(output / 'run.json', state)
        print(f'Finished: {len(items)} images; resumed {skipped}; decode errors {errors}', flush=True)
        print('Actual recognition batches: ' + json.dumps(state['recognition_batches']), flush=True)
    except BaseException as exc:
        state.update(phase='fatal', complete=False, error=f'{type(exc).__name__}: {exc}')
        atomic_json(output / 'run.json', state)
        traceback.print_exc()
        raise


def select_preview_items(items, count, identities=None):
    if type(count) is not int or not 1 <= count <= 50:
        raise ValueError('Preview count must be in [1,50]')
    if identities is None:
        identities = []
    if (not isinstance(identities, list) or len(identities) > 50
            or any(not isinstance(v, str) or not re.fullmatch(r'L\d{2}_V\d{3}:[1-9]\d*', v) for v in identities)
            or len(set(identities)) != len(identities)):
        raise ValueError('Preview identities must be a unique list of at most50 canonical VIDEO:n values')
    if identities:
        lookup = {item['id']: item for item in items}
        if any(v not in lookup for v in identities):
            raise ValueError('Requested preview identity absent from INPUT_ROOT; no fallback')
        return [lookup[v] for v in identities]
    count = min(count, len(items))
    return [items[(i * (len(items) - 1)) // max(1, count - 1)] for i in range(count)]


def preview(root, output, env_root, run_id, count, identities=None):
    from PIL import Image, ImageDraw
    root, output, env_root = validate_paths(root, output, env_root)
    claim_output(root, output)
    invalidate_ready(output)
    items = inventory(root)
    selected = select_preview_items(items, count, identities)
    adapter = Adapter()
    profile, profile_hash = make_profile(adapter)
    destination = output / 'previews' / run_id
    destination.mkdir(parents=True, exist_ok=False)
    records = []
    window = OPTIONS['recognition_frame_window']
    for start in range(0, len(selected), window):
        rows = []
        for item in selected[start:start + window]:
            raw = safe_path(root / item['relative_path']).read_bytes()
            decoded, error = decode_or_error(raw)
            rows.append((item, digest(raw), decoded, error))
        batch = records_for_batch(rows, profile_hash, adapter)
        for offset, (record, (_, _, decoded, error)) in enumerate(zip(batch, rows)):
            record['public_projection'] = public_record(record)
            if not error:
                image = decoded[0].copy()
                draw = ImageDraw.Draw(image)
                for i, line in enumerate(record['lines']):
                    points = [tuple(p) for p in line['polygon_padded']]
                    draw.line(points + points[:1], fill='lime' if line['public_included'] else 'red', width=2)
                    draw.text(points[0], str(i + 1), fill='yellow')
                image.save(destination / f'{start + offset:02d}.png')
            records.append(record)
    atomic_json(destination / 'preview.json', dict(schema=SCHEMA, profile=profile, records=records,
        recognition_batches=getattr(adapter, 'batch_stats', None),
        note='GPU/alphabet smoke is not an accuracy test. Review raw lines and boxes; no preview checkpoint is reused by bulk.'))
    atomic_json(output / 'latest-preview.json', dict(run_id=run_id, directory=str(destination)))
    print('Preview ready: ' + str(destination), flush=True)


_legacy_export = export_current
def export_current(root, output, env_root, run_id, archive='/kaggle/working/ocr_vietnamese_v3_batch16.zip'):
    root, output, env_root = validate_paths(root, output, env_root)
    archive = safe_path(archive)
    technical = archive.with_name(archive.stem + '_diagnostics.zip')
    if archive == technical:
        raise ValueError('Public and technical archives must be distinct')
    receipt = output / 'technical-publication.json'
    if technical.exists() and (not receipt.exists() or read_json(receipt).get('sha256') != file_digest(technical)):
        raise PublicationCollision('Unowned/changed technical ZIP; refusing overwrite')
    destination = _legacy_export(root, output, env_root, run_id, archive)
    temporary = technical.with_name('.ocr-vi-diagnostics-' + uuid.uuid4().hex + '.tmp')
    try:
        state = read_json(output / 'run.json')
        with zipfile.ZipFile(temporary, 'x', zipfile.ZIP_DEFLATED) as z:
            z.write(output / 'profile.json', 'profile.json')
            z.write(output / 'run.json', 'run.json')
            for item in state['inventory']:
                sha = file_digest(safe_path(root / item['relative_path']))
                record = cached(output, item, sha, state['profile_sha256'])
                if record is None:
                    raise RuntimeError('Technical checkpoint changed during export')
                name = f'checkpoints/{item["video_id"].split("_V")[0]}/{item["video_id"]}/{Path(item["relative_path"]).stem}.json'
                z.writestr(name, encoded(record))
        with zipfile.ZipFile(temporary) as z:
            if z.testzip() is not None:
                raise RuntimeError('Technical ZIP CRC failure')
        os.replace(temporary, technical)
        atomic_json(receipt, dict(run_id=run_id, archive=str(technical), sha256=file_digest(technical)))
    except BaseException:
        invalidate_current_ready(root, output, env_root, run_id)
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main():
    if sys.argv[1:] == ['detector-worker']:
        detector_worker()
        return
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['preview', 'run', 'export'])
    for name in ('input_root', 'output', 'env_root', 'run_id'):
        parser.add_argument(name)
    parser.add_argument('--archive', default='/kaggle/working/ocr_vietnamese_v3_batch16.zip')
    parser.add_argument('--preview-count', type=int, default=8)
    parser.add_argument('--preview-identities', default='[]')
    args = parser.parse_args()
    if not re.fullmatch(r'[a-f0-9]{32}', args.run_id):
        raise ValueError('Invalid invocation ID')
    validate_publish(args.input_root, args.output, args.env_root, args.archive)
    if args.action == 'preview':
        preview(args.input_root, args.output, args.env_root, args.run_id, args.preview_count,
                json.loads(args.preview_identities))
    elif args.action == 'run':
        output = safe_path(args.output)
        latest = read_json(output / 'latest-preview.json')
        preview_path = safe_path(latest['directory'])
        if preview_path.parent != output / 'previews':
            raise ValueError('Preview path outside owned output')
        expected = read_json(preview_path / 'preview.json')['profile']
        adapter = Adapter()
        if make_profile(adapter)[1] != digest(encoded(expected)):
            raise RuntimeError('Model/settings changed since preview; preview again before bulk')
        run(args.input_root, args.output, args.env_root, args.run_id, adapter=adapter)
    else:
        export_current(args.input_root, args.output, args.env_root, args.run_id, args.archive)


if __name__ == '__main__':
    main()
