"""Actual CPU tensor batching with toy logits, never real OCR weights/GPU."""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from tools.build_kaggle_ocr_vietnamese_notebook import ORIGINAL, worker_source


class Batch16Test(unittest.TestCase):
    def setUp(self):
        self.ns = {'__name__': 'batch_test', '__file__': str(ORIGINAL)}
        exec(compile(worker_source()[3], '<batch-worker>', 'exec'), self.ns)
        self.adapter = self.ns['Adapter'].__new__(self.ns['Adapter'])
        self.adapter.engine = dict(test_only=True)
        self.adapter.vocab = SimpleNamespace(go=1, eos=2,
            decode=lambda ids: '|'.join(str(i-4) for i in ids if i >= 4))
        self.encoder_shapes, self.decoder_shapes = [], []
        self.endless, self.invalid = set(), set()
        def cnn(images):
            self.encoder_shapes.append(tuple(images.shape))
            marker = (images[:, 0, 0, 0] * 255).round()
            return marker.reshape(1, -1, 1)
        def decoder(tokens, memory):
            length, batch = tokens.shape
            self.decoder_shapes.append((length, batch))
            logits = torch.full((batch, length, 260), -20.)
            for i in range(batch):
                marker = int(memory[0, i, 0])
                token = 0 if marker in self.invalid else 4 + marker
                if marker not in self.endless and length > marker % 3 + 1:
                    token = 2
                logits[i, -1, token] = 20.
            return logits, memory
        self.adapter.model = SimpleNamespace(cnn=cnn,
            transformer=SimpleNamespace(forward_encoder=lambda x: x, forward_decoder=decoder))

    @contextmanager
    def cpu(self):
        real_tensor, real_to = torch.tensor, torch.Tensor.to
        def tensor(*args, **kwargs):
            kwargs.pop('device', None)
            return real_tensor(*args, **kwargs)
        def to(value, *args, **kwargs):
            return value if args and args[0] == 'cuda:0' else real_to(value, *args, **kwargs)
        with patch('torch.tensor', side_effect=tensor), patch.object(torch.Tensor, 'to', to):
            yield

    def crop(self, marker, width=140):
        return np.full((32, width, 3), marker, dtype=np.uint8)

    def test_real_batch_tensor_16_and_tail_5_maintain_order_and_independent_eos(self):
        with self.cpu():
            results = self.adapter.recognize_crops([self.crop(i) for i in range(37)])
        self.assertEqual([s[0] for s in self.encoder_shapes], [16, 16, 5])
        self.assertEqual(len(results), 37)
        for i, row in enumerate(results):
            self.assertEqual(row['text'], '|'.join([str(i)] * (i % 3 + 1)))
            self.assertEqual(row['decoded_tokens'], i % 3 + 1)
            self.assertTrue(row['decoder_eos'])
            self.assertFalse(row['truncated'])
        self.assertEqual(self.adapter.batch_stats['batch_size_histogram'], {'16': 2, '5': 1})

    def test_exact_width_buckets_no_input_padding_and_restore_original_order(self):
        widths = [140, 70, 140, 200, 70]
        with self.cpu():
            results = self.adapter.recognize_crops([self.crop(i, w) for i,w in enumerate(widths)])
        self.assertEqual([(s[0], s[3]) for s in self.encoder_shapes], [(2, 140), (2, 70), (1, 200)])
        self.assertEqual([r['text'].split('|')[0] for r in results], list(map(str, range(5))))
        self.assertEqual([r['recognizer_size'][0] for r in results], widths)

    def test_batch1_and_batch16_toy_decoder_identical_text_confidence_flags(self):
        crops = [self.crop(i, 100 if i%2 else 140) for i in range(20)]
        with self.cpu(), patch.dict(self.ns['OPTIONS'], recognition_batch_size=1):
            scalar = self.adapter.recognize_crops(crops)
        with self.cpu():
            batched = self.adapter.recognize_crops(crops)
        for a,b in zip(scalar,batched):
            for key in ['text','confidence','truncated','decoder_eos','decoded_tokens','recognizer_size']:
                self.assertEqual(a[key],b[key])

    def test_finished_rows_removed_and_token_cap_applies_only_to_unfinished(self):
        self.endless.add(1)
        with self.cpu(), patch.dict(self.ns['OPTIONS'], recognition_max_tokens=4):
            rows = self.adapter.recognize_crops([self.crop(0), self.crop(1)])
        self.assertEqual(self.decoder_shapes, [(1,2),(2,2),(3,1),(4,1)])
        self.assertFalse(rows[0]['truncated'])
        self.assertTrue(rows[1]['truncated'])
        self.assertEqual(rows[0]['decoded_tokens'],1)
        self.assertEqual(rows[1]['decoded_tokens'],4)

    def test_invalid_token_and_oom_are_not_silently_empty_or_smaller_batch(self):
        self.invalid.add(3)
        with self.cpu(), self.assertRaisesRegex(RuntimeError, 'non-text token'):
            self.adapter.recognize_crops([self.crop(3)])
        def oom(_):
            raise torch.cuda.OutOfMemoryError('synthetic OOM')
        self.adapter.model.cnn = oom
        with self.cpu(), self.assertRaisesRegex(RuntimeError, 'batch.*16|Batch.*16'):
            self.adapter.recognize_crops([self.crop(i) for i in range(16)])

    def test_many_frames_pack_same_width_into_one_batch_and_scatter_identity(self):
        self.adapter.detector = SimpleNamespace(detect=lambda image: (
            [[[0,0],[2,0],[2,2],[0,2]]], [self.crop(int(image[0,0,0]))]))
        frames = [Image.new('RGB',(4,4),(i,i,i)) for i in range(16)]
        with self.cpu():
            rows = self.adapter.recognize_many(frames)
        self.assertEqual([s[0] for s in self.encoder_shapes],[16])
        self.assertEqual([r[0]['text'].split('|')[0] for r in rows],list(map(str,range(16))))
        self.assertTrue(all(r[0]['polygon_padded']==[[0,0],[2,0],[2,2],[0,2]] for r in rows))

    def test_empty_inputs_and_empty_detection_no_phantom_slots(self):
        with self.cpu():
            self.assertEqual(self.adapter.recognize_crops([]),[])
            self.adapter.detector = SimpleNamespace(detect=lambda _: ([],[]))
            self.assertEqual(self.adapter.recognize_many([Image.new('RGB',(2,2))]),[[]])
        self.assertEqual(self.encoder_shapes,[])

    def test_nonfinite_decoder_and_frame_count_mismatch_fail_explicitly(self):
        self.adapter.model.transformer.forward_decoder = lambda tokens, memory: (
            torch.full((tokens.shape[1],tokens.shape[0],260),float('nan')),memory)
        with self.cpu(), self.assertRaisesRegex(RuntimeError,'Non-finite'):
            self.adapter.recognize_crops([self.crop(1)])
        item=dict(id='L01_V001:1',video_id='L01_V001',keyframe_n=1,relative_path='L01_V001/001.png')
        decoded=(Image.new('RGB',(66,66)),(2,2))
        with self.assertRaisesRegex(RuntimeError,'Frame batch reply count mismatch'):
            self.ns['records_for_batch']([(item,'a'*64,decoded,None)],'b'*64,
                SimpleNamespace(recognize_many=lambda images: []))

    def test_failed_window_keeps_previous_checkpoint_and_blocks_completion(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);root=base/'input';video=root/'L01_V001';video.mkdir(parents=True)
            for i in range(1,18):
                Image.new('RGB',(2,2),(i,i,i)).save(video/f'{i:03d}.png')
            class Failure:
                engine={'test_failure':True}
                def recognize(self,image):
                    return [dict(text='smoke',confidence=.9,polygon_padded=[[32,32]]*4)]
                def recognize_many(self,images):
                    raise RuntimeError('synthetic batch backend failure')
            output=base/'state'
            with patch('builtins.print'), patch('traceback.print_exc'), \
                    self.assertRaisesRegex(RuntimeError,'batch backend failure'):
                self.ns['run'](root,output,base/'env','c'*32,adapter=Failure(),progress=lambda x:x)
            state=json.loads((output/'run.json').read_text())
            self.assertEqual(state['phase'],'fatal')
            self.assertFalse(state['complete'])
            self.assertFalse((output/'ready.json').exists())
            checkpoints=list((output/'checkpoints').glob('*.json'))
            self.assertEqual(len(checkpoints),1)
            self.assertEqual(json.loads(checkpoints[0].read_text())['record']['lines'][0]['text'],'smoke')

    def test_preview_buffers_frames_and_keeps_decode_errors_in_identity_order(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);root=base/'input';video=root/'L01_V001';video.mkdir(parents=True)
            for i in range(1,19):
                Image.new('RGB',(2,2),(i,i,i)).save(video/f'{i:03d}.png')
            (video/'007.png').write_bytes(b'bad-image')
            windows=[]
            class PreviewBatch:
                engine={'test_preview':True}
                def recognize_many(self,images):
                    windows.append(len(images))
                    return [[dict(text=str(im.getpixel((32,32))[0]),confidence=.9,
                        polygon_padded=[[32,32],[34,32],[34,34],[32,34]])] for im in images]
            output=base/'state'
            with patch.dict(self.ns,Adapter=PreviewBatch), patch('builtins.print'):
                self.ns['preview'](root,output,base/'env','d'*32,18)
            self.assertEqual(windows,[15,2])
            data=json.loads((output/'previews'/('d'*32)/'preview.json').read_text())
            self.assertEqual([r['keyframe_n'] for r in data['records']],list(range(1,19)))
            self.assertEqual(data['records'][6]['status'],'error')
            self.assertEqual(data['records'][7]['lines'][0]['text'],'8')
            self.assertFalse((output/'checkpoints').exists())
            self.assertFalse((output/'ready.json').exists())

    def test_run_frame_windows_resume_and_error_checkpoint_integrity(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);root=base/'input';video=root/'L01_V001';video.mkdir(parents=True)
            for i in range(1,35):
                Image.new('RGB',(2,2),(i,i,i)).save(video/f'{i:03d}.png')
            (video/'035.png').write_bytes(b'not-an-image')
            windows=[]
            class FakeBatch:
                engine={'fake_batch':True}
                def recognize(self, image):return self.recognize_many([image])[0]
                def recognize_many(self, images):
                    windows.append(len(images))
                    return [[dict(text=str(im.getpixel((32,32))[0]),confidence=.9,
                        polygon_padded=[[32,32],[34,32],[34,34],[32,34]])] for im in images]
            output,env=base/'state',base/'env'
            with patch('builtins.print'):
                self.ns['run'](root,output,env,'a'*32,adapter=FakeBatch(),progress=lambda rows:rows)
            self.assertEqual(windows,[1,16,16,1])
            state=json.loads((output/'run.json').read_text())
            self.assertEqual(state['errors'],1)
            self.assertEqual(state['total'],35)
            self.assertFalse(state['complete'])
            for n in range(1,35):
                item=self.ns['inventory'](root)[n-1]
                record=self.ns['cached'](output,item,self.ns['file_digest'](video/f'{n:03d}.png'),state['profile_sha256'])
                self.assertEqual(record['lines'][0]['text'],str(n))
            windows.clear()
            with patch('builtins.print'):
                self.ns['run'](root,output,env,'b'*32,adapter=FakeBatch(),progress=lambda rows:rows)
            self.assertEqual(windows,[1])  # mandatory real-image smoke; remaining cached
            state=json.loads((output/'run.json').read_text())
            self.assertEqual(state['skipped'],34)


if __name__=='__main__':
    unittest.main()
