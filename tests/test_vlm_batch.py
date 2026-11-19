from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy
import torch
from PIL import Image

from kis.solve_kis import rerank_candidates_with_vlm
from kis.vlm_reranker import VLMReranker


class VLMBatchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = []
        for i in range(20):
            path = Path(self.temp.name) / f"{i:03d}.png"
            Image.new("RGB", (8, 8), (i, 0, 0)).save(path)
            self.paths.append(path)
        self.scorer = VLMReranker.__new__(VLMReranker)
        self.scorer.batch_size = 20
        self.scorer.device = torch.device("cpu")

    def test_twenty_rows_really_use_one_generate_and_keep_caption_order(self):
        def generate(images, prompts):
            self.assertEqual(len(images), 20)
            for i, (image, prompt) in enumerate(zip(images, prompts)):
                self.assertEqual(image.getpixel((0, 0))[0], i)
                self.assertIn(f"caption-{i}\n</CAPTION>", prompt)
            return list(range(20))
        self.scorer._generate_batch = Mock(side_effect=generate)
        values = self.scorer.score_batch(self.paths, "query", [f"caption-{i}" for i in range(20)], oom_fallback=False)
        self.assertEqual(values, list(range(20)))
        self.assertEqual(self.scorer._generate_batch.call_count, 1)
        self.assertEqual(self.scorer.last_batch_stats["effective_batches"], [20])

    def test_chunk_sizes_are_real_not_twenty_scalar_calls(self):
        self.scorer._generate_batch = Mock(side_effect=lambda images, prompts: [5] * len(images))
        self.assertEqual(self.scorer.score_batch(self.paths, "query", batch_size=8), [5] * 20)
        self.assertEqual([len(call.args[0]) for call in self.scorer._generate_batch.call_args_list], [8, 8, 4])

    def test_strict_benchmark_does_not_hide_oom_with_subbatches(self):
        self.scorer._generate_batch = Mock(side_effect=torch.cuda.OutOfMemoryError("synthetic test OOM"))
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            self.scorer.score_batch(self.paths, "query", oom_fallback=False)
        self.assertEqual(self.scorer._generate_batch.call_count, 1)
        self.assertEqual(self.scorer.last_batch_stats["attempts"][0]["status"], "oom")

    def test_production_oom_split_keeps_all_original_positions(self):
        def generate(images, prompts):
            if len(images) > 4:
                raise torch.cuda.OutOfMemoryError("synthetic test OOM")
            return [image.getpixel((0, 0))[0] for image in images]
        self.scorer._generate_batch = Mock(side_effect=generate)
        self.assertEqual(self.scorer.score_batch(self.paths, "query"), list(range(20)))
        self.assertGreater(self.scorer.last_batch_stats["oom_splits"], 0)
        self.assertEqual(sum(self.scorer.last_batch_stats["effective_batches"]), 20)
        self.assertTrue(all(size <= 4 for size in self.scorer.last_batch_stats["effective_batches"]))

    def test_bad_image_does_not_shift_scores_or_poison_valid_images(self):
        self.scorer._generate_batch = Mock(side_effect=lambda images, prompts: [im.getpixel((0, 0))[0] for im in images])
        self.paths[3].write_bytes(b"broken")
        values = self.scorer.score_batch(self.paths, "query")
        self.assertEqual(values[2:5], [2, None, 4])
        self.assertEqual(self.scorer.last_batch_errors[3], "image_open_failed")

    def test_candidate_budget_and_parallel_fields_are_validated(self):
        for paths, captions in [(self.paths + self.paths[:1], None), (self.paths, ["short"])]:
            with self.assertRaises(ValueError):
                self.scorer.score_batch(paths, "query", captions)

    def test_tensor_batch_has_independent_conversations_and_per_row_strict_parse(self):
        processor = Mock(return_value={"input_ids": torch.tensor([[0, 1, 2], [1, 2, 3]])})
        processor.apply_chat_template.side_effect = lambda messages, **kwargs: messages[0]["content"][1]["text"]
        processor.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2, padding_side="left")
        processor.batch_decode.return_value = ["8", "8 because good"]
        self.scorer.processor = processor
        self.scorer.model = SimpleNamespace(device="cpu", generate=Mock(return_value=torch.tensor([[0, 1, 2, 8], [1, 2, 3, 8]])))
        with Image.open(self.paths[0]) as a, Image.open(self.paths[1]) as b:
            values = self.scorer._generate_batch([a, b], ["prompt-a", "prompt-b"])
        self.assertEqual(values, [8, None])
        self.assertEqual(processor.call_args.kwargs["text"], ["prompt-a", "prompt-b"])
        self.assertEqual(len(processor.call_args.kwargs["images"]), 2)
        self.assertEqual(processor.batch_decode.call_args.args[0].shape, (2, 1))
        self.scorer.model.generate.assert_called_once()

    def test_decoder_padding_is_left_and_batch_setting_does_not_load_cpu_by_accident(self):
        processor = SimpleNamespace(tokenizer=SimpleNamespace(padding_side="right"))
        model = Mock()
        with patch.dict("os.environ", {"KIS_VLM_BATCH_SIZE": "8"}), \
                patch("kis.vlm_reranker.torch.cuda.is_available", return_value=False), \
                patch("kis.vlm_reranker.AutoProcessor.from_pretrained", return_value=processor), \
                patch("kis.vlm_reranker.Qwen2_5_VLForConditionalGeneration.from_pretrained", return_value=model):
            scorer = VLMReranker(use_4bit=False, allow_cpu=True)
        self.assertEqual(processor.tokenizer.padding_side, "left")
        self.assertEqual(scorer.batch_size, 8)


class BatchPipelineTest(unittest.TestCase):
    def test_batch_forwards_separate_ocr_per_image_without_caption_merge(self):
        rows = [{"video_id": "L01_V001", "frame_id": i, "score": .9,
                 "retrieval_text": "original caption", "ocr": {"source_format": "public_json", "text": f"text-{i}",
                 "verified": False, "status": "ok"}} for i in range(2)]
        scorer = SimpleNamespace(supports_batch=True, supports_ocr=True, score_batch=Mock(return_value=[5, 7]))
        with patch("kis.solve_kis._resolve_candidate_image_path", return_value=Path("fixture.jpg")):
            rerank_candidates_with_vlm(rows, 'Find visible text "text-0" or "text-1"', scorer, None)
        self.assertEqual(scorer.score_batch.call_args.args[2], ["original caption"] * 2)
        self.assertEqual([r["text"] for r in scorer.score_batch.call_args.kwargs["ocrs"]], ["text-0", "text-1"])

    def test_unrelated_and_missing_ocr_preserve_plain_batch_prompt(self):
        rows = [{"video_id": "L01_V001", "frame_id": i, "score": .9,
                 "ocr": {"source_format": "public_json", "text": "ticker", "status": s}}
                for i, s in enumerate(['ok', 'missing'])]
        scorer = SimpleNamespace(supports_batch=True, supports_ocr=True, score_batch=Mock(return_value=[5, 7]))
        with patch('kis.solve_kis._resolve_candidate_image_path', return_value=Path('fixture.jpg')):
            rerank_candidates_with_vlm(rows, 'A person driving', scorer, None)
        self.assertNotIn('ocrs', scorer.score_batch.call_args.kwargs)
        self.assertTrue(all(r['vlm_ocr_used'] is False for r in rows))

    def test_lazy_loader_restores_inference_threads_even_on_failure(self):
        from kis.web_app import LazyVLMReranker
        factory = Mock(side_effect=RuntimeError("synthetic loader failure"))
        with patch.dict("sys.modules", {"kis.vlm_reranker": SimpleNamespace(VLMReranker=factory)}), \
                patch("torch.get_num_threads", return_value=4), patch("torch.set_num_threads") as set_threads:
            lazy = LazyVLMReranker()
            with self.assertRaisesRegex(RuntimeError, "synthetic loader failure"):
                lazy.score_batch([], "query")
        self.assertEqual([call.args[0] for call in set_threads.call_args_list], [1, 4])

    def test_streams_stay_open_for_batch_and_missing_never_backfilled(self):
        opened = []
        class Source:
            @contextmanager
            def open_candidate(self, row, lookup):
                if row["frame_id"] == 0:
                    raise FileNotFoundError()
                with BytesIO(b"fixture") as stream:
                    opened.append(stream)
                    yield stream
        class Scorer:
            supports_batch = True
            supports_ocr = True
            def score_batch(self, paths, query, captions):
                self.paths = paths
                self.captions = captions
                self.assertions = all(not p.closed for p in paths)
                return [7] * len(paths)
        candidates = [{"video_id": "L01_V001", "frame_id": i, "score": 1 - i / 101,
                       "retrieval_text": f"caption-{i}"} for i in range(100)]
        scorer = Scorer()
        result = rerank_candidates_with_vlm(candidates, "query", scorer, None, image_source=Source(), top_k=40)
        self.assertTrue(scorer.assertions)
        self.assertEqual(len(scorer.paths), 19)
        self.assertEqual(scorer.captions, [f"caption-{i}" for i in range(1, 20)])
        self.assertTrue(all(p.closed for p in opened))
        by_id = {row["frame_id"]: row for row in result}
        self.assertEqual(by_id[0]["vlm_status"], "missing_image")
        self.assertTrue(all(by_id[i]["vlm_status"] == "unscored" for i in range(20, 100)))

    def test_bad_batch_count_is_error_not_a_scalar_retry_or_fabricated_scores(self):
        scorer = SimpleNamespace(supports_batch=True, score_batch=Mock(return_value=[]), score=Mock())
        with patch("kis.solve_kis._resolve_candidate_image_path", return_value=Path("fixture.jpg")):
            result = rerank_candidates_with_vlm([{"video_id": "L01_V001", "frame_id": 1, "score": .8}], "query", scorer, None)
        self.assertEqual(result[0]["vlm_status"], "error")
        self.assertIsNone(result[0]["vlm_raw_score"])
        scorer.score.assert_not_called()


if __name__ == "__main__":
    unittest.main()
