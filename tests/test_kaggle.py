import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import json
import subprocess
import sys
import zipfile

from tools.package_kaggle import FILES, KEYFRAMES_CAPTION_FILES, package


class KaggleBundleTest(unittest.TestCase):
    def test_full_runner_keeps_all_artifact_stages_in_order(self):
        from tools.kaggle_prepare import run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "output"
            root.mkdir()
            feature = root / "clip-features-32" / "L21" / "L21_V001.npy"
            feature.parent.mkdir(parents=True)
            feature.write_bytes(b"feature fixture")
            backup = root / "preparation" / "caption_backups" / "L21_V001" / "old" / "L21_V001.json"
            backup.parent.mkdir(parents=True)
            backup.write_text("[]")
            builder = Mock()
            builder.validate_video.return_value = {"status": "OK"}
            builder.artifact_paths.return_value = {"features": feature}
            with patch("tools.kaggle_prepare.configure_device", return_value={}), \
                 patch("tools.kaggle_prepare.discover_videos", return_value={"L21_V001": root / "L21_V001.mp4"}), \
                 patch("tools.kaggle_prepare.VideoDatasetBuilder", return_value=builder), \
                 patch("tools.kaggle_prepare.release_models"), \
                 patch("tools.kaggle_prepare.audit_video_formats", return_value=[{"status": "OK"}]):
                archive_path = run(root, root)
            stages = [call[0] for call in builder.method_calls[:6]]
            self.assertEqual(stages, ["extract_keyframes", "encode_visual_features", "generate_objects",
                                     "generate_captions", "build_caption_indexes", "validate_video"])
            self.assertIsNone(builder.generate_captions.call_args.kwargs["use_4bit"])
            self.assertEqual(builder.generate_captions.call_args.kwargs["batch_size"], 16)
            with zipfile.ZipFile(archive_path) as archive:
                self.assertIn("clip-features-32/L21/L21_V001.npy", archive.namelist())
                self.assertIn("preparation/caption_backups/L21_V001/old/L21_V001.json", archive.namelist())

    def test_caption_only_zip_has_correct_entrypoint_and_can_boot_in_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "caption.zip"
            package(archive_path, profile="keyframes-caption")
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(set(archive.namelist()), {f"kis_kaggle/{name}" for name in KEYFRAMES_CAPTION_FILES})
                self.assertNotIn("kis_kaggle/kis/visual.py", archive.namelist())
                self.assertNotIn("kis_kaggle/tools/ingest.py", archive.namelist())
                requirements = archive.read("kis_kaggle/requirements-kaggle.txt").decode()
                self.assertNotIn("faiss", requirements)
                self.assertNotIn("open-clip", requirements)
                archive.extractall(root / "unpacked")
            code = "import sys; from tools import kaggle_prepare; assert 'faiss' not in sys.modules; assert 'open_clip' not in sys.modules; assert 'sentence_transformers' not in sys.modules; print(kaggle_prepare.__doc__)"
            result = subprocess.run([sys.executable, "-B", "-c", code], cwd=root / "unpacked" / "kis_kaggle", capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("no retrieval stages", result.stdout)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_caption_only_runner_never_calls_feature_object_or_index_stages(self):
        import numpy
        from tools.kaggle_keyframes_caption import run

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builder = Mock(root=root)
            builder._state.return_value = {}
            builder.mapping_rows.return_value = [{"n": 1}]
            builder._caption_records.return_value = [{"keyframe_n": 1}]
            with patch("tools.kaggle_keyframes_caption.configure_device", return_value={}), \
                 patch("tools.kaggle_keyframes_caption.discover_videos", return_value={"L19_V001": root / "L19_V001.mp4"}), \
                 patch("tools.kaggle_keyframes_caption.VideoDatasetBuilder", return_value=builder), \
                 patch("tools.kaggle_keyframes_caption.resolve_caption_backend_runtime", return_value={"compute_dtype": "fp32", "use_4bit": False}), \
                 patch("tools.kaggle_keyframes_caption.archive_output", return_value=root / "out.zip"), \
                 patch("torch.cuda.empty_cache"):
                run(root, root, caption_batch=16)
            builder.extract_keyframes.assert_called_once()
            builder.generate_captions.assert_called_once()
            builder.encode_visual_features.assert_not_called()
            builder.generate_objects.assert_not_called()
            builder.build_caption_indexes.assert_not_called()
            builder.validate_video.assert_not_called()
            self.assertEqual(json.loads((root / "caption_run.json").read_text())["stages"], ["keyframes", "captions"])
            self.assertEqual(builder.generate_captions.call_args.kwargs["backend"], "internvl")

    def test_old_fp16_caption_checkpoint_is_backed_up_without_changing_frames(self):
        from kis.artifact_io import atomic_write_json
        from tools.kaggle_keyframes_caption import restart_caption_runtime_if_changed
        from tools.prepare_video_data import VideoDatasetBuilder
        from kis.structured_caption import caption_generation_config

        with tempfile.TemporaryDirectory() as directory:
            builder = VideoDatasetBuilder(directory)
            new = caption_generation_config("qwen", True, "fp32")
            old = {key: value for key, value in new.items() if key not in {"compute_dtype", "inference_revision"}}
            old["precision"] = "fp16"
            state = {"caption_generation": old, "keyframes": {"mapping_sha256": "unchanged"}}
            builder._save_state("L19_V001", state)
            caption = builder.artifact_paths("L19_V001")["captions"]
            atomic_write_json(caption, [{"preserved": "original caption"}])
            restart_caption_runtime_if_changed(builder, "L19_V001", new)
            updated = builder._state("L19_V001")
            self.assertEqual(updated["keyframes"], state["keyframes"])
            self.assertNotIn("caption_generation", updated)
            backup = builder.root / updated["caption_generation_history"][0]["backup"] / caption.name
            self.assertEqual(json.loads(backup.read_text()), [{"preserved": "original caption"}])
            self.assertFalse(caption.exists())

    def test_changed_prompt_is_not_mistaken_for_a_precision_only_migration(self):
        from kis.artifact_io import atomic_write_json
        from tools.kaggle_keyframes_caption import restart_caption_runtime_if_changed
        from tools.prepare_video_data import VideoDatasetBuilder
        from kis.structured_caption import caption_generation_config

        with tempfile.TemporaryDirectory() as directory:
            builder = VideoDatasetBuilder(directory)
            new = caption_generation_config("qwen", True, "fp32")
            old = dict(new, prompt_sha256="different-prompt")
            builder._save_state("L19_V001", {"caption_generation": old})
            caption = builder.artifact_paths("L19_V001")["captions"]
            atomic_write_json(caption, [{"preserved": "original"}])
            with self.assertRaisesRegex(ValueError, "model/prompt changed"):
                restart_caption_runtime_if_changed(builder, "L19_V001", new)
            self.assertEqual(json.loads(caption.read_text()), [{"preserved": "original"}])

    def test_caption_only_artifact_zip_excludes_existing_retrieval_artifacts(self):
        from kis.artifact_io import atomic_write_json
        from tools.kaggle_keyframes_caption import archive_output
        from tools.prepare_video_data import VideoDatasetBuilder

        with tempfile.TemporaryDirectory() as directory:
            builder = VideoDatasetBuilder(Path(directory) / "output")
            paths = builder.artifact_paths("L19_V001")
            paths["images"].mkdir(parents=True)
            (paths["images"] / "001.jpg").write_bytes(b"image")
            for key in ("mapping", "captions", "preparation", "features", "index"):
                paths[key].parent.mkdir(parents=True, exist_ok=True)
                paths[key].write_text("data")
            atomic_write_json(builder.root / "caption_run.json", {"status": "complete"})
            archive = archive_output(builder, ["L19_V001"], "complete")
            with zipfile.ZipFile(archive) as target:
                self.assertIn("data/L19/L19_V001/001.jpg", target.namelist())
                self.assertTrue(all(not name.startswith(("clip-features-32/", "objects/", "index/")) for name in target.namelist()))

    def test_bundle_has_only_required_files_and_import_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            archive_path = Path(directory) / "code.zip"
            package(archive_path)
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(set(archive.namelist()), {f"kis_kaggle/{name}" for name in FILES})
                self.assertIn("kis_kaggle/config/object_labels.json", archive.namelist())
                for name in FILES:
                    if name.endswith(".py"):
                        compile(archive.read(f"kis_kaggle/{name}"), name, "exec")

    @unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch")
    def test_cuda_configuration_fails_explicitly_if_gpu_missing(self):
        from tools.prepare_video_data import configure_device

        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "Enable Kaggle GPU"):
                configure_device("cuda", threads=1)

    @unittest.skipUnless(importlib.util.find_spec("transformers"), "requires ML dependencies")
    def test_caption_stage_forwards_cuda_and_batch_configuration(self):
        from tools.prepare_video_data import VideoDatasetBuilder

        with tempfile.TemporaryDirectory() as directory:
            builder = VideoDatasetBuilder(directory, device="cuda")
            with patch.object(builder, "mapping_rows"), patch.object(builder, "_save_state"), \
                 patch.object(builder, "_caption_records"), \
                 patch("tools.structured_caption_generator.resolve_caption_backend_runtime", return_value={"compute_dtype": "fp32", "use_4bit": False}), \
                 patch("tools.structured_caption_generator.generate_structured_captions_with_mapping") as generate:
                builder.generate_captions(["L20_V002"], batch_size=2)
            self.assertEqual(generate.call_args.kwargs["device"], "cuda")
            self.assertEqual(generate.call_args.kwargs["batch_size"], 2)
            self.assertEqual(generate.call_args.kwargs["num_workers"], 0)


if __name__ == "__main__":
    unittest.main()
