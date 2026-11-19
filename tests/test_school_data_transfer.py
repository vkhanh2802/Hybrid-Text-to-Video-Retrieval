import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from tools import transfer_school_data as transfer


class SchoolDataTransferTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project"
        self.root.mkdir()
        for name in transfer.CONFIG_FILES:
            self.write("config/" + name, b"{}")
        self.write("dataset/caption_generator/L01/L01_V001.json", b'[{"caption":"fixture"}]')
        self.write("dataset/clip-features-32/L01/L01_V001.npy", b"embedding fixture")

    def write(self, path, data):
        dest = self.root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return dest

    def test_inventory_excludes_keyframes_source_zips_and_credentials(self):
        self.write("dataset/data/L01/L01_V001/001.jpg", b"image")
        self.write("dataset/data/L01.zip", b"archive")
        self.write("dataset/objects/L01/preview.jpg", b"image")
        self.write("dataset/objects/L01/credentials.json", b"secret")
        self.write("config/drive-readonly.local.json", b"secret")
        self.write(".env", b"secret")
        plan = transfer.inventory(self.root)
        self.assertEqual(plan["file_count"], 5)
        self.assertFalse(any("secret" in row["path"] or row["path"].endswith(".jpg") for row in plan["files"]))
        self.assertFalse(any(row["path"].startswith("dataset/data/") for row in plan["files"]))

    def test_allowlist_refuses_path_traversal_absolute_private_and_nonartifact_paths(self):
        for name in ("/dataset/objects/file.json", "dataset/objects/../../.ssh/id_rsa",
                     "config/dres.local.json", "dataset/objects/hidden.key", "dataset/data/frame.npy",
                     "dataset/objects/.env", "kis/local_images.py"):
            self.assertFalse(transfer.eligible(name), name)

    def test_pack_hashes_every_file_and_leaves_sources_untouched(self):
        plan = transfer.inventory(self.root)
        before = {row["path"]: (self.root / row["path"]).read_bytes() for row in plan["files"]}
        package = transfer.pack(plan, Path(self.temp.name) / "artifacts.tar.gz")
        self.assertEqual(package["archive_sha256"], transfer.sha256_file(package["archive"]))
        with tarfile.open(package["archive"]) as archive:
            manifest = json.load(archive.extractfile("TRANSFER_MANIFEST.json"))
            self.assertEqual(len(manifest["files"]), 5)
            for row in manifest["files"]:
                self.assertEqual(row["sha256"], transfer.sha256_file(self.root / row["path"]))
                self.assertEqual(archive.extractfile(row["path"]).read(), before[row["path"]])
        self.assertEqual(before, {path: (self.root / path).read_bytes() for path in before})

    def test_changed_source_or_existing_archive_is_not_overwritten(self):
        plan = transfer.inventory(self.root)
        self.write(plan["files"][0]["path"], b"modified source")
        with self.assertRaisesRegex(RuntimeError, "Source changed"):
            transfer.pack(plan, Path(self.temp.name) / "changed.tar.gz")
        archive = Path(self.temp.name) / "keep.tar.gz"
        archive.write_bytes(b"existing user data")
        with self.assertRaises(FileExistsError):
            transfer.pack(transfer.inventory(self.root), archive)
        self.assertEqual(archive.read_bytes(), b"existing user data")

    def test_remote_verifier_checks_extracted_hashes_and_retains_archive(self):
        package = transfer.pack(transfer.inventory(self.root), Path(self.temp.name) / "source.tar.gz")
        destination = Path(self.temp.name) / "remote"
        destination.mkdir()
        (destination / "artifacts.tar.gz.partial").write_bytes(Path(package["archive"]).read_bytes())
        code = transfer.REMOTE_VERIFY.replace("CONFIG", repr(dict(package, destination=str(destination))), 1)
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt["status"], "verified")
        self.assertFalse(receipt["sources_deleted"])
        self.assertTrue((destination / "artifacts.tar.gz").is_file())
        self.assertTrue((destination / "TRANSFER_VERIFIED.json").is_file())
        self.assertTrue(Path(package["archive"]).is_file())

    def test_corrupt_remote_archive_does_not_extract_or_claim_success(self):
        destination = Path(self.temp.name) / "remote"
        destination.mkdir()
        partial = destination / "artifacts.tar.gz.partial"
        partial.write_bytes(b"corrupt")
        code = transfer.REMOTE_VERIFY.replace("CONFIG", repr(dict(
            destination=str(destination), archive_sha256="0" * 64, source_bytes=1)), 1)
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(partial.is_file())
        self.assertFalse((destination / "TRANSFER_VERIFIED.json").exists())

    def test_upload_stops_before_remote_mkdir_if_space_is_insufficient(self):
        with patch.object(transfer, "remote_python", return_value={"home": "/home/user", "free_bytes": 1}) as remote:
            with self.assertRaisesRegex(RuntimeError, "enough free space"):
                transfer.upload("user@school-host", {"source_bytes": 100, "archive_bytes": 100}, "kis-data-test")
        self.assertEqual(remote.call_count, 1)

    def test_remote_verifier_does_not_overwrite_a_preexisting_file(self):
        plan = transfer.inventory(self.root)
        package = transfer.pack(plan, Path(self.temp.name) / "source.tar.gz")
        destination = Path(self.temp.name) / "remote"
        destination.mkdir()
        original = destination / plan["files"][0]["path"]
        original.parent.mkdir(parents=True)
        original.write_bytes(b"existing user data")
        (destination / "artifacts.tar.gz.partial").write_bytes(Path(package["archive"]).read_bytes())
        code = transfer.REMOTE_VERIFY.replace("CONFIG", repr(dict(package, destination=str(destination))), 1)
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(original.read_bytes(), b"existing user data")
        self.assertFalse((destination / "TRANSFER_VERIFIED.json").exists())

    def test_remote_verifier_rejects_traversal_even_with_matching_archive_hash(self):
        destination = Path(self.temp.name) / "remote"
        destination.mkdir()
        partial = destination / "artifacts.tar.gz.partial"
        with tarfile.open(partial, "w:gz") as archive:
            member = tarfile.TarInfo("../escaped.txt")
            member.size = 4
            archive.addfile(member, io.BytesIO(b"evil"))
        package = dict(destination=str(destination), archive_sha256=transfer.sha256_file(partial), source_bytes=4)
        code = transfer.REMOTE_VERIFY.replace("CONFIG", repr(package), 1)
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((destination.parent / "escaped.txt").exists())
        self.assertFalse((destination / "TRANSFER_VERIFIED.json").exists())

    def test_existing_remote_directory_blocks_upload_before_sftp(self):
        with patch.object(transfer, "remote_python", side_effect=[
                {"home": "/home/user", "free_bytes": 10 ** 12}, RuntimeError("FileExistsError")]), \
                patch.object(transfer.subprocess, "run") as sftp:
            with self.assertRaisesRegex(RuntimeError, "FileExistsError"):
                transfer.upload("user@school-host", {"source_bytes": 100, "archive_bytes": 100}, "kis-data-test")
            sftp.assert_not_called()

    def prepare(self, destination, *, resume=False, archive_hash="a" * 64):
        config = dict(destination=str(destination), resume=resume, archive_sha256=archive_hash,
                      archive_bytes=1000, source_files=5, source_bytes=2000)
        code = transfer.REMOTE_PREPARE.replace("CONFIG", repr(config), 1)
        return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)

    def test_prepare_creates_empty_partial_for_sftp_resume_mode_on_first_upload(self):
        destination = Path(self.temp.name) / "new-remote"
        result = self.prepare(destination)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((destination / "artifacts.tar.gz.partial").stat().st_size, 0)
        self.assertEqual(json.loads(result.stdout)["resume_bytes"], 0)
        self.assertTrue((destination / "UPLOAD_PACKAGE.json").is_file())

    def test_resume_keeps_existing_partial_and_checks_package_identity(self):
        destination = Path(self.temp.name) / "remote"
        self.assertEqual(self.prepare(destination).returncode, 0)
        partial = destination / "artifacts.tar.gz.partial"
        partial.write_bytes(b"partial upload")
        result = self.prepare(destination, resume=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["resume_bytes"], len(b"partial upload"))
        mismatch = self.prepare(destination, resume=True, archive_hash="b" * 64)
        self.assertNotEqual(mismatch.returncode, 0)
        self.assertEqual(partial.read_bytes(), b"partial upload")

    def test_resume_empty_directory_from_failed_initial_upload_without_repacking(self):
        destination = Path(self.temp.name) / "remote"
        destination.mkdir()
        result = self.prepare(destination, resume=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((destination / "UPLOAD_PACKAGE.json").is_file())

    def test_resume_refuses_unmarked_partial_or_directory_with_other_data(self):
        for unknown in ("artifacts.tar.gz.partial", "keep-my-data.txt"):
            with self.subTest(unknown=unknown):
                destination = Path(self.temp.name) / unknown.replace(".", "-")
                destination.mkdir()
                existing = destination / unknown
                existing.write_bytes(b"keep")
                result = self.prepare(destination, resume=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(existing.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
