import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.update_school_ocr_groups import REMOTE_INSTALL, install


class UpdateSchoolOCRTest(unittest.TestCase):
    def test_remote_program_compiles_no_ocr_install_or_existing_worker_stop(self):
        ast.parse(REMOTE_INSTALL)
        self.assertIn("assert sha(previous/'MANIFEST.json')",REMOTE_INSTALL)
        self.assertIn("assert old_hashes==",REMOTE_INSTALL)
        self.assertIn("source=previous/row['path']",REMOTE_INSTALL)
        self.assertIn("out.open('xb')",REMOTE_INSTALL)
        self.assertNotIn('kill(',REMOTE_INSTALL)
        self.assertNotIn('pip install',REMOTE_INSTALL)
        self.assertNotIn('rmtree',REMOTE_INSTALL)
        self.assertLess(REMOTE_INSTALL.index("assert sha(base/'import_school_ocr.py')"),
                        REMOTE_INSTALL.index('spec.loader.exec_module'))

    def test_existing_settings_refused_before_any_network_or_output_write(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);settings=root/'private.json';settings.write_text('user data')
            with patch('tools.update_school_ocr_groups.remote_python') as remote:
                with self.assertRaises(FileExistsError):install(root/'out',settings)
                remote.assert_not_called()
            self.assertEqual(settings.read_text(),'user data')
            self.assertFalse((root/'out').exists())


if __name__=='__main__':unittest.main()
