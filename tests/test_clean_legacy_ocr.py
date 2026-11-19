import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from tools.clean_legacy_ocr import inventory_tree,delete_inventoried_tree,strip_legacy_ocr_from_transfer,sha


class CleanLegacyOCRTest(unittest.TestCase):
    def test_stream_hash_works_without_python311_file_digest(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'value';p.write_bytes(b'bytes-for-python310')
            with patch('hashlib.file_digest',None,create=True):
                self.assertEqual(sha(p),hashlib.sha256(p.read_bytes()).hexdigest())

    def archive(self,base,corrupt=False):
        p=base/'artifacts.tar.gz'
        data={'dataset/ocr/L02/L02_V001/001.json':b'old OCR',
              'dataset/caption_generator/L02/a.json':b'original caption',
              'dataset/mapping/L02/a.csv':b'n,frame_idx\n1,500\n',
              'config/pipeline.json':b'{}'}
        manifest=dict(schema_version=1,files=[dict(path=n,size=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for n,raw in data.items()],
            file_count=len(data),bytes=sum(map(len,data.values())),components={'ocr':1,'caption_generator':1,'mapping':1})
        if corrupt:manifest['files'][1]['sha256']='0'*64
        with tarfile.open(p,'w:gz') as t:
            for name,raw in {**data,'TRANSFER_MANIFEST.json':json.dumps(manifest).encode()}.items():
                info=tarfile.TarInfo(name);info.size=len(raw);t.addfile(info,io.BytesIO(raw))
        return p,data

    def test_mixed_archive_removes_only_ocr_and_preserves_every_other_byte(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);p,data=self.archive(base)
            receipt=strip_legacy_ocr_from_transfer(p,base/'audit')
            self.assertEqual(receipt['removed_OCR_members'],1)
            with tarfile.open(p,'r:gz') as t:
                self.assertFalse(any(n.startswith('dataset/ocr/') for n in t.getnames()))
                for n,raw in data.items():
                    if '/ocr/' not in n:self.assertEqual(t.extractfile(n).read(),raw)
                m=json.load(t.extractfile('TRANSFER_MANIFEST.json'))
                self.assertEqual(m['file_count'],3);self.assertNotIn('ocr',m['components'])
            self.assertEqual(sha(p),receipt['new_sha256'])

    def test_corrupt_manifest_does_not_replace_source(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);p,_=self.archive(base,True);before=p.read_bytes()
            with self.assertRaises(ValueError):strip_legacy_ocr_from_transfer(p,base/'audit')
            self.assertEqual(p.read_bytes(),before)
            self.assertFalse(any(base.glob('.no-legacy-ocr-*')))

    def test_deletion_is_exact_allowlisted_and_changed_tree_is_not_removed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve()/'old-ocr';root.mkdir();p=root/'001.json';p.write_bytes(b'old')
            plan=inventory_tree(root,[root])
            with self.assertRaises(ValueError):delete_inventoried_tree(plan,[])
            p.write_bytes(b'new user work')
            with self.assertRaises(ValueError):delete_inventoried_tree(plan,[root])
            self.assertEqual(p.read_bytes(),b'new user work')
            plan=inventory_tree(root,[root]);receipt=delete_inventoried_tree(plan,[root])
            self.assertEqual(receipt['deleted_files'],1);self.assertFalse(root.exists())

    def test_unexpected_file_type_stops_before_delete(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();(root/'photo.jpg').write_bytes(b'protected image')
            with self.assertRaises(ValueError):inventory_tree(root,[root])
            self.assertTrue((root/'photo.jpg').exists())


if __name__=='__main__':unittest.main()
