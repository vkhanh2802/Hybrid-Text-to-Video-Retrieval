from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.analyze_school_ocr_ablation import analyze, valid
from tools.school_ocr_accuracy import sha


class AnalyzeAblationTest(unittest.TestCase):
    def fixture(self,root):
        query={'id':'q','text':'visible example','literal':None,
               'positives':[{'video_id':'L01_V001','keyframe_n':1}]}
        pool=[dict(video_id='L01_V001',frame_id=i,keyframe_n=i+1,
                   vlm_status='reranked' if i<20 else 'unscored',vlm_raw_score=5 if i<20 else None)
              for i in range(100)]
        def write(name,value):
            (root/name).write_text(json.dumps(value),encoding='utf-8')
        write('QUERIES.json',[query]);write('PROTOCOL.json',{})
        original=dict(query=query,retrieval={arm:dict(pool=deepcopy(pool),rpc_seconds=.2,
            diagnostics={'ocr':{'hits':0}}) for arm in ['A','B']},
            images={f'L01_V001/{i+1}':{'status':'ready'} for i in range(20)})
        packet=dict(queries=[original],queries_sha256=sha(root/'QUERIES.json'),
                    protocol_sha256=sha(root/'PROTOCOL.json'))
        arms={arm:dict(inputs=20,scores=[5]*20,batch={'requested_batch_size':12,'oom_splits':0,
            'attempts':[{'status':'ok'},{'status':'ok'}],'effective_batches':[12,8]},
            final_ranking=deepcopy(pool),post_qwen={'first_known_positive_rank':1,'known_hit_at_1':1},seconds=1.)
              for arm in ['A','B','C']}
        arms['C']['scores'][0]=6
        write('PACKET.json',packet);write('REPORT.json',{'records':[{'query':query,'arms':arms}]})
        return arms

    def test_summary_distinguishes_changes_on_missing_ocr_and_sparse_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            with patch('builtins.print'):analyze(root)
            result=json.loads((root/'ANALYSIS.json').read_text())
            self.assertEqual(result['B_C_changed_scores'],1)
            self.assertEqual(result['B_C_changed_by_ocr_status'],{'missing':1})
            self.assertEqual(result['summary']['A']['queries'],1)
            self.assertIsNone(result['summary']['A']['qwen_median_excluding_first_query'])
            self.assertNotIn('precision',result['summary']['A'])

    def test_duplicate_results_fail_without_publishing_analysis(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            report=json.loads((root/'REPORT.json').read_text())
            report['records'][0]['arms']['A']['final_ranking'][1]=report['records'][0]['arms']['A']['final_ranking'][0]
            (root/'REPORT.json').write_text(json.dumps(report))
            with self.assertRaises(AssertionError):analyze(root)
            self.assertFalse((root/'ANALYSIS.json').exists())

    def test_invalid_unknown_scores_not_treated_as_zero(self):
        for score in [None,True,float('nan'),-1,11]:self.assertFalse(valid(score))
        self.assertTrue(valid(0));self.assertTrue(valid(10))


if __name__=='__main__':unittest.main()
