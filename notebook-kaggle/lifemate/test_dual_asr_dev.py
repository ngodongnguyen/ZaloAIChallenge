"""CPU checks for the dual-ASR data boundary and paired score comparison."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent
SOURCE = HERE / 'agent-dual-asr-dev/dual_asr.py'


class DualAsrTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SOURCE.exists():
            return
        spec = importlib.util.spec_from_file_location('dual_asr_tested', SOURCE)
        cls.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.mod)

    def setUp(self):
        self.assertTrue(SOURCE.exists(), 'dual-ASR implementation is missing')

    def test_agreement_preserves_primary_exactly_but_disagreement_retains_both(self):
        primary = {'final.wav': 'Huỷ vé tàu', 'history.wav': '  Gọi Mẹ  '}
        secondary = {'final.wav': 'Đổi vé tàu', 'history.wav': 'gọi mẹ'}
        result = self.mod.combine_transcripts(primary, secondary, ['final.wav', 'history.wav'])
        self.assertEqual(result['history.wav'], '  Gọi Mẹ  ')
        self.assertIn('Huỷ vé tàu', result['final.wav'])
        self.assertIn('Đổi vé tàu', result['final.wav'])
        self.assertIn('Whisper', result['final.wav'])
        self.assertIn('Gipformer', result['final.wav'])
        self.assertIn('CÙNG MỘT', result['final.wav'])
        self.assertEqual(primary['final.wav'], 'Huỷ vé tàu')
        self.assertNotIn('unused.wav', result)

    def test_missing_history_and_empty_transcripts_fail_instead_of_filling_gold(self):
        for p, s in [({'f': 'ASR'}, {'f': 'OTHER'}),
                     ({'f': 'ASR', 'h': '  '}, {'f': 'OTHER', 'h': 'text'})]:
            with self.assertRaisesRegex(ValueError, 'h'):
                self.mod.combine_transcripts(p, s, ['f', 'h'])

    def test_cached_control_must_match_checkpoint_audio_and_every_id(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            conf = {'checkpoint_sha256': 'weights', 'asr_source_sha256': 'cache', 'subset': ['a', 'b']}
            (root/'config.json').write_text(json.dumps(conf))
            records = [{'id': 'a', 'score': 0.5, 'pending': False}, {'id': 'b', 'score': 1.0, 'pending': False}]
            def write():
                (root/'per_id_scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            write()
            got = self.mod.checked_control(root, ['a','b'], 'weights', 'cache')
            self.assertEqual(got['a']['score'], 0.5)
            for checkpoint, cache in [('wrong','cache'),('weights','wrong')]:
                with self.assertRaises(ValueError):
                    self.mod.checked_control(root, ['a','b'], checkpoint, cache)
            records[1]['id'] = 'a'; write()
            with self.assertRaises(ValueError):
                self.mod.checked_control(root, ['a','b'], 'weights', 'cache')
            records[1]['id'] = 'b'; records[1]['pending'] = True; write()
            with self.assertRaises(ValueError):
                self.mod.checked_control(root, ['a','b'], 'weights', 'cache')

    def test_paired_report_counts_regressions_and_rejects_incomplete_scores(self):
        control = {'a': {'score': 0.0}, 'b': {'score': 1.0}, 'c': {'score': 0.5}}
        candidate = {'a': {'score': 1.0}, 'b': {'score': 0.5}, 'c': {'score': 0.5}}
        report = self.mod.paired_report(candidate, control)
        self.assertAlmostEqual(report['delta_points'], 100/6)
        self.assertEqual((report['improved'],report['worsened'],report['unchanged']), (1,1,1))
        self.assertEqual(report, self.mod.paired_report(candidate, control))
        for changed in [{'a': {'score': 0}},dict(candidate,b={'score': float('nan')}),dict(candidate,b={'score': 1.1})]:
            with self.assertRaises(ValueError):
                self.mod.paired_report(changed, control)


    def test_notebook_preflight_feeds_both_final_and_history_to_the_worker(self):
        import contextlib
        import copy
        import hashlib
        import importlib
        import io
        import sys
        nb = json.loads((HERE/'agent-dual-asr-dev/lifemate_agent_dual_asr_dev.ipynb').read_text())
        saved = {k:sys.modules.pop(k) for k in list(sys.modules) if k=='lmlib' or k.startswith('lmlib.')}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); lib = root/'lib'; inputs = root/'input'; out = root/'out'; out.mkdir()
            for cell in nb['cells']:
                source = ''.join(cell['source'])
                if source.startswith('%%writefile /kaggle/working/lib/'):
                    head, body = source.split('\n', 1)
                    path = lib/head.split('/kaggle/working/lib/',1)[1]
                    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(body)
            sys.path.insert(0,str(lib))
            try:
                ns = {'Path':Path,'json':json,'sys':sys,'LIB':lib,'OUT':out}
                exec(''.join(nb['cells'][1]['source']),ns)
                ns['CONFIG'].update(input_roots=[str(inputs)], subset=['a','b'],expected_samples=2,expected_audio=3)
                rows = [{'id':'a','audio':'final.wav','audio_text':'SECRET GOLD','label':{'tool_calls':[]},
                         'context':{},'history':[{'role':'user','audio':'history.wav','audio_text':'SECRET HISTORY'}]},
                        {'id':'b','audio':'other.wav','context':{},'history':[]}]
                split = inputs/'dataset/train'; split.mkdir(parents=True)
                (split/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                checkpoint = inputs/'agent/lm_agent/sft_ep2/ckpt-000380'; checkpoint.mkdir(parents=True)
                for name in ['COMPLETE','adapter_state.pt','adapter_config.json']:(checkpoint/name).write_text('{}')
                checkpoint_sha = hashlib.sha256(b'{}').hexdigest()
                for name,relative,tag,text in [
                    ('whisper_medium','asr_ft/medium/asr_train_medium.shard0.jsonl','asrft-medium-ep5','WHISPER'),
                    ('gipft_lr15','gip_ft/lr15/asr_train_gipft_lr15.shard0.jsonl','gipft-lr15-ep4','GIP')]:
                    path=inputs/name/relative;path.parent.mkdir(parents=True)
                    records=[{'_header':{'run_hash':tag}}]+[{'id':a,'status':'completed','text':text+' '+a} for a in ['final.wav','history.wav','other.wav']]
                    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
                    control=inputs/'control/lm_asr_ft_eval'/f'eval_{name}';control.mkdir(parents=True)
                    (control/'config.json').write_text(json.dumps({'checkpoint_sha256':checkpoint_sha,
                        'asr_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'subset':['a','b']}))
                    (control/'per_id_scores.jsonl').write_text(''.join(json.dumps({'id':qid,'score':0.5,'pending':False})+'\n' for qid in ['a','b']))
                source=next(''.join(c['source']) for c in nb['cells'] if 'PREPARED = prepare_inputs(CONFIG, OUT)' in ''.join(c['source']))
                with contextlib.redirect_stdout(io.StringIO()):exec(source.split('gpu_info =',1)[0],ns)
                run=importlib.import_module('lmlib.run'); agent=importlib.import_module('lmlib.agent')
                texts=run.transcripts_for(ns['CONFIG'],rows)
                messages=agent.build_messages(run.inference_row(rows[0],'e2e'),texts['final.wav'],agent.Catalog({'apps':[]}),texts,[])
                prompt=json.dumps(messages,ensure_ascii=False)
                for expected in ['WHISPER final.wav','GIP final.wav','WHISPER history.wav','GIP history.wav']:
                    self.assertIn(expected,prompt)
                self.assertNotIn('SECRET',prompt)
                self.assertEqual(set(texts),{'final.wav','history.wav','other.wav'})
            finally:
                sys.path.remove(str(lib))
                for k in list(sys.modules):
                    if k=='lmlib' or k.startswith('lmlib.'):del sys.modules[k]
                sys.modules.update(saved)


if __name__ == '__main__':
    unittest.main()
