"""Exercise the actual worker callback using the upstream keyword call contract."""
import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from architectures.seer.adapters.latentloop_freshness_controls import campaign


class WorkerContractTests(unittest.TestCase):
    def test_upstream_keyword_call_reaches_runtime_in_every_stage(self):
        tree = ast.parse((ROOT / 'architectures/seer/upstream/eval_libero.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'eval_one_epoch_libero_ddp']
        self.assertTrue(calls)
        self.assertEqual({kw.arg for kw in calls[0].keywords},
                         {'args', 'model', 'image_processor', 'tokenizer'})
        for stage in ('smoke', 'calibration') + campaign.ROWS:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                config = {'initial_state_hashes': {}, 'libero': '/unused/libero',
                          'seed': 42, 'max_steps': 600,
                          'assets': {'vit': 'vit', 'teacher': 'teacher', 'adapter': 'adapter'}}
                (root / 'config.json').write_text(json.dumps(config))
                (root / 'source_hashes.json').write_text('{}')
                evaluator = ModuleType('eval_libero')
                runtime = ModuleType('architectures.seer.adapters.latentloop_freshness_controls.runtime')
                runtime.evaluate_loaded = Mock()
                values = {key: object() for key in ('args', 'model', 'image_processor', 'tokenizer')}

                def upstream_main():
                    self.assertEqual(sys.argv[sys.argv.index('--seed') + 1],
                                     '4242' if stage == 'calibration' else '42')
                    evaluator.eval_one_epoch_libero_ddp(**values)

                evaluator.main = upstream_main
                with patch.object(campaign, 'sources', return_value={}), \
                     patch.dict(sys.modules, {'eval_libero': evaluator, runtime.__name__: runtime}), \
                     patch.object(sys, 'path', list(sys.path)), \
                     patch.object(sys, 'argv', list(sys.argv)), \
                     patch('torch.distributed.is_initialized', return_value=False):
                    campaign.worker(SimpleNamespace(config=str(root / 'config.json'), stage=stage))
                runtime.evaluate_loaded.assert_called_once_with(
                    values['args'], values['model'], values['image_processor'], values['tokenizer'],
                    config, stage, root / stage)


if __name__ == '__main__':
    unittest.main()
