import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from architectures.seer.adapters.latentloop_freshness_confirmation import campaign
from architectures.seer.adapters.latentloop_freshness_confirmation.control import (
    NEW_ROW, ROWS, SEEDS, experiment_plan, fixed_gate_cached_update)
from architectures.seer.adapters.latentloop_freshness.interventions import replaced_method

spec = importlib.util.spec_from_file_location('confirmation_native', ROOT / 'architectures/seer/upstream/models/lrnode_modules.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.node = native.ControlledLatentNODE(24,motion_dim=8,hidden_dim=16,action_pred_steps=3)
        self.model = SimpleNamespace(lrnode_dynamics=self.node,lrnode_apply_dynamics=self.node)
        self.z,self.d = torch.randn(1,3,24),torch.randn(1,8)
        self.gates = {'1':.1148,'2':.1082,'3':.1025}

    def test_formula_preserves_feature_and_replaces_only_gate(self):
        for age in (1,2,3):
            self.node(self.z,self.d,dt=1.,age=age)
            residual = self.node.last_dz.clone()
            actual = fixed_gate_cached_update(self.model,self.z,self.d,age,self.gates)
            gate = torch.full_like(self.z[...,:1],self.gates[str(age)])
            self.assertTrue(torch.equal(actual,self.z+gate*1.*residual))
            self.assertTrue(torch.equal(self.node.last_gate,gate))
            self.assertTrue(torch.equal(self.node.last_dz,residual))
            changed = fixed_gate_cached_update(self.model,self.z,self.d+10,age,self.gates)
            self.assertFalse(torch.equal(actual,changed))
            self.assertTrue(torch.equal(self.node.last_gate,gate))

    def test_invalid_conditions_rejected(self):
        for gates, age in ((None,1),({'1':float('nan')},1),(self.gates,4)):
            with self.assertRaises(ValueError):
                fixed_gate_cached_update(self.model,self.z,self.d,age,gates)
        self.node.use_post_layernorm=True
        with self.assertRaises(ValueError):
            fixed_gate_cached_update(self.model,self.z,self.d,1,self.gates)

    def test_plan_reuses_six_rows_without_duplicate_or_training(self):
        plan = experiment_plan()
        self.assertEqual(len(plan),15)
        self.assertEqual(len(set(plan)),15)
        self.assertEqual([r for s,r in plan if s==42],[NEW_ROW])
        for seed in (43,44):
            self.assertEqual([r for s,r in plan if s==seed],list(ROWS))

    def test_metadata_resume_is_immutable(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'plan.json'
            value={'seeds':list(SEEDS),'rows':[list(r) for r in experiment_plan()]}
            campaign.immutable_json(path,value)
            campaign.immutable_json(path,value)
            with self.assertRaises(RuntimeError): campaign.immutable_json(path,{'changed':True})

    def test_smoke_verifier_does_not_recurse_with_stage_runner_hook(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'smoke').mkdir()
            for rank in range(4):
                (root/'smoke'/f'rank{rank}.json').write_text(json.dumps({
                    'normal_action_max_abs_error':0.,'formula_checks':18,
                    'fixed_cached_formula_checks':6,'fixed_cached_reset_pass':True}))
            with replaced_method(campaign.retained,'verify_stage',campaign.verify_stage):
                self.assertTrue(campaign.verify_stage(root,'smoke',{}))
            path=root/'smoke/rank0.json'; value=json.loads(path.read_text())
            value['fixed_cached_formula_checks']=0; path.write_text(json.dumps(value))
            with self.assertRaises(RuntimeError): campaign.verify_stage(root,'smoke',{})

    def test_upstream_keyword_callback_and_seed_every_row(self):
        tree=ast.parse((ROOT/'architectures/seer/upstream/eval_libero.py').read_text())
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)
               and n.func.id=='eval_one_epoch_libero_ddp']
        self.assertEqual({k.arg for k in calls[0].keywords},{'args','model','image_processor','tokenizer'})
        for seed in SEEDS:
            for row in ('smoke',)+ROWS:
                with self.subTest(seed=seed,row=row),tempfile.TemporaryDirectory() as temp:
                    root=Path(temp)
                    config={'initial_state_hashes':{},'libero':'/unused','seed':seed,'max_steps':600,
                            'assets':{'vit':'vit','teacher':'teacher','adapter':'adapter'},'fixed_gates':self.gates}
                    (root/'config.json').write_text(json.dumps(config))
                    (root/'source_hashes.json').write_text('{}')
                    (root/'calibration.json').write_text(json.dumps({'gates':self.gates}))
                    evaluator=ModuleType('eval_libero')
                    runtime=ModuleType('architectures.seer.adapters.latentloop_freshness_confirmation.runtime')
                    runtime.evaluate_loaded=Mock()
                    values={k:object() for k in ('args','model','image_processor','tokenizer')}
                    def upstream_main():
                        self.assertEqual(sys.argv[sys.argv.index('--seed')+1],str(seed))
                        evaluator.eval_one_epoch_libero_ddp(**values)
                    evaluator.main=upstream_main
                    with patch.object(campaign,'sources',return_value={}), \
                         patch.dict(sys.modules,{'eval_libero':evaluator,runtime.__name__:runtime}), \
                         patch.object(sys,'path',list(sys.path)),patch.object(sys,'argv',list(sys.argv)), \
                         patch('torch.distributed.is_initialized',return_value=False):
                        campaign.worker(SimpleNamespace(config=str(root/'config.json'),stage=row))
                    runtime.evaluate_loaded.assert_called_once_with(values['args'],values['model'],
                        values['image_processor'],values['tokenizer'],config,row,root/row)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
