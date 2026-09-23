import importlib.util
import sys
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from architectures.seer.adapters.latentloop_freshness.config import assigned_ids, evaluator_argv, proprio_source_steps
from architectures.seer.adapters.latentloop_freshness.interventions import Inputs, feature, replaced_method, update
from architectures.seer.adapters.latentloop_freshness.campaign import verify_stage

spec = importlib.util.spec_from_file_location('tested_lrnode_modules', ROOT / 'architectures/seer/upstream/models/lrnode_modules.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class FreshnessTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.node = native.ControlledLatentNODE(24, motion_dim=8, hidden_dim=16, action_pred_steps=3)
        self.model = SimpleNamespace(lrnode_dynamics=self.node,
            lrnode_apply_dynamics=lambda z_prev, u_delta, dt, age: self.node(z_prev, u_delta, dt, age))
        self.z, self.d = torch.randn(1, 3, 24), torch.randn(1, 8)

    def test_native_normal_parity_all_ages(self):
        for age in (1, 2, 3):
            a = self.node(self.z, self.d, age=age)
            b = update(self.model, self.z, self.d, age)
            self.assertTrue(torch.equal(a, b))

    def test_crosses_and_zero_and_hold(self):
        observed = self.node(self.z, self.d, age=2)
        gate, residual = self.node.last_gate.clone(), self.node.last_dz.clone()
        zero = self.node(self.z, torch.zeros_like(self.d), age=2)
        gate0, residual0 = self.node.last_gate.clone(), self.node.last_dz.clone()
        self.assertTrue(torch.equal(update(self.model, self.z, self.d, 2, 'hold'), self.z))
        self.assertTrue(torch.equal(update(self.model, self.z, self.d, 2, 'zero'), zero))
        self.assertTrue(torch.equal(update(self.model, self.z, self.d, 2, 'gate_observed_residual_zero'), self.z + gate * 1. * residual0))
        self.assertTrue(torch.equal(update(self.model, self.z, self.d, 2, 'gate_zero_residual_observed'), self.z + gate0 * 1. * residual))
        self.assertTrue(torch.allclose(self.z + (zero-self.z) + (observed-zero), observed))

    def test_feature_inputs_and_legacy_history(self):
        seen = {}
        def encode(**kwargs):
            seen.update(kwargs)
            return torch.ones(1, 8)
        self.model.lrnode_encode_delta = encode
        image0, image1 = torch.zeros(1,3,4,4), torch.ones(1,3,4,4)
        prev = Inputs(image0, image0, torch.arange(56).reshape(1,7,8).float())
        cur = Inputs(image1, image1, prev.state_history + 10)
        feature(self.model, prev, cur, 'normal')
        self.assertTrue(torch.equal(seen['q_cur'], cur.state_history[:,0]))
        feature(self.model, prev, cur, 'current_proprio')
        self.assertTrue(torch.equal(seen['q_cur'], cur.state_history[:,-1]))
        for variant in ('stale_visual', 'repeated_observation'):
            feature(self.model, prev, cur, variant)
            self.assertTrue(torch.equal(seen['cur_image_primary'], image0))
            self.assertTrue(torch.equal(seen['cur_image_wrist'], image0))
        self.assertTrue(torch.equal(seen['q_cur'], prev.q))
        feature(self.model, prev, cur, 'stale_proprio')
        self.assertTrue(torch.equal(seen['cur_image_primary'], image1))
        self.assertTrue(torch.equal(seen['q_cur'], prev.q))

    def test_repeated_observation_is_not_zero_feature(self):
        enc = native.FastVisualDeltaEncoder(motion_dim=8, proprio_dim=8)
        image, q = torch.rand(1,3,16,16), torch.rand(1,8)
        actual = enc([image,image], [image,image], q, q)
        self.assertGreater(actual.norm().item(), 0)

    def test_restore_methods_even_after_exception(self):
        before = self.model.lrnode_apply_dynamics
        with self.assertRaises(RuntimeError):
            with replaced_method(self.model, 'lrnode_apply_dynamics', None):
                raise RuntimeError('intentional')
        self.assertIs(self.model.lrnode_apply_dynamics, before)

    def test_uneven_rank_partition(self):
        for n in (2, 20, 500):
            ids = sum([assigned_ids(n, 4, rank) for rank in range(4)], [])
            self.assertEqual(ids, list(range(n)))

    def test_required_eval_arguments(self):
        args = evaluator_argv({'seed':42, 'assets':dict(vit='v', teacher='t', adapter='a'), 'libero':'l', 'max_steps':600}, '/tmp/output')
        for name in ('--save_checkpoint_path', '--phase', '--vit_checkpoint_path', '--finetune_from_pretrained_ckpt', '--resume_from_checkpoint'):
            self.assertIn(name, args)

    def test_proprio_timestamp_labels_match_interventions(self):
        self.assertEqual(proprio_source_steps(41, 7, 'normal'), (34, 35))
        self.assertEqual(proprio_source_steps(41, 7, 'stale_proprio'), (34, 34))
        self.assertEqual(proprio_source_steps(41, 7, 'repeated_observation'), (34, 34))
        self.assertEqual(proprio_source_steps(41, 7, 'current_proprio'), (40, 41))
        self.assertEqual(proprio_source_steps(1, 7, 'normal'), (0, 0))

    def test_completion_requires_all_rank_markers_and_intact_payloads(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stage = root / 'normal'
            stage.mkdir()
            config = {'tasks':1, 'episodes_per_task':2}
            self.assertFalse(verify_stage(root, 'normal', config, 4))
            for rank in range(4):
                (stage / f'rank{rank}_complete.json').write_text(json.dumps(
                    {'stage':'normal', 'ids':assigned_ids(2,4,rank)}))
            for i in range(2):
                (stage / f'episode_{i:04d}.pt').write_bytes(b'payload')
                (stage / f'episode_{i:04d}.json').write_text(json.dumps({'eval_id':i, 'artifact_bytes':7}))
            self.assertTrue(verify_stage(root, 'normal', config, 4))
            (stage / 'episode_0001.pt').write_bytes(b'bad')
            with self.assertRaises(RuntimeError):
                verify_stage(root, 'normal', config, 4)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
