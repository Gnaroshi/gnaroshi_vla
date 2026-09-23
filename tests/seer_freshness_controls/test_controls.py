import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from architectures.seer.adapters.latentloop_freshness_controls.control import (
    ROWS, RefreshFeature, calibration_means, controlled_update)

spec = importlib.util.spec_from_file_location('tested_native_node', ROOT / 'architectures/seer/upstream/models/lrnode_modules.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


class ControlsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.node = native.ControlledLatentNODE(24, motion_dim=8, hidden_dim=16, action_pred_steps=3)
        self.model = SimpleNamespace(lrnode_dynamics=self.node,
            lrnode_apply_dynamics=lambda z_prev,u_delta,dt,age:self.node(z_prev,u_delta,dt,age))
        self.z, self.d = torch.randn(1,3,24), torch.randn(1,8)

    def test_formulas_all_ages(self):
        for age in (1,2,3):
            normal = self.node(self.z,self.d,age=age)
            gate = self.node.last_gate.clone()
            self.node(self.z,torch.zeros_like(self.d),age=age)
            residual0 = self.node.last_dz.clone()
            self.assertTrue(torch.equal(controlled_update(self.model,self.z,self.d,age,'cached_feature'),normal))
            actual = controlled_update(self.model,self.z,self.d,age,'gate_cached_residual_zero')
            self.assertTrue(torch.equal(actual,self.z+gate*1.*residual0))
            gates = {'1':.1,'2':.2,'3':.3}
            actual = controlled_update(self.model,self.z,self.d,age,'gate_fixed_residual_zero',gates)
            self.assertTrue(torch.equal(actual,self.z+torch.full_like(self.z[...,:1],gates[str(age)])*1.*residual0))

    def test_fixed_gate_does_not_depend_on_feature(self):
        gates = {'1':.1,'2':.2,'3':.3}
        for age in (1,2,3):
            a = controlled_update(self.model,self.z,self.d,age,'gate_fixed_residual_zero',gates)
            b = controlled_update(self.model,self.z,self.d+100,age,'gate_fixed_residual_zero',gates)
            self.assertTrue(torch.equal(a,b))

    def test_refresh_cache_is_immutable_and_segment_bound(self):
        cache = RefreshFeature()
        with self.assertRaises(RuntimeError): cache.read(1)
        cache.commit(self.d,0)
        saved = cache.read(1).clone()
        self.d.add_(100)
        for t in (1,2,3): self.assertTrue(torch.equal(cache.read(t),saved))
        with self.assertRaises(RuntimeError): cache.read(4)
        cache.commit(self.d,4)
        self.assertTrue(torch.equal(cache.read(5),self.d))

    def test_calibration_equal_episode_weight_no_success_selection(self):
        rows = [{'gate_mean_by_age':{'1':.1,'2':.2,'3':.3},'success':False},
                {'gate_mean_by_age':{'1':.3,'2':.4,'3':.5},'success':True}]
        self.assertAlmostEqual(calibration_means(rows)['1'],.2)
        rows[0]['success'] = True
        self.assertAlmostEqual(calibration_means(rows)['1'],.2)
        with self.assertRaises(ValueError): calibration_means([])

    def test_reject_missing_calibration_and_invalid_age(self):
        with self.assertRaises(ValueError): controlled_update(self.model,self.z,self.d,1,'gate_fixed_residual_zero')
        with self.assertRaises(ValueError): controlled_update(self.model,self.z,self.d,4,'cached_feature')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
