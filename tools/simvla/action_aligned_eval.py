"""Use one jointly adapted Condition/Generation pair at every policy query."""
import argparse
from pathlib import Path

import torch

from methods.latentloop.modules.action_aligned_joint import ARMS
from tools.simvla.action_aligned_campaign import CONFIG
from tools.simvla.error_compensation_common import identity, read_json
from tools.simvla.error_compensation_eval import make_policy as parent_policy, run


def make_policy(c, row, *, smoke=False, k_c=4):
    policy = parent_policy(c, "parent", k_c=k_c)
    path = Path(c["output"]) / ("smoke" if smoke else "train") / row / "latest.pt"
    payload = torch.load(path, map_location="cuda", weights_only=False)
    expected_steps = c["smoke_steps"] if smoke else c["steps"]
    if (payload["format"] != "simvla_action_aligned_joint_v1" or payload["arm"] != row
            or payload["identity"] != identity(c) or payload["step"] != expected_steps):
        raise RuntimeError("Incompatible or incomplete joint checkpoint")
    policy.native_v0.load_state_dict(payload["condition_state"], strict=True)
    policy._experiment_loops[0].updater.load_state_dict(payload["generation_state"], strict=True)
    policy.row_name = row
    return policy


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(CONFIG))
    p.add_argument("--row", choices=ARMS, required=True)
    p.add_argument("--k-c", type=int, choices=(2, 4), required=True)
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    run(read_json(a.config), a.row, smoke=a.smoke, k_c=a.k_c,
        policy_factory=make_policy, counter_row="parent")
