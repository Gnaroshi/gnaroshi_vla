# 3DFlow-Seer real-world deployment snapshot

This directory preserves the deployment files that were actually present on the
robot inference computer before the LatentLoop integration.

- Source host: `jbr@210.107.197.121:9000`
- Source repository: `/home/jbr/3DFlow-Seer`
- Source commit: `a21f0d1ee8cf2c369095c34cbe2d1f99b7b396ba`
- Captured: 2026-08-14

The files are copied byte-for-byte. LatentLoop code must not edit these files;
architecture-specific integration lives in
`architectures/seer/adapters/latentloop_real_deploy/`.

| File | SHA-256 |
|---|---|
| `deploy.py` | `ff6d3f518f0681688105804d635a4598ecb916bc31d50fb5d6367f567feb6d85` |
| `deploy_gui.py` | `ea44180a64b4b37d7af97fce165f1f6adddeb86618921c41f875757bd7b3f97e` |
| `real_controller/controller.py` | `e38058ead24158e69edb2a7c28f993236d593170b0623de3ad094e5667161d6c` |
| `real_controller/robotiq_gripper.py` | `8c479d2cbe8d35ce00d658ec2d14ae71c38883d3c3bd786916ccb7c8f9697165` |
| `scripts/REAL/deploy_gui.sh` | `a0af7019e6581e6866ba4530649e4d7edac1ddaef5a073f3969701424bfa1589` |

The captured `deploy.py` contains the basketball deployment home configuration
used on the inference computer. The source repository itself was not modified by
this copy operation.
