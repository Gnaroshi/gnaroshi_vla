# Basketball real-world checkpoints

Binary checkpoints are intentionally excluded from Git. The deployment launcher
expects these files in this directory:

```text
mae_pretrain_vit_base.pth
teacher_34.pth
teacher_34_adapter_39.pth
teacher_35.pth
teacher_35_adapter_39.pth
teacher_37.pth
teacher_37_adapter_39.pth
```

`checkpoint_manifest.json` records the required SHA-256 and source pairing. The
launcher defaults to teacher 37 with its own adapter epoch 39. Change only the
`teacher_id` assignment near the top of `deploy_ll_gui.sh` to use teacher 34 or
35; never mix an adapter with a different teacher.
