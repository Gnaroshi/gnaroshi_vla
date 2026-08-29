# Basketball real-world checkpoints

Binary checkpoints are intentionally excluded from Git. Baseline teachers,
LatentLoop adapters, and the shared ViT are separated explicitly:

```text
shared/
  mae_pretrain_vit_base.pth
baseline/
  teacher_34.pth
  teacher_35.pth
  teacher_37.pth
latentloop/
  teacher_34/teacher_34_adapter_39.pth
  teacher_35/teacher_35_adapter_39.pth
  teacher_37/teacher_37_adapter_39.pth
```

`checkpoint_manifest.json` records the required SHA-256 and source pairing. The
launcher defaults to teacher 37 with its own adapter epoch 39. Change only the
method-selection lines to compare full Seer baseline against LatentLoop, and
change only `teacher_id` to use teacher 34 or 35. Never mix an adapter with a
different teacher.
