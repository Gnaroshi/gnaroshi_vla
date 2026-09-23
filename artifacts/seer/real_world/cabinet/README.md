# Cabinet real-world checkpoints

Binary checkpoints are intentionally excluded from Git. The frozen Seer teacher
and its LatentLoop adapter are paired by SHA-256 in `checkpoint_manifest.json`.

```text
shared/mae_pretrain_vit_base.pth
baseline/teacher_38.pth
latentloop/teacher_38/teacher_38_adapter_39.pth
```

Teacher 38 is the DROID-finetuned Seer checkpoint trained on the 40-episode
Cabinet dataset. Adapter 39 was trained on s4 with that exact frozen teacher. Do
not mix the adapter with another teacher checkpoint.
