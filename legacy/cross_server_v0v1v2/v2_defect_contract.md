# LatentLoop V2 Defect Contract

V2 reuses the selected V1 transition checkpoint and does not train a new transition.

At each lightweight query:

`eta = RMS((z_seq - z_dir) / sigma_train)`

`sigma_train` is a per-coordinate RMS scale fit only on transition-training episodes. No learned projection is permitted.

Defect fitting, defect validation, scheduler calibration, and final 200-episode evaluation are episode-disjoint. The defect signal passes only when action error is monotonic over defect bins, Spearman correlation is positive and at least 0.10, high-error-tail AUROC is at least 0.70, and defect AUROC strictly exceeds age, observation-change norm, and previous-action magnitude.

Scheduler semantics are fixed:

- Level 0 keeps `z_seq`.
- Level 1 uses `z_dir`, writes it as recurrent current state, and resets sequential age without claiming a Full Seer refresh.
- Level 2 executes Full Seer, replaces full anchor/current state, and resets all ages/history.

Thresholds and maximum ages are selected on scheduler-calibration episodes at measured `3.8 <= K_hat <= 4.2`. Final 200 episodes cannot influence them.
