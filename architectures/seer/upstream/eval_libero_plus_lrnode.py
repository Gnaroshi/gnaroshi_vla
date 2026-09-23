"""LIBERO-Plus entry point backed by the canonical Seer/LR-NODE evaluator."""

import os

import eval_libero as canonical_eval
from utils.eval_utils_libero_plus_lrnode import eval_one_epoch_libero_plus_ddp


if __name__ == "__main__":
    os.environ["NCCL_BLOCKING_WAIT"] = "0"
    # Reuse canonical model construction and checkpoint-contract validation.
    canonical_eval.eval_one_epoch_libero_ddp = eval_one_epoch_libero_plus_ddp
    canonical_eval.main()
