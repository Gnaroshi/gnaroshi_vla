"""The Seer-only freshness experiment contract (no training)."""
from pathlib import Path

SCHEMA = 1
SHARED = Path('/home/mingyujung/shared/nvme1/mingyujung/robotics')
PAPER = SHARED / 'gnaroshi_vla/artifacts/checkpoints/seer/paper/libero_long'
ASSETS = {
    'teacher': (str(PAPER / 'teacher_public33.pth'), 'a74f200bb91618a27cbb8e25bc6e1008647056ebe4155348095d63b658936646'),
    'adapter': (str(PAPER / 'latentloop_adapter39.pth'), '3f70179ab9b1bae64fc772d71c57a93592b9f82e53b5fcaf1a6beb319c280462'),
    'vit': (str(SHARED / 'seer/vit_mae/mae_pretrain_vit_base.pth'), 'aec5f0b68e5f3193a00b07bc65a37440db549c15b36b8bea242606cc40c4bc5d'),
    'clip': (str(Path.home() / '.cache/clip/ViT-B-32.pt'), '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'),
}
VARIANTS = ('normal', 'hold', 'zero', 'gate_observed_residual_zero',
            'gate_zero_residual_observed', 'stale_visual', 'stale_proprio',
            'repeated_observation', 'current_proprio')
ROWS = ('full',) + VARIANTS
EXTERNAL_ROWS = ('full', 'normal', 'zero', 'stale_visual', 'hold')
DEFAULT_RESULT = str(SHARED / 'gnaroshi_vla/results/seer/latentloop/freshness/public33_seed42_r1')


def evaluator_argv(config, output):
    """Match the retained public33 K4 paper snapshot, including legacy state indexing."""
    return [
        '--phase', 'evaluate', '--finetune_type', 'libero_10',
        '--save_checkpoint_path', str(Path(output) / 'eval_metadata'),
        '--run_name', 'seer_freshness', '--seed', str(config['seed']),
        '--precision', 'fp32', '--bf16_module', 'vision_encoder',
        '--vit_checkpoint_path', config['assets']['vit'],
        '--libero_path', config['libero'], '--libero_img_size', '128',
        '--libero_eval_max_steps', str(config['max_steps']),
        '--sequence_length', '7', '--future_steps', '3', '--action_pred_steps', '3',
        '--num_resampler_query', '6', '--num_obs_token_per_image', '9',
        '--transformer_layers', '24', '--hidden_dim', '384', '--transformer_heads', '12',
        '--obs_pred', '--gripper_width', '--eval_libero_ensembling', '--ensembling_temp', '0.01',
        '--traj_cons', '--rgb_pad', '10', '--gripper_pad', '4',
        '--calvin_dataset', '', '--workers', '0', '--batch_size', '64',
        '--multi_step_action', '1', '--use_lrnode_latent_update', '1',
        '--lrnode_train_protocol', 'adapter', '--lrnode_eval_skip_full_forward', '1',
        '--lrnode_query_interval', '4', '--lrnode_hidden_dim', '256',
        '--lrnode_motion_dim', '128', '--lrnode_fast_encoder_type', 'diffcnn',
        '--lrnode_use_post_layernorm', '0', '--lrnode_gate_init_bias', '-4',
        '--lrnode_eval_step_log', '0', '--lrnode_eval_profile_full_action_head', '0',
        '--finetune_from_pretrained_ckpt', config['assets']['teacher'],
        '--resume_from_checkpoint', config['assets']['adapter'],
    ]


def assigned_ids(total, world, rank):
    count = (total + world - 1) // world
    return list(range(rank * count, min((rank + 1) * count, total)))


def proprio_source_steps(timestep, history, variant):
    previous, current = max(0, timestep - history), max(0, timestep - history + 1)
    if variant == 'current_proprio':
        return timestep - 1, timestep
    if variant in ('stale_proprio', 'repeated_observation'):
        current = previous
    return previous, current
