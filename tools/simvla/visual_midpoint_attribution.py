"""Matched-input visual interventions for completed Condition checkpoints."""
import argparse
import statistics

from tools.simvla import visual_midpoint as study
from tools.simvla.compile_benchmark import write_json


class InterventionFeatures:
    def __init__(self, current, refresh, mode):
        if mode not in ('current', 'refresh'):
            raise ValueError(mode)
        self.current, self.refresh, self.mode = current, refresh, mode

    def __call__(self, age, images):
        if self.mode == 'current':
            return self.current(age, images)
        return self.refresh.to(images.device)


def run(arm, smoke=False):
    import torch
    from tools.simvla.visual_midpoint_followup import contract, OUTPUT
    from tools.simvla.visual_midpoint_features import CachedFeatures, VisionOnly
    from tools.simvla.error_compensation_common import configure, snapshots
    from tools.simvla.condition_output_split_train import unroll, action_prediction
    from tools.simvla.rollout_state_repair import sample_from_sequence
    from methods.latentloop.modules.action_aligned_joint import action_loss
    from methods.latentloop.modules.condition_output_split import geometry
    from architectures.simvla.adapters.latentloop.native_v0_prepare import _official_training_image_inputs
    from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch, load_frozen_simvla, configure_strict_torch_determinism
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import load_native_v0_checkpoint
    from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import install_exact_uint8_delta_path
    from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import _balanced_indices, make_datasets
    from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences, _load_rgb_ref

    c=study.checked(); followup=contract(); cfg=c['model']['config']; configure(cfg)
    device=torch.device('cuda:0'); torch.set_num_threads(1)
    configure_strict_torch_determinism(cfg['seed'])
    total=torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction((total-2*1024**3)/total,device)
    native,native_payload=load_native_v0_checkpoint(cfg['condition_checkpoint'],device=device,require_final_150k=True)
    native.requires_grad_(False).eval(); install_exact_uint8_delta_path(native)
    provenance=snapshots(cfg)
    frozen,processor,action=load_frozen_simvla(checkpoint=provenance['checkpoint_snapshot'],
        norm_stats=cfg['norm_stats'],smolvlm_model=provenance['backbone_snapshot'],device=device)
    frozen.requires_grad_(False).eval()
    _,heldout=make_datasets(cfg,native_payload)
    payload,model_hash=study.model_payload(c,arm)
    model=study.initial_model(c,arm,native).to(device).eval()
    model.load_state_dict(payload['model'],strict=True); model.requires_grad_(False)
    vision=VisionOnly(frozen)
    features=CachedFeatures(study.OUTPUT/'features',c['identity'])
    records=[]
    with torch.inference_mode():
        for index in _balanced_indices(heldout.identities,limit=1 if smoke else 30,seed=c['seed']):
            seq=move_batch(collate_exact_teacher_sequences([heldout[index]]),device)
            features.bind(seq)
            query=seq['query_ids'][0][0]
            raw=_load_rgb_ref(heldout.store.query(query)['metadata']['raw_rgb_ref'])
            batch=_official_training_image_inputs(raw,image_size=processor.image_size,num_views=processor.num_views)
            refresh=vision(batch['image_input'][:,:2].to(device))
            for mode in ('current','refresh'):
                model.feature_provider=InterventionFeatures(features,refresh,mode)
                for age in (4,5,6,7):
                    z,_,_=unroll(model,seq,age,8); sample=sample_from_sequence(seq,age)
                    pred=action_prediction(action,z,sample,steps=1,grad=False)
                    target=action.action_space.normalize_action(sample['target_action'])
                    reference=action_prediction(action,sample['target_condition'],sample,steps=1,grad=False)
                    difference=(pred[:,:5,:7]-target[:,:5,:7]).abs()
                    records.append(dict(window=index,query=seq['query_ids'][0][age],age=age,mode=mode,
                        action_l1=float(action_loss(pred,target)),
                        action_l1_vs_original1=float(action_loss(pred,reference)),
                        original1_l1_vs_original10=float(action_loss(reference,target)),
                        translation_l1=float(difference[...,:3].mean()),
                        rotation_l1=float(difference[...,3:6].mean()),
                        continuous_gripper_l1=float(difference[...,6].mean()),
                        condition=geometry(z,sample['target_condition'],sample['valid'])))
            print(f'ATTRIBUTION arm={arm} window={index} records={len(records)}',flush=True)
    pairs=[]
    for fresh in (r for r in records if r['mode']=='current'):
        stale=next(r for r in records if r['mode']=='refresh' and r['window']==fresh['window'] and r['age']==fresh['age'])
        pairs.append(dict(window=fresh['window'],age=fresh['age'],
            current_minus_refresh_action_l1=fresh['action_l1']-stale['action_l1'],
            current_minus_refresh_raw_mse=fresh['condition']['raw_mse']-stale['condition']['raw_mse'],
            current_minus_refresh_cosine=fresh['condition']['cosine']-stale['condition']['cosine']))
    by_age={str(age):{key:statistics.mean(r[key] for r in pairs if r['age']==age)
        for key in pairs[0] if key not in ('window','age')} for age in (4,5,6,7)}
    write_json(OUTPUT/('attribution_smoke' if smoke else 'attribution')/arm/'summary.json',dict(
        verdict='ATTRIBUTION_SMOKE_PASS' if smoke else 'ATTRIBUTION_COMPLETE',
        identity=followup['identity'],arm=arm,model_sha256=model_hash,windows=len(records)//8,records=records,
        paired_differences=pairs,by_age=by_age,reference='Original cached NFE10 actions and Condition; NFE1 original Condition reference also measured. All action metrics use normalized continuous first5 actions.',
        scope='Same heldout observations, proprioception, checkpoint, initial noise and recursive history protocol. Only vision features use current versus refresh images. No training or deployment latency claim.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True,choices=study.ARMS)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    run(args.arm,args.smoke)
