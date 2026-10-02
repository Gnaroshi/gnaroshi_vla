"""Offline-only oracle progress: teacher values can never enter online inference."""
import argparse
from pathlib import Path
import torch
from tqdm import tqdm

from methods.latentloop.modules.observed_progress import projection_coefficient
from methods.latentloop.modules.trend_condition import TrendCondition, scaled_mse
from methods.latentloop.modules.action_aligned_joint import differentiable_rollout
from methods.latentloop.modules.simvla_generation_loop import SimVLAGenerationLoop
from architectures.simvla.adapters.latentloop.efficient_multirate.shared_refinement_train import load_runtime
from architectures.simvla.adapters.latentloop.efficient_multirate.condition_mechanism import _balanced_indices, action_metrics
from architectures.simvla.adapters.latentloop.efficient_multirate.exact_teacher_cache import collate_exact_teacher_sequences
from architectures.simvla.adapters.latentloop.efficient_multirate.generation_checkpoint import load_generation_checkpoint
from architectures.simvla.adapters.latentloop.native_v0_runtime import move_batch
from tools.simvla.compile_runtime import ActionStep
from tools.simvla.error_compensation_common import configure, snapshots, read_json, write_json, sha, identity


@torch.no_grad()
def run(c, output):
    configure(c)
    torch.set_num_threads(1)
    spec=c['frozen_trend_checkpoint']
    if sha(spec['path'])!=spec['sha256']: raise RuntimeError('Trend checkpoint changed')
    device,parent,frozen,action,_,heldout=load_runtime(c,snapshots(c))
    model=TrendCondition(parent,'trend_only',max_age=c['training_k_c']-1).to(device).eval()
    model.load_state_dict(torch.load(spec['path'],map_location=device,weights_only=False)['model'],strict=True)
    generation,_=load_generation_checkpoint(c['generation_checkpoint'],device=device)
    loop=SimVLAGenerationLoop(generation,frozen.transformer.action_decoder).eval()
    step=ActionStep(frozen.transformer).eval()
    rows=[]
    for i in tqdm(_balanced_indices(heldout.identities,limit=c['heldout_windows'],seed=c['seed']),desc='Offline oracle progress'):
        s=move_batch(collate_exact_teacher_sequences([heldout[i]]),device)
        ctx=model.prepare(s['anchor_condition'],s['image_sequence'][:,0],s['proprio_sequence'][:,0],s['valid_mask'].bool(),s['group_ids'])
        for age in range(1,model.max_age+1):
            target=s['teacher_conditions'][:,age-1]
            alpha=projection_coefficient(target-ctx.anchor,ctx.trend,ctx.anchor,ctx.valid)
            for label,condition in [('fixed_age',ctx.anchor+age*ctx.trend),
                    ('oracle_progress',ctx.anchor+alpha*ctx.trend),('original_condition',target)]:
                pred=differentiable_rollout(loop,step,condition,action.normalize_proprio(s['proprio_sequence'][:,age]),s['explicit_noises'][:,age-1])
                rows.append(dict(index=i,task_id=int(s['task_id'][0]),age=age,variant=label,alpha=float(alpha.flatten()[0]),
                    condition_mse=float(scaled_mse(condition,target,ctx.anchor,ctx.valid)),
                    **action_metrics(action.action_space.postprocess(pred),s['teacher_actions'][:,age-1])))
    metrics=[k for k in rows[0] if k not in ('index','task_id','age','variant')]
    def mean(rs): return {k:sum(r[k] for r in rs)/len(rs) for k in metrics}
    write_json(output/'queries.json',rows)
    write_json(output/'summary.json',dict(verdict='OFFLINE_ORACLE_COMPLETE',queries=len(rows)//3,
        identity=identity(c),
        model_sha256=spec['sha256'],config=c,heldout=heldout.contract(),online_eligible=False,
        interpretation='Oracle coefficient uses teacher condition for diagnostic only. Original-condition arm keeps Generation N_G3; action targets are original NFE10.',
        variants={v:mean([r for r in rows if r['variant']==v]) for v in ('fixed_age','oracle_progress','original_condition')},
        by_age={str(a):{v:mean([r for r in rows if r['variant']==v and r['age']==a]) for v in ('fixed_age','oracle_progress','original_condition')} for a in range(1,model.max_age+1)}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    run(read_json(a.config),a.output)
