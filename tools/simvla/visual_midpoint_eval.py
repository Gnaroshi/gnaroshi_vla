"""Evaluate token-reference models with the existing paired Long500 contract."""
import hashlib
import logging
import os
import sys
import time

from tools.simvla import visual_midpoint as study
from tools.simvla import failure_interventions as prior
from tools.simvla.age_specialization_eval import attach, check as parent_check, compile_check as parent_compile_check
from tools.simvla.compile_benchmark import read_json, write_json



def attach_visual(policy,model,interval,compiler):
    from types import MethodType
    from tools.simvla.visual_midpoint_features import VisionOnly
    policy=attach(policy,model,interval,compiler)
    vision=VisionOnly(policy.model)
    if compiler:
        vision.forward=compiler.wrap('vision_only',vision.forward)
        model.visual.forward=compiler.wrap('visual_reference',model.visual.forward)
    reset,update=policy.reset,policy._v0_update
    def reset_visual(self):
        reset();self._extra_vision_calls=0;self._vision_batch=None
    def provide(age,images):
        batch=policy._vision_batch
        if batch is None:raise RuntimeError('Missing current policy input')
        policy._extra_vision_calls+=1
        return vision(batch['image_input'][:,:2])
    def update_visual(self,batch,*,age,policy_query_index):
        self._vision_batch=batch
        return update(batch,age=age,policy_query_index=policy_query_index)
    model.feature_provider=provide
    policy.reset=MethodType(reset_visual,policy)
    policy._v0_update=MethodType(update_visual,policy)
    policy.reset()
    return policy


def check(policy,arm):
    if arm=='bridge_f4':
        from tools.simvla.condition_interval_compare import check_policy
        check_policy(policy,'bridge_k4')
        return dict(policy.metrics.counters)
    counts=parent_check(policy,'protected')
    expected=sum(policy._age_model.uses_vision(t%policy.k_c) for t in range(counts['queries']))
    if policy._extra_vision_calls!=expected:raise RuntimeError('Vision invocation schedule mismatch')
    return dict(**counts,extra_vision_calls=expected)


def compile_check(compiler,arm):
    if arm=='bridge_f4':
        from tools.simvla.condition_interval_compare import check_compiler
        check_compiler(compiler,'bridge_k4');return
    parent_compile_check(compiler,'protected',8)
    if compiler:
        for name in ('vision_only','visual_reference'):
            if not compiler.records.get(name,{}).get('graphs',0):raise RuntimeError('Compile bypass: '+name)

def evaluate(arm, smoke=False):
    from tools.simvla.refresh_allocation import setup
    c=study.checked();cfg=c['model']['config'];setup(cfg)
    import numpy as np
    import torch
    from libero.libero import benchmark
    from architectures.simvla.adapters.latentloop.native_v0_runtime import configure_strict_torch_determinism
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import load_native_v0_checkpoint
    from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import install_exact_uint8_delta_path
    from architectures.simvla.wrappers.dcld_eval.rollout_runner import build_env_obs,get_libero_env,video_frame_from_obs,save_episode_video
    from tools.simvla.compile_audit import WarningCapture
    torch.set_num_threads(1);configure_strict_torch_determinism(20260815)
    total=torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((total-2*1024**3)/total)
    out=study.OUTPUT/('eval_smoke' if smoke else 'rows')/arm
    capture=WarningCapture()
    for name in ('torch._dynamo','torch._inductor'):logging.getLogger(name).addHandler(capture)
    suite=benchmark.get_benchmark_dict()['libero_10']()
    with torch.inference_mode():
        parity=None
        if arm=='bridge_f4':
            from pathlib import Path
            from tools.simvla.condition_interval_compare import replay_factory,policy_factory
            from tools.simvla.condition_output_split_rb2 import move
            from tools.simvla.compile_runtime import Compiler
            from tools.simvla.compile_benchmark import sha
            compiler=Compiler(True)
            samples=move(torch.load(Path(cfg['recorded_input'])/'recorded_inputs.pt',map_location='cpu',weights_only=False)[:1])
            replay=replay_factory(cfg,'bridge_k4',compiler,samples)
            policy=policy_factory(replay,cfg,'bridge_k4',c['manifest'])
            model_hash=sha(cfg['bridge_checkpoint'])
        else:
            policy,compiler=prior.build_policy('recurrent_k4',cfg)
            payload,model_hash=study.model_payload(c,arm)
        env,prompt=get_libero_env(suite.get_task(9),256,7)
        try:
            inputs=build_env_obs(env.reset())
            if arm!='bridge_f4':
                reference=[policy.act(*inputs,prompt).action.copy() for _ in range(80)]
                native,_=load_native_v0_checkpoint(cfg['condition_checkpoint'],device='cuda',require_final_150k=True)
                install_exact_uint8_delta_path(native)
                model=study.initial_model(c,arm,native).to('cuda').eval()
                model.load_state_dict(payload['model'],strict=True);model.requires_grad_(False)
                policy=attach_visual(policy,model,4,compiler)
                actual=[policy.act(*inputs,prompt).action.copy() for _ in range(80)]
                parity=float(np.abs(np.stack(reference)-np.stack(actual)).max())
                if parity!=0:raise RuntimeError('Frozen K4 action parity changed')
                policy.k_c=policy.refresh_every=8;policy.reset()
            for _ in range(160):policy.act(*inputs,prompt)
            check(policy,arm);compile_check(compiler,arm)
            write_json(out/'runtime.json',dict(parent_checkpoint_sha256=c['model']['sha256'],model_sha256=model_hash,
                protected_k4_max_action_diff=parity,compiler=compiler.records if compiler else None,
                gpu=torch.cuda.get_device_name(0),torch=torch.__version__,command=sys.argv,
                environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','MUJOCO_GL','PYOPENGL_PLATFORM','LIBERO_CONFIG_PATH')}))
        finally:env.close()
        if smoke:
            write_json(out/'summary.json',dict(verdict='EVAL_SMOKE_PASS',identity=c['identity'],
                arm=arm,protected_k4_max_action_diff=parity,counts=check(policy,arm)))
            return
        started=time.monotonic();rows=[]
        for spec in c['manifest']['episodes']:
            task,trial=spec['task_id'],spec['trial_id'];path=out/'episodes'/f'task{task}_trial{trial:02d}.json'
            if path.exists():
                record=read_json(path)
                if record['identity']!=c['identity'] or record['model_sha256']!=model_hash:raise RuntimeError('Mixed resumed episode')
                rows.append(record);continue
            configure_strict_torch_determinism(20260815)
            env,prompt=get_libero_env(suite.get_task(task),256,7)
            try:
                env.seed(7);env.reset();obs=env.set_init_state(suite.get_task_init_states(task)[trial])
                for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
                policy.task_id,policy.trial_id=task,trial;policy.reset()
                graphs=compiler.graph_count() if compiler else 0
                times,actions,frames=[],[],[];tick=last=time.monotonic();success=False
                for i in range(900):
                    inputs=build_env_obs(obs)
                    if trial==0 and i%2==0:frames.append(video_frame_from_obs(obs))
                    torch.cuda.synchronize();began=time.perf_counter();step=policy.act(*inputs,prompt)
                    torch.cuda.synchronize();times.append(1000*(time.perf_counter()-began))
                    if not np.isfinite(step.action).all():raise RuntimeError('Nonfinite action')
                    actions.append(step.action.copy());obs,_,success,_=env.step(step.action.tolist())
                    if time.monotonic()-last>30:
                        write_json(out/'progress.json',dict(completed=len(rows),successes=sum(r['success'] for r in rows),
                            total=500,task=task,trial=trial,actions=i+1,phase='rollout'));last=time.monotonic()
                    if success:break
                counts=check(policy,arm);compile_check(compiler,arm)
                fallback=any(any(s in msg for s in ('hit config.recompile_limit','hit config.cache_size_limit',"WON'T CONVERT")) for msg in capture.messages)
                record=dict(identity=c['identity'],arm=arm,model_sha256=model_hash,task_id=task,trial_id=trial,
                    success=bool(success),episode_length=len(actions),policy_ms_total=sum(times),counts=counts,
                    timing_valid=not compiler or (compiler.graph_count()==graphs and not fallback),
                    wall_seconds=time.monotonic()-tick,action_sha256=hashlib.sha256(np.stack(actions).tobytes()).hexdigest())
                path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path.with_suffix('.npz'),actions=np.stack(actions));write_json(path,record)
                if frames:
                    try:save_episode_video(frames,out/'videos'/f'task{task}_trial0.mp4',fps=10)
                    except Exception as exc:write_json(out/f'video_error_task{task}.json',dict(error=str(exc)))
                rows.append(record)
                write_json(out/'progress.json',dict(completed=len(rows),total=500,successes=sum(r['success'] for r in rows),
                    recent_wall_seconds=record['wall_seconds'],updated_unix=time.time(),phase='rollout'))
                print(f"{arm} {len(rows)}/500 success={sum(r['success'] for r in rows)}/{len(rows)} wall={record['wall_seconds']:.1f}s",flush=True)
            finally:env.close()
        valid=[r for r in rows if r['timing_valid']]
        summary=dict(verdict='EVAL_COMPLETE',identity=c['identity'],arm=arm,episodes=len(rows),
            successes=sum(r['success'] for r in rows),success_rate=sum(r['success'] for r in rows)/len(rows),
            policy_ms_per_action=sum(r['policy_ms_total'] for r in valid)/sum(r['episode_length'] for r in valid) if valid else None,
            timing_valid_episodes=len(valid),per_task={str(t):sum(r['success'] for r in rows if r['task_id']==t) for t in range(10)},
            elapsed_seconds=time.monotonic()-started,hardware='rb2 RTX5090 compiled' if prior.IS_RB2 else 'sd1 RTX3090 eager',
            timing_scope='Synchronized policy.act includes current vision encoder, visual attention and action queue, excludes environment/render/video. Compile-invalid episode times excluded.')
        write_json(out/'summary.json',summary)
        comparisons={}
        for name in ('recurrent_parent_k4','recurrent_protected_k8'):
            p=study.previous.OUTPUT/'rows'/name/'summary.json'
            if p.exists():comparisons[name]=read_json(p)
        write_json(out/'comparison.json',dict(new=summary,completed_references=comparisons,
            comparison='Same server and manifest. Parent K4 is a target, protected K8 is the no-reference control. Tokenwise arm isolates access to other tokens.'))

