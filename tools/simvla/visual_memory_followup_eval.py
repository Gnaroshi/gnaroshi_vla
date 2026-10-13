"""Independent-environment paired Long500 for missing comparators and controls."""
import hashlib
import logging
import os
from pathlib import Path
import sys
import time
from types import MethodType

from tools.simvla import visual_memory_followup as study
from tools.simvla import failure_interventions as prior
from tools.simvla.compile_benchmark import read_json, write_json, sha
from tools.simvla.visual_midpoint_eval import attach_visual
from tools.simvla.age_specialization_eval import check as check_ours, compile_check as compile_ours


def attach_reference(policy, model, interval, compiler):
    policy = attach_visual(policy,model,interval,compiler)
    if model.reference_age == 4:
        return policy
    capture = dict(enabled=False,value=None)
    def hook(module,args,output):
        if capture['enabled']:
            capture['value'] = output.reshape(1,-1,model.early.condition_dim).detach()
    policy._reference_vision_hook = policy.model.vlm.model.connector.register_forward_hook(hook)
    full, reset = policy._full_refresh, policy.reset
    def full_capture(self,batch,*,policy_query_index):
        capture.update(enabled=True,value=None)
        try:
            result = full(batch,policy_query_index=policy_query_index)
        finally:
            capture['enabled'] = False
        if capture['value'] is None:
            raise RuntimeError('Missing original connector capture')
        return result
    def provide(age,images):
        if age != 0 or capture['value'] is None:
            raise RuntimeError('Unexpected additional vision request')
        return capture['value']
    def reset_capture(self):
        reset(); capture.update(enabled=False,value=None)
    model.feature_provider = provide
    policy._full_refresh = MethodType(full_capture,policy)
    policy.reset = MethodType(reset_capture,policy)
    policy.reset()
    return policy


def check(policy,compiler,spec,row):
    if spec['kind']=='bridge':
        from tools.simvla import condition_interval_compare as bridge
        bridge.check_policy(policy,row); bridge.check_compiler(compiler,row)
        return dict(policy.metrics.counters)
    counts = check_ours(policy,'protected' if spec['kind']=='ours' else 'parent')
    if spec['kind']=='ours':
        compile_ours(compiler,'protected',8)
        expected = sum(policy._age_model.uses_vision(t%8) for t in range(counts['queries']))
        if policy._extra_vision_calls != expected:
            raise RuntimeError('Unexpected original vision call count')
        if compiler:
            names = ['visual_reference'] + (['vision_only'] if spec['reference_age']==4 else [])
            if any(not compiler.records.get(n,{}).get('graphs',0) for n in names):
                raise RuntimeError('Visual path bypassed compile')
        counts['extra_vision_calls'] = expected
    elif compiler and any(not compiler.records.get(n,{}).get('graphs',0) for n in ('vlm','action_transformer')):
        raise RuntimeError('Baseline compile bypass')
    return counts


def evaluate(row,smoke=False):
    from tools.simvla.refresh_allocation import setup
    c=study.checked();cfg=c['model']['config'];setup(cfg);spec=study.row_spec(row)
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
    out=study.OUTPUT/('eval_smoke' if smoke else 'rows')/row
    capture=WarningCapture()
    for name in ('torch._dynamo','torch._inductor'):logging.getLogger(name).addHandler(capture)
    suite=benchmark.get_benchmark_dict()['libero_10']()
    with torch.inference_mode():
        parity=None
        if spec['kind']=='bridge':
            from tools.simvla import condition_interval_compare as bridge
            from tools.simvla.compiled_campaign import move
            from tools.simvla.compile_runtime import Compiler
            bridge.ROWS[row]=('bridge',spec['interval'])
            compiler=Compiler(True)
            samples=move(torch.load(Path(cfg['recorded_input'])/'recorded_inputs.pt',map_location='cpu',weights_only=False)[:1])
            replay=bridge.replay_factory(cfg,row,compiler,samples)
            policy=bridge.policy_factory(replay,cfg,row,c['manifest'])
            model_hash=sha(cfg['bridge_checkpoint'])
        else:
            policy,compiler=prior.build_policy('recurrent_k4',cfg)
            if spec['kind']=='ours':payload,model_hash=study.model_payload(c,row)
            else:
                policy.k_c=policy.refresh_every=1;policy.reset()
                model_hash=prior.digest(dict(checkpoint_revision=cfg['checkpoint_revision'],norm_stats=sha(cfg['norm_stats'])))
        env,prompt=get_libero_env(suite.get_task(9),256,7)
        try:
            inputs=build_env_obs(env.reset())
            if spec['kind']=='ours':
                reference=[policy.act(*inputs,prompt).action.copy() for _ in range(80)]
                native,_=load_native_v0_checkpoint(cfg['condition_checkpoint'],device='cuda',require_final_150k=True)
                install_exact_uint8_delta_path(native)
                model=study.initial_model(c,row,native).to('cuda').eval()
                model.load_state_dict(payload['model'],strict=True);model.requires_grad_(False)
                policy=attach_reference(policy,model,4,compiler)
                actual=[policy.act(*inputs,prompt).action.copy() for _ in range(80)]
                parity=float(np.abs(np.stack(reference)-np.stack(actual)).max())
                if parity!=0:raise RuntimeError('Protected K4 action parity changed')
                policy.k_c=policy.refresh_every=8;policy.reset()
            for _ in range(160):policy.act(*inputs,prompt)
            counts=check(policy,compiler,spec,row)
            write_json(out/'runtime.json',dict(model_sha256=model_hash,model_spec=spec,
                protected_k4_max_action_diff=parity,compiler=compiler.records if compiler else None,
                gpu=torch.cuda.get_device_name(0),torch=torch.__version__,command=sys.argv,
                base_checkpoint_revision=cfg['checkpoint_revision'],norm_sha256=sha(cfg['norm_stats']),
                environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','MUJOCO_GL','PYOPENGL_PLATFORM','LIBERO_CONFIG_PATH')}))
        finally:env.close()
        if smoke:
            write_json(out/'summary.json',dict(verdict='EVAL_SMOKE_PASS',identity=c['identity'],
                row=row,model_sha256=model_hash,protected_k4_max_action_diff=parity,counts=counts))
            return
        started=time.monotonic();records=[]
        for episode in c['manifest']['episodes']:
            task,trial=episode['task_id'],episode['trial_id'];path=out/'episodes'/f'task{task}_trial{trial:02d}.json'
            if path.exists():
                record=read_json(path)
                if record['identity']!=c['identity'] or record['model_sha256']!=model_hash:raise RuntimeError('Mixed resumed episode')
                records.append(record);continue
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
                        write_json(out/'progress.json',dict(completed=len(records),successes=sum(r['success'] for r in records),
                            total=500,task=task,trial=trial,actions=i+1,phase='rollout'));last=time.monotonic()
                    if success:break
                counts=check(policy,compiler,spec,row)
                fallback=any(any(s in msg for s in ('hit config.recompile_limit','hit config.cache_size_limit',"WON'T CONVERT")) for msg in capture.messages)
                record=dict(identity=c['identity'],row=row,model_sha256=model_hash,task_id=task,trial_id=trial,
                    success=bool(success),episode_length=len(actions),policy_ms_total=sum(times),counts=counts,
                    timing_valid=not compiler or (compiler.graph_count()==graphs and not fallback),
                    wall_seconds=time.monotonic()-tick,action_sha256=hashlib.sha256(np.stack(actions).tobytes()).hexdigest())
                path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path.with_suffix('.npz'),actions=np.stack(actions));write_json(path,record)
                if frames:
                    try:save_episode_video(frames,out/'videos'/f'task{task}_trial0.mp4',fps=10)
                    except Exception as exc:write_json(out/f'video_error_task{task}.json',dict(error=str(exc)))
                records.append(record)
                write_json(out/'progress.json',dict(completed=len(records),total=500,successes=sum(r['success'] for r in records),
                    recent_wall_seconds=record['wall_seconds'],updated_unix=time.time(),phase='rollout'))
                print(f"{row} {len(records)}/500 success={sum(r['success'] for r in records)}/{len(records)} wall={record['wall_seconds']:.1f}s",flush=True)
            finally:env.close()
        valid=[r for r in records if r['timing_valid']]
        write_json(out/'summary.json',dict(verdict='EVAL_COMPLETE',identity=c['identity'],row=row,episodes=len(records),
            successes=sum(r['success'] for r in records),success_rate=sum(r['success'] for r in records)/len(records),
            policy_ms_per_action=sum(r['policy_ms_total'] for r in valid)/sum(r['episode_length'] for r in valid) if valid else None,
            timing_valid_episodes=len(valid),per_task={str(t):sum(r['success'] for r in records if r['task_id']==t) for t in range(10)},
            elapsed_seconds=time.monotonic()-started,hardware='rb2 RTX5090 compiled' if prior.IS_RB2 else 'sd1 RTX3090 eager',
            model_spec=spec,model_sha256=model_hash,
            timing_scope='Synchronized policy.act includes preprocessing, vision, attention, action and queue; excludes env/render/video and compile-invalid episode times.'))
