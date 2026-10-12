"""Paired Long500 evaluation for preserved short-interval Condition updates."""
import hashlib
import logging
from types import MethodType
import time

from tools.simvla import age_specialization as study
from tools.simvla import failure_interventions as prior
from tools.simvla.compile_benchmark import read_json,write_json


def attach(policy, model, interval, compiler):
    from tools.simvla.observation_correction_eval import tensor_age_call
    from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import use_normalized_float_delta_inputs
    policy._age_calls={'early':0,'late':0}
    for label,expert in model.experts():
        use_normalized_float_delta_inputs(expert)
        if compiler:
            for attr,name in [('delta_encoder','observation_encoder'),('condition_updater','condition_updater')]:
                module=getattr(expert,attr)
                compiled=compiler.wrap(label+'_'+name,module.forward)
                if attr=='condition_updater':
                    module.forward=lambda *a,_f=compiled,**kw:tensor_age_call(_f,*a,**kw)
                else:module.forward=compiled
    original_full,original_reset=policy._full_refresh,policy.reset
    policy.k_c=policy.refresh_every=interval
    policy._age_model=model
    def reset(self):
        original_reset();self._age_context=None;self._age_calls={'early':0,'late':0}
    def full(self,batch,*,policy_query_index):
        z,chunk,seed=original_full(batch,policy_query_index=policy_query_index)
        self._age_context=model.prepare(z,batch['raw_rgb'],batch['proprio'],
            self.condition_layout.valid_mask,self.condition_layout.group_ids,self.k_c)
        return z,chunk,seed
    def update(self,batch,*,age,policy_query_index):
        if self._age_context is None:raise RuntimeError('Missing original refresh')
        z,_=model.predict(self._age_context,age,batch['raw_rgb'],batch['proprio'])
        label='late' if model.protected and age>=4 else 'early'
        self._age_calls[label]+=1
        self.metrics.counters['num_condition_updater_calls']+=1
        self.metrics.counters['num_observation_encoder_calls']+=1
        chunk,seed=self._decode(z,batch['proprio'],policy_query_index=policy_query_index)
        self.cached_condition,self.cached_action_chunk=z.detach(),chunk.detach()
        return z,chunk,seed
    policy.reset=MethodType(reset,policy);policy._full_refresh=MethodType(full,policy)
    policy._v0_update=MethodType(update,policy);policy.reset()
    return policy


def check(policy,kind):
    c=policy.metrics.counters;q=int(c['num_policy_queries']);k=policy.k_c
    full=(q+k-1)//k;light=q-full
    if (c['num_full_vlm_calls']!=full or c.get('num_condition_updater_calls',0)!=light
            or c['num_action_transformer_calls']!=q or c.get('num_generation_decoder_only_steps',0)!=0
            or q!=(policy.step_index+4)//5 or policy.nfe!=1):
        raise RuntimeError('Invocation contract changed')
    if kind!='parent':
        late=sum(1 for t in range(q) if t%k>=4) if kind=='protected' else 0
        if policy._age_calls!={'early':light-late,'late':late}:raise RuntimeError('Expert routing mismatch')
    return dict(queries=q,backbone_calls=full,updater_calls=light,action_transformer_calls=q)


def compile_check(compiler,kind,interval):
    if not compiler:return
    names={'vlm','action_transformer'}
    if kind=='parent':names|={'observation_encoder','condition_updater'}
    else:
        names|={'early_observation_encoder','early_condition_updater'}
        if kind=='protected' and interval==8:names|={'late_observation_encoder','late_condition_updater'}
    missing=[n for n in names if not compiler.records.get(n,{}).get('graphs',0)]
    if missing:raise RuntimeError('Compile bypass: '+str(missing))


def evaluate(variant,kind,interval,smoke=False):
    import sys
    import os
    from tools.simvla.refresh_allocation import setup
    c=study.checked();cfg=c['models'][variant]['config'];setup(cfg)
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
    row=f'{variant}_{kind}_k{interval}';out=study.OUTPUT/('eval_smoke' if smoke else 'rows')/row
    capture=WarningCapture()
    for name in ('torch._dynamo','torch._inductor'):logging.getLogger(name).addHandler(capture)
    suite=benchmark.get_benchmark_dict()['libero_10']()
    with torch.inference_mode():
        policy,compiler=prior.build_policy(variant+'_k4',cfg)
        parent_hash=c['models'][variant]['sha256'];model_hash=parent_hash;parity=None
        env,prompt=get_libero_env(suite.get_task(9),256,7)
        try:
            inputs=build_env_obs(env.reset())
            reference=[policy.act(*inputs,prompt).action.copy() for _ in range(80)]
            if kind!='parent':
                if not smoke:payload,model_hash=study.model_payload(c,variant,kind)
                native,_=load_native_v0_checkpoint(cfg['condition_checkpoint'],device='cuda',require_final_150k=True)
                install_exact_uint8_delta_path(native)
                model=study.initial_model(c,variant,kind,native).to('cuda').eval()
                if not smoke:model.load_state_dict(payload['model'],strict=True)
                model.requires_grad_(False)
                del native
                policy=attach(policy,model,4,compiler)
                actual=[policy.act(*inputs,prompt).action.copy() for _ in range(80)]
                if kind=='protected':
                    parity=float(np.abs(np.stack(reference)-np.stack(actual)).max())
                    if parity!=0:raise RuntimeError('Protected K4 action parity changed')
                policy.k_c=policy.refresh_every=interval
            elif interval!=4:raise ValueError('Parent control uses K4')
            policy.reset()
            for _ in range(160):policy.act(*inputs,prompt)
            check(policy,kind);compile_check(compiler,kind,interval)
            write_json(out/'runtime.json',dict(parent_checkpoint_sha256=parent_hash,model_sha256=model_hash,
                protected_k4_max_action_diff=parity,compiler=compiler.records if compiler else None,
                gpu=torch.cuda.get_device_name(0),torch=torch.__version__,command=sys.argv,
                environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','MUJOCO_GL','PYOPENGL_PLATFORM','LIBERO_CONFIG_PATH')}))
        finally:env.close()
        if smoke:
            write_json(out/'summary.json',dict(verdict='EVAL_SMOKE_PASS',identity=c['identity'],
                row=row,protected_k4_max_action_diff=parity,counts=check(policy,kind)))
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
                counts=check(policy,kind);compile_check(compiler,kind,interval)
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
                rows.append(record)
                write_json(out/'progress.json',dict(completed=len(rows),total=500,successes=sum(r['success'] for r in rows),
                    recent_wall_seconds=record['wall_seconds'],updated_unix=time.time(),phase='rollout'))
                print(f"{row} {len(rows)}/500 success={sum(r['success'] for r in rows)}/{len(rows)} wall={record['wall_seconds']:.1f}s",flush=True)
            finally:env.close()
        valid=[r for r in rows if r['timing_valid']]
        summary=dict(verdict='EVAL_COMPLETE',identity=c['identity'],row=row,episodes=len(rows),
            successes=sum(r['success'] for r in rows),success_rate=sum(r['success'] for r in rows)/len(rows),
            policy_ms_per_action=sum(r['policy_ms_total'] for r in valid)/sum(r['episode_length'] for r in valid) if valid else None,
            timing_valid_episodes=len(valid),per_task={str(t):sum(r['success'] for r in rows if r['task_id']==t) for t in range(10)},
            elapsed_seconds=time.monotonic()-started,hardware='rb2 RTX5090 compiled' if prior.IS_RB2 else 'sd1 RTX3090 eager',
            timing_scope='Synchronized policy.act includes expert selection, excludes environment/render/video. Invalid compile timings excluded from latency only.')
        write_json(out/'summary.json',summary)
