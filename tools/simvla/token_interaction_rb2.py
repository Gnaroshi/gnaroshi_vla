"""Wait for sd1 selection, then verify and evaluate the compiled K8 model."""
import argparse
import copy
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from tools.simvla import failure_interventions as prior
from tools.simvla import gpu_followup_queue as queue
from tools.simvla import geometry_comparison_rb2 as portable
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha
from methods.latentloop.modules.token_interaction import install

OUTPUT = prior.STORAGE / 'results/simvla/token_interaction/matched_nfe1_v1'
BUNDLE = prior.STORAGE / 'transfers/simvla_token_interaction'
MODULE = 'tools.simvla.token_interaction_rb2'


def validate_bundle():
    if sha(BUNDLE/'bundle.json') != read_json(BUNDLE/'READY.json')['sha256']:
        raise RuntimeError('Incomplete bundle')
    b = read_json(BUNDLE/'bundle.json')
    for name,key in [('selected.pt','checkpoint_sha256'),('fresh_parent.pt','fresh_sha256'),
                     ('warm_parent.pt','warm_sha256'),('origin.json','origin_sha256')]:
        if sha(BUNDLE/name) != b[key]:
            raise RuntimeError('Transfer hash mismatch: '+name)
    for name,h in b['source_sha256'].items():
        if sha(ROOT/name) != h:
            raise RuntimeError('Source differs: '+name)
    return b


def prepare():
    b = validate_bundle()
    cfg = prior.config('recurrent_k4')
    if (cfg['checkpoint_revision'] != b['checkpoint_revision'] or sha(cfg['norm_stats']) != b['norm_sha256']
            or cfg['model_checkpoint']['sha256'] != b['fresh_sha256']):
        raise RuntimeError('Official model, normalization or K4 reference changed')
    c = dict(bundle=b, model=dict(config=cfg,sha256=b['fresh_sha256']), manifest=b['manifest'],
             selection=b['selection'], source_sha256=b['source_sha256'],
             protocol='RTX5090,compiled,Long500x2,NFE1,H10/R5,K8; unchanged early K4 reused after exact parity')
    c['identity'] = prior.digest(c)
    path = OUTPUT/'contract.json'
    if path.exists() and read_json(path) != c:
        raise RuntimeError('Remote contract changed')
    write_json(path,c)
    return c


def checked():
    c = read_json(OUTPUT/'contract.json')
    if c['identity'] != prior.digest({k:v for k,v in c.items() if k!='identity'}) or validate_bundle()!=c['bundle']:
        raise RuntimeError('Remote runtime identity changed')
    return c


def initial_model(c, row, native):
    import torch
    arm = c['selection']['arm']
    warm = arm.startswith('warm_')
    if warm:
        model = portable.construct(native)
        model.load_state_dict(torch.load(BUNDLE/'warm_parent.pt',map_location='cpu',weights_only=False)['model'],strict=True)
    else:
        from methods.latentloop.modules.short_visual_training import build_model
        state = torch.load(BUNDLE/'fresh_parent.pt',map_location='cpu',weights_only=False)['model']
        model = build_model(native,'protected_short',seed=7,early_state=state)
    return install(model,arm.split('_')[1],warm=warm)


def payload(c, row):
    import torch
    value = torch.load(BUNDLE/'selected.pt',map_location='cpu',weights_only=False)
    arm = c['selection']['arm']
    if (value['identity'] != c['selection']['identity'] or value['arm'] != arm
            or value['step'] != (5000 if arm.startswith('warm_') else 10000)):
        raise RuntimeError('Selected checkpoint provenance mismatch')
    return value,c['bundle']['checkpoint_sha256']


def evaluate(seed_index, smoke=False):
    from tools.simvla import attention_query_eval as evaluation
    c = copy.deepcopy(checked())
    row = f'selected_seed{seed_index}'
    seed = c['manifest']['action_noise_seed_base'] + seed_index - 1
    c['manifest']['action_noise_seed_base'] = seed
    original = evaluation.prior.build_policy
    def builder(mode,cfg):
        policy,compiler = original(mode,cfg)
        policy.action_noise_seed_base = seed
        return policy,compiler
    writer = evaluation.write_json
    def write(path,value):
        if Path(path).name in ('summary.json','runtime.json') or Path(path).parent.name=='episodes':
            value = dict(value,action_noise_seed_base=seed)
        writer(path,value)
    evaluation.prior.build_policy = builder
    evaluation.write_json = write
    evaluation.study = SimpleNamespace(OUTPUT=OUTPUT,checked=lambda:c,model_payload=payload,
        initial_model=initial_model,previous=SimpleNamespace(OUTPUT=OUTPUT/'unused'))
    evaluation.evaluate(row,smoke=smoke)
    if not smoke:
        summarize()


def summarize():
    c = checked()
    expected = {(r['task_id'],r['trial_id']) for r in c['manifest']['episodes']}
    rows = {}
    for seed_index in (1,2):
        name = f'selected_seed{seed_index}'
        path = OUTPUT/'rows'/name/'summary.json'
        if not path.exists():continue
        summary = read_json(path)
        data = [read_json(p) for p in (path.parent/'episodes').glob('*.json')]
        seed = c['manifest']['action_noise_seed_base'] + seed_index - 1
        if (summary['identity'] != c['identity'] or len(data)!=500 or len(expected)!=500
                or {(r['task_id'],r['trial_id']) for r in data}!=expected
                or any(r['identity']!=c['identity'] or r['model_sha256']!=c['bundle']['checkpoint_sha256']
                       or r['action_noise_seed_base']!=seed for r in data)
                or summary['successes']!=sum(r['success'] for r in data)):
            raise RuntimeError('Mixed/incomplete remote row')
        rows[name] = summary
    write_json(OUTPUT/'comparison_summary.json',dict(identity=c['identity'],complete=len(rows)==2,
        rows=rows,selection=c['selection'],baseline_references='Existing matched baseline NFE1, recurrent K4, Bridgef2/f3/f4/f8; see completed protocols.',
        hardware='rb2 RTX5090 compiled',seed_scope='Two action-noise seeds, same trained checkpoint and initial environment states.'))


def gpu_smoke(kind='attention', family='warm'):
    """Compile the new path immediately, before sd1's selected artifact arrives."""
    import numpy as np
    import torch
    from tools.simvla.refresh_allocation import setup
    from tools.simvla.attention_query_eval import attach_visual,check,compile_check
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import load_native_v0_checkpoint
    from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import install_exact_uint8_delta_path
    from architectures.simvla.adapters.latentloop.native_v0_runtime import configure_strict_torch_determinism
    cfg = prior.config('recurrent_k4');setup(cfg)
    from libero.libero import benchmark
    from architectures.simvla.wrappers.dcld_eval.rollout_runner import build_env_obs,get_libero_env
    torch.set_num_threads(1);configure_strict_torch_determinism(20260815)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((total-2*1024**3)/total)
    b = portable.validate_bundle()
    with torch.inference_mode():
        policy,compiler = prior.build_policy('recurrent_k4',cfg)
        suite = benchmark.get_benchmark_dict()['libero_10']()
        env,prompt = get_libero_env(suite.get_task(9),256,7)
        try:
            inputs=build_env_obs(env.reset())
            reference=np.stack([policy.act(*inputs,prompt).action.copy() for _ in range(80)])
            native,_=load_native_v0_checkpoint(cfg['condition_checkpoint'],device='cuda',require_final_150k=True)
            install_exact_uint8_delta_path(native)
            if family=='warm':
                model=portable.construct(native).to('cuda').eval()
                model.load_state_dict(torch.load(portable.BUNDLE/'selected.pt',map_location='cpu',weights_only=False)['model'],strict=True)
            else:
                from methods.latentloop.modules.short_visual_training import build_model
                state=torch.load(cfg['model_checkpoint']['path'],map_location='cpu',weights_only=False)['model']
                model=build_model(native,'protected_short',seed=7,early_state=state).to('cuda').eval()
            install(model,kind,warm=family=='warm');model.requires_grad_(False)
            policy=attach_visual(policy,model,4,compiler)
            actual=np.stack([policy.act(*inputs,prompt).action.copy() for _ in range(80)])
            diff=float(np.abs(reference-actual).max())
            if diff!=0:raise RuntimeError('Preselection K4 parity changed')
            policy.k_c=policy.refresh_every=8;policy.reset()
            for _ in range(160):policy.act(*inputs,prompt)
            counts=check(policy,'selected');compile_check(compiler,'selected')
            components = component_profile(policy,model,compiler,inputs,prompt) if family=='warm' and kind=='attention' else None
            write_json(OUTPUT/f'audit/compiled_runtime_{family}_{kind}.json',dict(verdict='COMPILED_RUNTIME_PASS',kind=kind,family=family,
                protected_k4_max_action_diff=diff,counts=counts,compiler=compiler.records,
                component_profile_smoke=components,
                source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (
                    ROOT/'methods/latentloop/modules/token_interaction.py',Path(__file__))},
                parent_sha256=b['sources']['selected']['checkpoint_sha256'],
                gpu=torch.cuda.get_device_name(0),torch=torch.__version__))
        finally:env.close()


def component_profile(policy, model, compiler, inputs, prompt):
    import numpy as np
    import torch
    captured, restored = {}, []
    entries = [('preprocess',policy,'preprocess'),
        ('backbone',policy.condition_adapter,'encode_condition'),
        ('action_transformer_and_decode',policy.action_adapter,'decode_action_from_condition'),
        ('vision',model,'feature_provider'),('visual_readout',model.visual,'forward')]
    for label,expert in model.experts():
        entries.extend([(label+'_observation_encoder',expert.delta_encoder,'forward'),
                        (label+'_condition_updater',expert.condition_updater,'forward')])
    try:
        for name,owner,attribute in entries:
            original=getattr(owner,attribute)
            def capture(*args,_name=name,_fn=original,**kwargs):
                captured[_name]=(_fn,args,kwargs)
                return _fn(*args,**kwargs)
            restored.append((owner,attribute,original))
            setattr(owner,attribute,capture)
        for _ in range(40):policy.act(*inputs,prompt)
    finally:
        for owner,attribute,original in restored:setattr(owner,attribute,original)
    missing={entry[0] for entry in entries}-captured.keys()
    if missing:raise RuntimeError('Missing profile inputs: '+str(missing))
    # The mixer is fused inside the late updater above. Its standalone number
    # describes intrinsic cost and must not be added again to the updater cost.
    _,args,kwargs=captured['late_condition_updater']
    z,code=args[:2];updater=model.late.condition_updater
    ids=torch.arange(z.shape[1],device=z.device)
    age=torch.as_tensor(kwargs['age'],device=z.device,dtype=torch.long)
    if age.ndim==0:age=age.expand(z.shape[0])
    hidden=updater.down(updater.norm(z))+updater.delta_projection(code)[:,None]
    hidden=hidden+updater.token_embedding(ids)[None]+updater.group_embedding(kwargs['group_ids'])
    hidden=hidden+updater.age_embedding(age).unsqueeze(1)+kwargs['token_feature']
    standalone=compiler.wrap('isolated_mixer',updater.mixer.forward)
    captured['mixer_standalone']=(standalone,(hidden,kwargs['valid_mask']),{})
    result={}
    for name,(fn,args,kwargs) in captured.items():
        for _ in range(20):fn(*args,**kwargs)
        graphs=compiler.graph_count();values=[]
        for _ in range(100):
            torch.cuda.synchronize();start=time.perf_counter();fn(*args,**kwargs);torch.cuda.synchronize()
            values.append(1000*(time.perf_counter()-start))
        if compiler.graph_count()!=graphs:raise RuntimeError('Component timing incurred compilation: '+name)
        result[name]=dict(mean_ms=float(np.mean(values)),median_ms=float(np.median(values)),
                          p95_ms=float(np.percentile(values,95)),samples=len(values))
    return result


def profile():
    """Isolated warmed fixed-observation timing; separate from episode latency."""
    import numpy as np
    import torch
    from tools.simvla.refresh_allocation import setup
    from tools.simvla.attention_query_eval import attach_visual,check,compile_check
    from architectures.simvla.adapters.latentloop.native_v0_checkpoint import load_native_v0_checkpoint
    from architectures.simvla.adapters.latentloop.efficient_multirate.efficient_delta import install_exact_uint8_delta_path
    from architectures.simvla.adapters.latentloop.native_v0_runtime import configure_strict_torch_determinism
    c=checked();cfg=c['model']['config'];setup(cfg)
    from libero.libero import benchmark
    from architectures.simvla.wrappers.dcld_eval.rollout_runner import build_env_obs,get_libero_env
    torch.set_num_threads(1);configure_strict_torch_determinism(20260815)
    with torch.inference_mode():
        policy,compiler=prior.build_policy('recurrent_k4',cfg)
        native,_=load_native_v0_checkpoint(cfg['condition_checkpoint'],device='cuda',require_final_150k=True)
        install_exact_uint8_delta_path(native)
        model=initial_model(c,'selected',native).to('cuda').eval()
        value,_=payload(c,'selected');model.load_state_dict(value['model'],strict=True);model.requires_grad_(False)
        policy=attach_visual(policy,model,8,compiler)
        env,prompt=get_libero_env(benchmark.get_benchmark_dict()['libero_10']().get_task(9),256,7)
        records=[]
        try:
            inputs=build_env_obs(env.reset())
            for _ in range(160):policy.act(*inputs,prompt)
            graph_count=compiler.graph_count()
            for _ in range(400):
                query=not bool(policy.action_queue);age=policy.query_index%8 if query else None
                torch.cuda.synchronize();start=time.perf_counter();policy.act(*inputs,prompt);torch.cuda.synchronize()
                records.append(dict(query=query,age=age,ms=1000*(time.perf_counter()-start)))
            compile_check(compiler,'selected');check(policy,'selected')
            if compiler.graph_count()!=graph_count:raise RuntimeError('Profile incurred compilation')
            by_age={str(j):float(np.mean([r['ms'] for r in records if r['age']==j])) for j in range(8)}
            components=component_profile(policy,model,compiler,inputs,prompt)
            write_json(OUTPUT/'profile/summary.json',dict(verdict='PROFILE_COMPLETE',identity=c['identity'],
                samples=len(records),policy_ms_per_action=float(np.mean([r['ms'] for r in records])),
                query_ms_by_age=by_age,queue_ms=float(np.mean([r['ms'] for r in records if not r['query']])),
                scope='Warmed fixed-observation synchronized policy.act; age0 full backbone,age4 vision,other ages early/late updater. Episode latency remains primary.',
                components=components,component_scope='Isolated same-input warmed calls; late updater includes mixer, visual readout includes predicted KV, vision excludes backbone. Do not sum overlapping calls.',
                compiler=compiler.records,checkpoint_sha256=c['bundle']['checkpoint_sha256']))
        finally:env.close()


def jobs(c):
    result=[]
    for seed_index in (1,2):
        row=f'selected_seed{seed_index}'
        for smoke in (True,False):
            name=row+('_smoke' if smoke else '_eval')
            result.append(dict(id=name,cmd=[sys.executable,'-u','-m',MODULE,'smoke' if smoke else 'eval',
                '--seed-index',str(seed_index)],deps=[] if smoke else [row+'_smoke'],
                summary=str(OUTPUT/('eval_smoke' if smoke else 'rows')/row/'summary.json'),
                completion=dict(identity=c['identity'],verdict='EVAL_SMOKE_PASS' if smoke else 'EVAL_COMPLETE',
                    **({} if smoke else dict(episodes=500)))))
    result.append(dict(id='profile',cmd=[sys.executable,'-u','-m',MODULE,'profile'],deps=['selected_seed1_eval','selected_seed2_eval'],
        summary=str(OUTPUT/'profile/summary.json'),completion=dict(identity=c['identity'],verdict='PROFILE_COMPLETE')))
    return result


def run():
    owner=queue.acquire_lock(OUTPUT/'campaign.lock')
    if owner is None:raise RuntimeError('Remote campaign already running')
    try:
        smoke=[dict(id='compile_runtime_'+family+'_'+kind,
            cmd=[sys.executable,'-u','-m',MODULE,'gpu-smoke','--kind',kind,'--family',family],deps=[],
            summary=str(OUTPUT/f'audit/compiled_runtime_{family}_{kind}.json'),
            completion=dict(verdict='COMPILED_RUNTIME_PASS',kind=kind,family=family))
            for family in ('warm','fresh') for kind in ('attention','mlp')]
        rc=queue.run_queue(OUTPUT/'preselection',smoke,gpus=(0,),predecessor=[],environment=prior.environment,cwd=ROOT,timeout=3600)
        if rc:return rc
        start=time.monotonic()
        while not (BUNDLE/'READY.json').exists():
            write_json(OUTPUT/'queue_status.json',dict(phase='waiting_for_sd1_selection',gpu_reserved=False,
                expected='Four matched sd1 training/evaluation rows then selected checkpoint',updated_unix=time.time()))
            if time.monotonic()-start>48*3600:raise TimeoutError('sd1 selection missing after48h')
            print('WAIT sd1 selection; no GPU lease held',flush=True);time.sleep(30)
        c=prepare();plan=jobs(c);write_json(OUTPUT/'planned_jobs.json',plan)
        rc=queue.run_queue(OUTPUT/'evaluation',plan,gpus=(0,),predecessor=[],environment=prior.environment,cwd=ROOT,timeout=12*3600)
        summarize()
        state=read_json(OUTPUT/'evaluation/queue_status.json')
        write_json(OUTPUT/'queue_status.json',dict(phase='complete' if rc==0 else 'finished_with_failures',
            failed=state['failed'],updated_unix=time.time()))
        return rc
    except Exception as exc:
        write_json(OUTPUT/'queue_error.json',dict(error=repr(exc),updated_unix=time.time()));raise
    finally:owner.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('all','preflight','gpu-smoke','smoke','eval','profile'))
    p.add_argument('--seed-index',type=int,choices=(1,2),default=1)
    p.add_argument('--kind',choices=('attention','mlp'),default='attention')
    p.add_argument('--family',choices=('warm','fresh'),default='warm')
    a=p.parse_args()
    if not prior.IS_RB2:p.error('This runtime is rb2 only')
    if a.command=='all':return run()
    if a.command=='preflight':
        cfg=prior.config('recurrent_k4');portable.validate_bundle()
        if not Path(cfg['norm_stats']).is_file():raise FileNotFoundError(cfg['norm_stats'])
        print('PREFLIGHT_PASS; selected checkpoint may arrive later');return 0
    if a.command=='gpu-smoke':gpu_smoke(a.kind,a.family)
    elif a.command=='profile':profile()
    else:evaluate(a.seed_index,a.command=='smoke')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
