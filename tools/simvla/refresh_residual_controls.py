"""Diagnose refresh fit error, then compare anchor-preserving mappings."""
import argparse
from pathlib import Path
import sys
import time
from types import MethodType

from tools.simvla import refresh_calibration as pipeline
from tools.simvla.error_compensation_common import ROOT, configure, identity, read_json, sha, write_json

PREVIOUS_SD = pipeline.SD_OUTPUT
PREVIOUS_RB = pipeline.RB_OUTPUT
SD_OUTPUT = PREVIOUS_SD.parent/'anchor_residual_k4_k8_seed01_v1'
RB_OUTPUT = PREVIOUS_RB.parent/'anchor_residual_k4_k8_compiled_seed01_v1'
MODULE = 'tools.simvla.refresh_residual_controls'
PARENT_CONFIG = pipeline.configuration
PARENT_JOBS = pipeline.sd_jobs
PARENT_SUMMARY = pipeline.summarize


def configuration(host):
    c = PARENT_CONFIG(host)
    if host == 'sd1':
        bootstrap = PREVIOUS_SD/'train/bootstrap'
        report = read_json(bootstrap/'summary.json')
        if report['verdict'] != 'TRAIN_COMPLETE' or report['steps'] != 3000:
            raise RuntimeError('Shared initialization incomplete')
        candidates = {}
        for variant in ('fixed','anchor_input','ridge'):
            for k in (4,8):
                path = PREVIOUS_SD/'train'/f'{variant}_k{k}'/'latest.pt'
                summary = read_json(path.parent/'summary.json')
                if summary['verdict'] != 'TRAIN_COMPLETE' or sha(path) != summary['checkpoint_sha256']:
                    raise RuntimeError('Diagnostic checkpoint incomplete or changed')
                candidates[f'{variant}_k{k}'] = dict(path=str(path),sha256=summary['checkpoint_sha256'],identity=summary['identity'])
        c.update(output=str(SD_OUTPUT), model={**c['model'],'anchor_exact':True},
            common_initialization=dict(path=str(bootstrap/'latest.pt'),sha256=report['checkpoint_sha256'],
                identity=report['identity'],step=3000), diagnostic_checkpoints=candidates,
            diagnostic_windows=30,
            predecessors=[dict(path=str(PREVIOUS_SD),lock='queue.lock',allow_when_all_assigned=True)],
            student_condition_description='C0 + [mapping(current observation) - mapping(refresh observation)], with or without refresh ridge fit. No recursive predicted condition; no Generation Loop.',
            training_description='Reuse verified shared3K initialization. Four matched3K NFE3 continuations (fixed/ridge x K4/K8). Same loss, data, optimizer and sampling as preceding campaign. Diagnose the original six candidates without modifying them; hold-C0 online controls. No SR/MSE stopping gate.')
    else:
        c.update(output=str(RB_OUTPUT),long_rows=['fixed','ridge'],
            predecessors=[dict(path=str(PREVIOUS_RB),lock='queue.lock')],
            scope='Anchor-preserving fixed/ridge mappings, K4/K8, seed01, 500 episodes per row, NFE3/H10/R5, no Generation Loop.')
    c['extra_source_files'] += ['tools/simvla/refresh_residual_controls.py',
        'architectures/simvla/wrappers/run_refresh_residual_controls.sh']
    return c


def diagnose(c, variant):
    import torch
    from architectures.simvla.adapters.refresh_calibration.train import runtime, sequence_at, language, _balanced_indices
    from methods.refresh_calibration.model import RefreshCalibratedCondition
    from methods.latentloop.modules.trend_condition import scaled_mse
    from methods.latentloop.modules.action_aligned_joint import action_loss
    run_id = identity(c)
    path = SD_OUTPUT/'diagnostics'/f'{variant}.json'
    if path.exists():
        if read_json(path)['identity'] != run_id: raise RuntimeError('Diagnostic identity changed')
        return
    device, frozen, action, data, heldout, bank = runtime(c)
    indices = _balanced_indices(heldout.identities,limit=c['diagnostic_windows'],seed=c['seed'])
    records=[]
    with torch.inference_mode():
        for k in (4,8):
            spec=c['diagnostic_checkpoints'][f'{variant}_k{k}']
            if sha(spec['path']) != spec['sha256']: raise RuntimeError('Diagnostic input checksum changed')
            payload=torch.load(spec['path'],map_location='cpu',weights_only=False)
            if payload['identity'] != spec['identity']: raise RuntimeError('Diagnostic input provenance changed')
            model=RefreshCalibratedCondition(variant,**payload['contract']['model']).to(device).eval()
            model.load_state_dict(payload['model'],strict=True)
            for position,index in enumerate(indices):
                batch=sequence_at(heldout,index,device)
                anchor=batch['anchor_condition']; mask=batch['valid_mask'].bool()
                torch.cuda.synchronize(); begin=time.perf_counter()
                state=model.prepare(anchor,batch['image_sequence'][:,0],batch['proprio_sequence'][:,0],
                    language(batch,bank,device),mask,batch['group_ids'])
                at_anchor=model.predict(state,batch['image_sequence'][:,0],batch['proprio_sequence'][:,0])
                torch.cuda.synchronize(); setup_ms=(time.perf_counter()-begin)*1000
                initial_error=float(scaled_mse(at_anchor,anchor,anchor,mask))
                for age in range(1,k):
                    torch.cuda.synchronize(); begin=time.perf_counter()
                    raw=model.predict(state,batch['image_sequence'][:,age],batch['proprio_sequence'][:,age])
                    torch.cuda.synchronize(); prediction_ms=(time.perf_counter()-begin)*1000
                    teacher=batch['teacher_conditions'][:,age-1]
                    reference=action.action_space.normalize_action(batch['teacher_actions'][:,age-1])
                    for intervention,condition in (
                        ('original_mapping',raw),
                        ('remove_refresh_fit_error',anchor+(raw-at_anchor)),
                        ('hold_anchor',anchor),('original_condition',teacher)):
                        output=action.decode_action_from_condition(condition,batch['proprio_sequence'][:,age],
                            steps=3,initial_noise=batch['explicit_noises'][:,age-1],return_debug=True).final_action_latent
                        cos=torch.nn.functional.cosine_similarity(condition.float(),teacher.float(),dim=-1)
                        records.append(dict(k_c=k,index=index,task_id=int(batch['task_id'][0]),age=age,
                            intervention=intervention,refresh_condition_mse=initial_error,
                            condition_mse=float(scaled_mse(condition,teacher,anchor,mask)),
                            condition_cosine=float(cos[mask].mean()),action_l1=float(action_loss(output,reference)),
                            diagnostic_setup_ms=setup_ms,diagnostic_prediction_ms=prediction_ms))
                print(f'DIAGNOSTIC {variant} K{k} {position+1}/{len(indices)}',flush=True)
    groups={}
    for k in (4,8):
        for name in sorted({r['intervention'] for r in records}):
            rows=[r for r in records if r['k_c']==k and r['intervention']==name]
            groups[f'k{k}/{name}']={metric:sum(r[metric] for r in rows)/len(rows)
                for metric in ('refresh_condition_mse','condition_mse','condition_cosine','action_l1')}
    write_json(path,dict(verdict='DIAGNOSTIC_COMPLETE',identity=run_id,variant=variant,records=records,means=groups,
        heldout_split=heldout.contract(),checkpoints=c['diagnostic_checkpoints'],
        scope='Fixed checkpoints; paired held-out inputs/noise; all actions decoded with native NFE3 against NFE10 cached teacher. Component times are diagnostics, not policy latency. No online success claim.'))


def hold_policy(c,row,*,smoke=False,k_c=4):
    from tools.simvla.error_compensation_eval import make_policy
    policy=make_policy(c,'condition_naive3',k_c=min(k_c,4))
    policy.k_c=policy.refresh_every=k_c
    policy.row_name=policy.mode=row
    def refill(self,batch):
        q=self.query_index; age=q % self.k_c
        self.metrics.counters['num_policy_queries'] += 1
        if age == 0:
            _,actions,seed=self._full_refresh(batch,policy_query_index=q)
        else:
            actions,seed=self._decode(self.cached_condition,batch['proprio'],policy_query_index=q)
            self.metrics.counters['num_condition_reuses'] += 1
        self.action_queue.clear()
        for action in actions[0,:5]: self.action_queue.append((action.detach(),'hold_anchor'))
        self.query_trace.append(dict(policy_query_index=q,age=age,action_horizon=10,execution_horizon=5))
        self.query_index += 1
        return dict(refreshed=age==0,age=age,queue_mode='hold_anchor',action_noise_seed=seed)
    policy._refill_action_queue=MethodType(refill,policy)
    return policy


def check_hold(policy,row,actual,k_c=4):
    q=int(policy.metrics.counters['num_policy_queries']); full=(q+k_c-1)//k_c
    expected=dict(transformer=3*q,generation=0,condition=0)
    if any(actual.get(k,0)!=v for k,v in expected.items()): raise RuntimeError('Hold control executed learned modules')
    if (policy.metrics.counters['num_full_vlm_calls'] != full or q != (policy.step_index+4)//5
            or policy.metrics.counters.get('num_condition_reuses',0)!=q-full):
        raise RuntimeError('Hold control cadence changed')
    return dict(queries=q,full_vlm=full,condition_reuses=q-full,**expected)


def sd_jobs(c):
    jobs=PARENT_JOBS(c)
    jobs=[j for j in jobs if j['id']!='train_bootstrap_k8']
    diagnostic_ids=[f'diagnose_{v}' for v in ('fixed','anchor_input','ridge')]
    for j in jobs:
        if j['id'].startswith('train_'): j['deps']=diagnostic_ids
    prefix=[c['python'],'-u','-m',MODULE]
    diagnostics=[dict(id=f'diagnose_{v}',cmd=prefix+['diagnose','--variant',v],
        summary=str(SD_OUTPUT/'diagnostics'/f'{v}.json'),
        completion=dict(verdict='DIAGNOSTIC_COMPLETE',identity=identity(c),variant=v))
        for v in ('fixed','anchor_input','ridge')]
    controls=[dict(id=f'hold_k{k}',deps=diagnostic_ids,cmd=prefix+['hold','--k',str(k)],
        summary=str(SD_OUTPUT/'online'/f'kc{k}_hold'/'summary.json'),
        completion=dict(verdict='EVALUATION_COMPLETE',identity=identity(c),episodes=500)) for k in (4,8)]
    return diagnostics+jobs+controls


def summarize(c,host):
    PARENT_SUMMARY(c,host)
    p=Path(c['output'])/'comparison_summary.json'
    d=read_json(p)
    d['method']='C0 + mapping(current observation) - mapping(refresh observation); fixed or refresh-calibrated mapping'
    d['training']='Reuse preceding verified3K bootstrap; four independently trained3K candidates; no bootstrap rerun'
    if host=='sd1':
        d['hold_controls']={str(k):read_json(p) for k in (4,8)
            if (p:=SD_OUTPUT/'online'/f'kc{k}_hold'/'summary.json').exists()}
        d['diagnostics']={v:read_json(p)['means'] for v in ('fixed','anchor_input','ridge')
            if (p:=SD_OUTPUT/'diagnostics'/f'{v}.json').exists()}
    write_json(Path(c['output'])/'comparison_summary.json',d)


def install():
    pipeline.SD_OUTPUT=SD_OUTPUT; pipeline.RB_OUTPUT=RB_OUTPUT
    pipeline.INCOMING=pipeline.RB_STORAGE/'incoming/simvla_refresh_residual_controls'
    pipeline.MODULE=MODULE
    pipeline.VARIANTS=('fixed','ridge')
    pipeline.CELLS=tuple((v,k) for k in (4,8) for v in pipeline.VARIANTS)
    pipeline.configuration=configuration; pipeline.sd_jobs=sd_jobs; pipeline.summarize=summarize


def main():
    install()
    if len(sys.argv)>1 and sys.argv[1] in ('diagnose','hold'):
        parser=argparse.ArgumentParser()
        parser.add_argument('command'); parser.add_argument('--variant',choices=('fixed','anchor_input','ridge'))
        parser.add_argument('--k',type=int,choices=(4,8),default=4)
        args=parser.parse_args()
        c=read_json(SD_OUTPUT/'runtime_config.json'); configure(c)
        if args.command=='diagnose': diagnose(c,args.variant)
        else:
            from tools.simvla.error_compensation_eval import run
            for smoke in (True,False):
                run(c,'hold',smoke=smoke,k_c=args.k,policy_factory=hold_policy,counter_checker=check_hold)
        return 0
    return pipeline.main()


if __name__=='__main__': raise SystemExit(main())
