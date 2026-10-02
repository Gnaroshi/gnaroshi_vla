"""Queued compiled K8 generation control and matched state-repair evaluation."""
import argparse
import fcntl
import os
from pathlib import Path
import time
from types import MethodType

from tools.simvla import compiled_campaign as campaign
from tools.simvla.compile_benchmark import ROOT,DEFAULT_CONFIG,Replay,configure,read_json,write_json,sha
from tools.simvla.compiled_policy import attach_policy
from tools.simvla.trend_compiled_rb2 import reset_policy,recover

STORAGE=Path('/home/mingyujung/private/gnaroshi_vla_storage')
OUTPUT=STORAGE/'results/simvla/trend_condition/rollout_repair_compiled_seed01_v1'
INCOMING=STORAGE/'incoming/simvla_rollout_state_repair'
PRIOR=STORAGE/'results/simvla/trend_condition/k4_controls_compiled_seed01_v1'
ROWS=('trend_k8_full10','selected_k8','offline_k8','rollout_k8')


def base_config():
    return {**read_json(DEFAULT_CONFIG),**read_json(ROOT/'architectures/simvla/configs/compile_campaign_rb2.json'),
        **read_json(ROOT/'architectures/simvla/configs/trend_compiled_rb2.json'),
        'output':str(OUTPUT),'long_rows':list(ROWS),'campaign_module':'tools.simvla.rollout_repair_rb2',
        'extra_source_files':['tools/simvla/rollout_repair_rb2.py','tools/simvla/trend_compiled_rb2.py',
            'architectures/simvla/wrappers/run_trend_compiled_rb2.sh'],
        'scope':'K8 mechanism and development validation, not final multi-seed paper results'}


def expected_counts(row,q):
    return dict(num_full_vlm_calls=(q+7)//8,num_condition_updater_calls=q-(q+7)//8,
        num_action_transformer_calls=q*(10 if row=='trend_k8_full10' else 3),
        num_generation_decoder_only_steps=0 if row=='trend_k8_full10' else 7*q,
        num_trend_head_calls=(q+7)//8)


def check_policy(policy,row):
    q=int(policy.metrics.counters['num_policy_queries'])
    for name,value in expected_counts(row,q).items():
        if int(policy.metrics.counters.get(name,0))!=value: raise RuntimeError(f'{row}: {name} mismatch')
    if q!=(policy.step_index+4)//5: raise RuntimeError('H10/R5 cadence changed')


def check_compiler(compiler,row):
    required={'vlm','action_transformer','trend_head'}
    if row!='trend_k8_full10': required |= {'action_decoder','generation_updater'}
    required |= set(getattr(compiler,'repair_required_components',()))
    missing=[k for k in required if not compiler.records.get(k,{}).get('graphs',0)]
    if missing: raise RuntimeError('Compile bypass: '+str(missing))


def replay_factory(c,row,compiler,samples):
    return Replay(c,'condition_nfe10' if row=='trend_k8_full10' else 'ours_kc2_ng3',compiler,samples)


def make_policy(replay,c,row,manifest):
    import torch
    from methods.latentloop.modules.observed_progress import build_model
    spec=c['model_checkpoint']
    if sha(spec['path'])!=spec['sha256']: raise RuntimeError('Transferred checkpoint changed')
    payload=torch.load(spec['path'],map_location='cpu',weights_only=False)
    if payload['arm']!=spec['arm'] or payload['contract']['k_c']!=8 or payload['step']!=spec['step']:
        raise RuntimeError('Checkpoint method/horizon/step mismatch')
    mode='condition_nfe10' if row=='trend_k8_full10' else 'ours_kc2_ng3'
    policy=attach_policy(replay,c,mode,manifest)
    model=build_model(replay.native,spec['arm'],max_age=7).to('cuda').eval()
    model.load_state_dict(payload['model'],strict=True); model.requires_grad_(False)
    required=set()
    model.trend_head.forward=replay.compiler.wrap('trend_head',model.trend_head.forward)
    if model.delta_encoder is not None:
        if spec['arm']=='progress_spatial':
            model.spatial_encode=replay.compiler.wrap('spatial_encoder',model.spatial_encode)
            required.add('spatial_encoder')
        else: required.add('observation_encoder')
    if model.condition_updater is not None: required.add('condition_updater')
    if hasattr(model,'progress_head'):
        model.progress_head.forward=replay.compiler.wrap('progress_head',model.progress_head.forward)
        required.add('progress_head')
    replay.compiler.repair_required_components=required
    original_full,original_reset=policy._full_refresh,policy.reset
    policy.native_v0=model; policy.row_name=policy.mode=row; policy.k_c=policy.refresh_every=8
    def reset(self):
        original_reset(); self._trend_context=None
    def full(self,batch,*,policy_query_index):
        condition,action,seed=original_full(batch,policy_query_index=policy_query_index)
        self._trend_context=model.prepare(condition,batch['raw_rgb'],batch['proprio'],self.condition_layout.valid_mask,self.condition_layout.group_ids)
        self.metrics.counters['num_trend_head_calls']+=1
        return condition,action,seed
    def update(self,batch,*,age,policy_query_index):
        condition,_=model.predict(self._trend_context,age,batch['raw_rgb'],batch['proprio'])
        self.metrics.counters['num_condition_updater_calls']+=1
        self.metrics.counters['num_observation_encoder_calls']+=int(model.delta_encoder is not None)
        action,seed=self._decode(condition,batch['proprio'],policy_query_index=policy_query_index)
        self.cached_condition,self.cached_action_chunk=condition.detach(),action.detach()
        return condition,action,seed
    policy.reset=MethodType(reset,policy); policy._full_refresh=MethodType(full,policy); policy._v0_update=MethodType(update,policy)
    policy.reset(); return policy


def lock_released(path):
    with Path(path).open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: return False
        return True


def bundle_ready():
    if not (INCOMING/'READY.json').exists(): return False
    ready=read_json(INCOMING/'READY.json')
    if ready['manifest_sha256']!=sha(INCOMING/'manifest.json'): raise RuntimeError('Bundle marker mismatch')
    manifest=read_json(INCOMING/'manifest.json')
    for spec in manifest['checkpoints'].values():
        if sha(INCOMING/spec['file'])!=spec['sha256']: raise RuntimeError('Transferred file mismatch')
    return manifest


def evaluate(c,row,spec):
    out=OUTPUT/'online'/row
    e={**c,'output':str(out),'long_rows':[row],'model_checkpoint':spec}
    config=out/'runtime_config.json'
    if config.exists() and read_json(config)!=e: raise RuntimeError('Row config changed')
    write_json(config,e)
    previous=read_json(Path(e['reference_root'])/'campaign_contract.json')
    for key in ('norm_stats','condition_checkpoint','generation_checkpoint'):
        if sha(e[key])!=previous['artifacts'][key]: raise RuntimeError('Paper input differs: '+key)
    m=campaign.validate_manifest(read_json(campaign.manifest_path(e,'libero_10','seed01')),'libero_10','seed01')
    if m['manifest_sha256']!=previous['manifest_hashes']['libero_10/seed01']: raise RuntimeError('Episode manifest differs')
    campaign.prepare(e,out)
    for phase in ('smoke','worker'):
        if recover(e,out,phase,row): continue
        ok=campaign.run_child(e,out,phase,'libero_10','seed01',row)
        if not ok and not recover(e,out,phase,row):
            directory=out/('smoke' if phase=='smoke' else 'rows')/'libero_10/seed01'/row
            archive=out/'failed_attempts'/phase
            if not archive.exists():
                archive.parent.mkdir(parents=True,exist_ok=True)
                if directory.exists(): directory.rename(archive)
                else: archive.mkdir()
                campaign.run_child(e,out,phase,'libero_10','seed01',row)
        if not recover(e,out,phase,row): raise RuntimeError(row+' '+phase+' remains incomplete')
    return read_json(out/'rows/libero_10/seed01'/row/'summary.json')


def run_all(c):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with (OUTPUT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while not lock_released(PRIOR/'launcher.lock'):
            write_json(OUTPUT/'pipeline_status.json',dict(phase='waiting_for_rb2_controls',gpu_used=False))
            print('WAIT: current rb2 controls are still active; no GPU use',flush=True); time.sleep(60)
        results=[]; failures=[]
        old=STORAGE/'results/simvla/trend_condition/residual_k4_k8_seed01_v1/training/trend_trained_k8/train/trend_only/latest.pt'
        spec=dict(path=str(old),sha256='b9687266e047c4cfd21d0b7b9bf5c90dc520b4252576d7e2514c1096c0308e14',arm='trend_only',step=3000)
        for row in ROWS:
            if row!='trend_k8_full10':
                while not (bundle:=bundle_ready()):
                    write_json(OUTPUT/'pipeline_status.json',dict(phase='waiting_for_sd1_bundle',gpu_used=False,completed_rows=len(results)))
                    print('WAIT: sd1 is collecting/training/evaluating; bundle not ready',flush=True); time.sleep(60)
                key={'selected_k8':'selected','offline_k8':'offline_control','rollout_k8':'rollout_repair'}[row]
                item=bundle['checkpoints'][key]
                spec=dict(path=str(INCOMING/item['file']),sha256=item['sha256'],arm=bundle['selected_arm'],step=6000 if key=='selected' else 3000)
            try:
                write_json(OUTPUT/'pipeline_status.json',dict(phase='evaluation',row=row))
                results.append(dict(row=row,**evaluate(c,row,spec)))
            except Exception as exc:
                failures.append(dict(row=row,error=str(exc))); print(f'ROW_FAILED {row}: {exc}',flush=True)
            write_json(OUTPUT/'combined_summary.json',dict(complete=len(results)==len(ROWS),results=results,failures=failures,
                interpretation='Generation replacement is a diagnostic intervention. Three repair rows have identical inference architecture and compute; only training data differ. Single development seed.'))
        write_json(OUTPUT/'pipeline_status.json',dict(phase='complete' if not failures else 'finished_with_failures',failures=failures))
        return int(bool(failures))


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',choices=('all','preflight','smoke','worker'),default='all',nargs='?')
    p.add_argument('--output',type=Path); p.add_argument('--suite',default='libero_10'); p.add_argument('--seed',default='seed01')
    p.add_argument('--row',choices=ROWS); a=p.parse_args()
    c=read_json(a.output/'runtime_config.json') if a.output else base_config()
    configure(c)
    import sys
    sys.path.insert(0,c['libero_root']); os.environ['LIBERO_CONFIG_PATH']=c['libero_config']
    if a.command=='preflight':
        from tools.simvla.compile_benchmark import preflight
        preflight(c)
        print('PREFLIGHT_PASS; GPU not used',flush=True); return 0
    if a.command=='all':
        try: return run_all(c)
        except BaseException as exc:
            write_json(OUTPUT/'pipeline_status.json',dict(phase='failed',error=str(exc))); raise
    campaign.worker(c,a.output,a.suite,a.seed,a.row,smoke=a.command=='smoke',
        replay_factory=replay_factory,policy_factory=make_policy,policy_checker=check_policy,
        compiler_checker=check_compiler,reset_checker=reset_policy)
    return 0


if __name__=='__main__': raise SystemExit(main())
