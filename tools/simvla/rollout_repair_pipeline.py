"""Wait for the current sd1 campaign, then collect, repair and export once."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import time

from tools.simvla.error_compensation_common import ROOT,read_json,write_json,sha,identity
from tools.simvla.error_compensation_campaign import prepare,campaign
from tools.simvla.rollout_state_repair import VARIANTS,disjoint_indices,state_hash_array

CONFIG=ROOT/'architectures/simvla/configs/rollout_state_repair_sd1.json'
ARMS=('frozen_trend_residual','progress_only','progress_residual','progress_spatial')


def dependency_ready(root):
    root=Path(root)
    p=root/'pipeline_status.json'
    if not p.exists(): return False
    status=read_json(p)
    if status.get('phase') in ('technical_failure','interrupted_or_error'):
        raise RuntimeError('Predecessor has a technical error; preserve current experiments and inspect '+str(p))
    if status.get('phase')!='complete': return False
    completion=read_json(root/'campaign_complete.json')
    if completion.get('verdict')!='COMPLETE' or completion.get('failed'): raise RuntimeError('Predecessor completion mismatch')
    for arm in ARMS:
        for k in (8,4):
            r=read_json(root/'online'/f'kc{k}_{arm}'/'summary.json')
            if r.get('verdict')!='EVALUATION_COMPLETE' or r.get('episodes')!=500:
                raise RuntimeError('Incomplete predecessor evaluation')
    return True


def choose_candidate(root):
    root=Path(root)
    rows=[]
    for arm in ARMS:
        row=read_json(root/'online'/f'kc8_{arm}'/'summary.json')
        rows.append((row['successes'],-row['policy_ms_per_action'],arm))
    winner=max(rows)[2]
    path=root/'train'/winner/'latest.pt'
    return dict(path=str(path),sha256=sha(path),arm=winner),dict(
        rule='Highest completed K8 development SR; tie: lower policy time; no absolute SR threshold',
        selected=winner,development_seed_reused=True,confirmatory_three_seed_result=False,
        candidates=[dict(arm=a,successes=s,policy_ms_per_action=-t) for s,t,a in rows])


def collection_pool(c):
    import torch
    from tools.simvla.error_compensation_common import configure
    configure(c)
    os.environ['LIBERO_CONFIG_PATH']=str(Path(c['predecessor'])/'libero_config')
    from libero.libero import benchmark
    suite=benchmark.get_benchmark_dict()['libero_10']()
    base=Path(c['upstream'])/'evaluation/libero/LIBERO/libero/libero/init_files'
    pools={}
    for i in range(10):
        task=suite.get_task(i)
        official_path=base/task.problem_folder/task.init_states_file
        raw_path=official_path.with_name(official_path.name.replace('.pruned_init','.init'))
        raw=torch.load(raw_path,map_location='cpu',weights_only=False)
        official=torch.load(official_path,map_location='cpu',weights_only=False)
        indices=disjoint_indices(raw,official,c['collection_episodes_per_task'],c['collection_seed']+i)
        pools[str(i)]=dict(path=str(raw_path),sha256=sha(raw_path),indices=indices,
            hashes=[state_hash_array(raw[j]) for j in indices],official_path=str(official_path),
            official_sha256=sha(official_path),official_state_hashes=[state_hash_array(x) for x in official],
            scope='Additional .init states excluded from standard 50 .pruned_init states; collection only')
    return pools


def jobs(c,config,smoke):
    out=Path(c['output']); extra=['--smoke'] if smoke else []
    prefix=[c['python'],'-u','-m','tools.simvla.rollout_state_repair']
    plan=[]
    shards=[0] if smoke else list(range(4))
    for s in shards:
        plan.append(dict(id=f'collect_{s}',deps=[],cmd=prefix+['collect','--config',str(config),'--shard',str(s)]+extra,
            expected_verdict='COLLECTION_COMPLETE',summary=str(out/('collection_smoke' if smoke else 'collection')/f'shard{s}_summary.json')))
    for v in VARIANTS:
        plan.append(dict(id='train_'+v,deps=[f'collect_{s}' for s in shards],
            cmd=prefix+['train','--config',str(config),'--variant',v]+extra,
            summary=str(out/('smoke' if smoke else 'train')/v/'summary.json')))
    if not smoke:
        plan.append(dict(id='export_rb2',deps=['train_'+v for v in VARIANTS],
            cmd=[c['python'],'-u','-m','tools.simvla.rollout_repair_pipeline','--export-config',str(config)],
            expected_verdict='BUNDLE_EXPORTED',summary=str(out/'export_summary.json')))
    for k in ([8] if smoke else [8,4]):
        for v in VARIANTS:
            plan.append(dict(id=f'eval_kc{k}_{v}',deps=['train_'+v],
                cmd=prefix+['eval','--config',str(config),'--variant',v,'--k-c',str(k)]+extra,
                summary=str(out/('eval_smoke' if smoke else 'online')/f'kc{k}_{v}'/'summary.json')))
    return plan


def summary_data(c):
    out=Path(c['output'])
    rows={}
    for k in (8,4):
        for v in VARIANTS:
            path=out/'online'/f'kc{k}_{v}'/'summary.json'
            if path.exists(): rows[f'kc{k}_{v}']=read_json(path)
    return dict(complete=len(rows)==4,rows=rows,
        selected_checkpoint=c['selected_checkpoint'],selection=read_json(out/'selection.json'),
        training_steps_each=c['steps'],shared_architecture=True,added_inference_operations=0,
        scope='Development test of student-state mismatch, not a new-method novelty claim or heldout seed confirmation')


def summarize(c):
    write_json(Path(c['output'])/'comparison_summary.json',summary_data(c))


def export(c):
    out=Path(c['output']); bundle=out/'rb2_bundle'; bundle.mkdir(exist_ok=True)
    files={'selected':Path(c['selected_checkpoint']['path'])}
    files.update({v:out/'train'/v/'latest.pt' for v in VARIANTS})
    manifest=dict(selected_arm=c['selected_checkpoint']['arm'],checkpoints={},
        sd1_contract_sha256=sha(out/'contract.json'),selection=read_json(out/'selection.json'))
    for name,source in files.items():
        target=bundle/(name+'.pt')
        if target.exists():
            if sha(target)!=sha(source): raise RuntimeError('Export checkpoint changed')
        else: os.link(source,target)
        manifest['checkpoints'][name]=dict(file=target.name,sha256=sha(target))
    write_json(bundle/'manifest.json',manifest)
    write_json(bundle/'sd1_results.json',summary_data(c))
    write_json(bundle/'READY.json',dict(manifest_sha256=sha(bundle/'manifest.json')))
    dest=c['rb2_destination']
    host,path=dest.split(':',1)
    for attempt in range(3):
        try:
            ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10']
            subprocess.run([*ssh,host,'mkdir','-p',path],check=True,timeout=60)
            remote=['-e','ssh -o BatchMode=yes -o ConnectTimeout=10']
            subprocess.run(['rsync','-a','--partial',*remote,'--exclude=READY.json',str(bundle)+'/',dest+'/'],check=True,timeout=600)
            subprocess.run(['rsync','-a',*remote,str(bundle/'READY.json'),dest+'/'],check=True,timeout=60)
            write_json(out/'export_summary.json',dict(identity=identity(c),verdict='BUNDLE_EXPORTED',manifest=manifest))
            return
        except (subprocess.SubprocessError,OSError):
            if attempt==2: raise
            time.sleep(30)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--preflight',action='store_true'); p.add_argument('--export-config'); a=p.parse_args()
    if a.export_config:
        export(read_json(a.export_config)); return 0
    override=read_json(CONFIG)
    c={**read_json(ROOT/'architectures/simvla/configs/trend_condition_sd1.json'),
       **read_json(ROOT/'architectures/simvla/configs/observed_progress_sd1.json'),**override}
    out=Path(c['output']); out.mkdir(parents=True,exist_ok=True)
    with (out/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if a.preflight:
                pool=collection_pool(c)
                write_json(out/'preflight.json',dict(verdict='CPU_INPUT_POOL_PASS',collection_states=pool,
                    gpu_used=False,source_dependency_ready=dependency_ready(c['predecessor'])))
                print('PREFLIGHT_PASS: disjoint initial states; no GPU use',flush=True); return 0
            while not dependency_ready(c['predecessor']):
                write_json(out/'pipeline_status.json',dict(phase='waiting_for_predecessor',gpu_used=False,predecessor=c['predecessor']))
                print('WAIT: current sd1 experiment must complete; no GPU use',flush=True); time.sleep(60)
            c['selected_checkpoint'],selection=choose_candidate(c['predecessor'])
            c['collection_states']=collection_pool(c)
            write_json(out/'selection.json',selection)
            prepare(c); config=out/'runtime_config.json'; write_json(config,c)
            for smoke in (True,False):
                write_json(out/'pipeline_status.json',dict(phase='smoke' if smoke else 'collect_train_evaluate'))
                if campaign(c,config,smoke,job_builder=jobs,summarizer=summarize):
                    raise RuntimeError('A technical worker failure remains; inspect status and logs')
            export(c)
            write_json(out/'pipeline_status.json',dict(phase='complete',rb2_bundle_sent=True)); return 0
        except BaseException as exc:
            write_json(out/'pipeline_status.json',dict(phase='failed',error=f'{type(exc).__name__}: {exc}'))
            raise


if __name__=='__main__': raise SystemExit(main())
