"""Recover missing branches in fresh processes, preserving strict prefix checks."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import traceback

from tools.simvla import failure_interventions as prior
from tools.simvla import failure_continuations as study
from tools.simvla.compile_benchmark import read_json,write_json
from tools.simvla.refresh_allocation import OUTPUT,setup

RECOVERY = OUTPUT/'recovery'


def one(phase, case_index, point_index):
    contract,parent = study.verify()
    mode = 'bounded_k8'
    config = parent['models'][mode]['config']
    setup(config)
    import torch
    from libero.libero import benchmark
    from architectures.simvla.adapters.latentloop.native_v0_runtime import configure_strict_torch_determinism
    from architectures.simvla.wrappers.dcld_eval.rollout_runner import get_libero_env,build_env_obs
    torch.set_num_threads(1)
    configure_strict_torch_determinism(20260815)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction((total-2*1024**3)/total)
    case = parent['cases'][case_index]
    name = f"task{case['task_id']}_trial{case['trial_id']}"
    directory = prior.OUTPUT/'models'/mode/'capture'/name
    point = read_json(directory/'branches.json')[point_index]
    reference = read_json(directory/'result.json')
    t = point['query']
    destination = study.OUTPUT/'models'/mode/f'{phase}_condition'/name/f'query{t}'
    status = RECOVERY/f'{phase}_{name}_query{t}.json'
    prior.execution_plan = study.execution_plan
    with torch.inference_mode():
        policy,compiler = prior.build_policy(mode,config)
        policy = prior.attach_recorder(policy)
        suite = benchmark.get_benchmark_dict()['libero_10']()
        env,prompt = get_libero_env(suite.get_task(case['task_id']),256,7)
        try:
            inputs = build_env_obs(env.reset())
            for _ in range(80):
                policy.act(*inputs,prompt)
        finally:
            env.close()
        try:
            r = prior.rollout(policy,compiler,suite,case,destination,contract['identity'],branch=t,
                arm='predicted_1',continuation=f'{phase}_condition',reference=reference)
            if r['trace'][t]['chunk_hash'] != reference['trace'][t]['chunk_hash']:
                raise RuntimeError('Current action chunk mismatch')
            write_json(status,dict(verdict='STRICT_BRANCH_RECOVERED',result=str(destination/'result.json')))
        except RuntimeError as exc:
            if 'prefix' not in str(exc).lower() and 'chunk mismatch' not in str(exc).lower():
                raise
            probe = prior.rollout(policy,compiler,suite,case,RECOVERY/'prefix_probes'/f'{phase}_{name}',
                contract['identity'],limit=min(900,(t+1)*5),save_video=False)
            differences = []
            for a,b in zip(probe['trace'],reference['trace']):
                keys = [k for k in ('state_hash','action_prefix_hash','chunk_hash','condition_hash') if a[k]!=b[k]]
                if keys:
                    differences.append(dict(query=a['query'],different_fields=keys,
                        max_object_coordinate_diff=max((abs(x-y) for key in a['objects']
                            for x,y in zip(a['objects'][key],b['objects'][key])),default=0)))
            write_json(status,dict(verdict='PREFIX_MISMATCH_PRESERVED',error=str(exc),
                differences=differences,probe=str(RECOVERY/'prefix_probes'/f'{phase}_{name}'/'result.json'),
                reference=str(directory/'result.json'),traceback=traceback.format_exc()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=study.PHASES)
    p.add_argument('--case-index',type=int)
    p.add_argument('--point-index',type=int)
    a=p.parse_args()
    if a.phase:
        one(a.phase,a.case_index,a.point_index)
        return
    _,parent=study.verify()
    failures=[]
    for phase in study.PHASES:
        for i,case in enumerate(parent['cases']):
            name=f"task{case['task_id']}_trial{case['trial_id']}"
            points=read_json(prior.OUTPUT/'models/bounded_k8/capture'/name/'branches.json')
            for j,point in enumerate(points):
                result=study.OUTPUT/'models/bounded_k8'/f'{phase}_condition'/name/f"query{point['query']}"/'result.json'
                status=RECOVERY/f"{phase}_{name}_query{point['query']}.json"
                if result.exists() or status.exists():
                    continue
                fd=os.environ.get('GNAROSHI_GPU_LEASE_FD')
                cmd=[sys.executable,'-u','-m','tools.simvla.recover_failure_continuations',
                     '--phase',phase,'--case-index',str(i),'--point-index',str(j)]
                rc=subprocess.run(cmd,pass_fds=() if fd is None else (int(fd),),timeout=600).returncode
                if rc:
                    failures.append(dict(phase=phase,case=case,point=point,returncode=rc))
    complete=[]
    for phase in study.PHASES:
        expected=[study.OUTPUT/'models/bounded_k8'/f'{phase}_condition'/f"task{case['task_id']}_trial{case['trial_id']}"/f"query{p['query']}"/'result.json'
            for case in parent['cases'] for p in read_json(prior.OUTPUT/'models/bounded_k8/capture'/f"task{case['task_id']}_trial{case['trial_id']}"/'branches.json')]
        if all(path.exists() for path in expected):
            study.worker('bounded_k8',phase,'condition')
            complete.append(phase)
    records={p.name:read_json(p) for p in RECOVERY.glob('*.json') if p.name!='summary.json'}
    write_json(RECOVERY/'summary.json',dict(verdict='RECOVERY_AUDIT_COMPLETE',
        completely_recovered_phases=complete,branch_records=records,technical_failures=failures,
        unresolved_prefixes=sum(r.get('verdict')=='PREFIX_MISMATCH_PRESERVED' for r in records.values()),
        note='Audit completion is separate from scientific completion. Unmatched branches remain excluded; no hash threshold relaxed.'))


if __name__=='__main__':
    main()
