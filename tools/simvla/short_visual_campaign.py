"""Two short-training studies and an automatic held-out-noise confirmation."""
import argparse
import os
from pathlib import Path
import sys
import time

from tools.simvla import short_visual_train as training
from tools.simvla import short_visual_eval as evaluation
from tools.simvla import geometry_target_train as predecessor
from tools.simvla import gpu_followup_queue as queue
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha

MODULE = 'tools.simvla.short_visual_campaign'


def evaluation_jobs(c, arm, interval, seed_index, deps=()):
    row = evaluation.row_name(arm, interval, seed_index)
    result = []
    arguments = ['--arm', arm, '--interval', str(interval), '--seed-index', str(seed_index)]
    def add(name, command, path, dependencies, **complete):
        result.append(dict(id=row+'_'+name,
            cmd=[sys.executable, '-u', '-m', MODULE, command]+arguments,
            deps=list(dependencies), summary=str(path),
            completion=dict(identity=c['identity'], **complete)))
    add('smoke', 'eval-smoke', evaluation.output_directory(arm,interval,seed_index,0)/'eval_smoke'/row/'summary.json',
        deps, verdict='EVAL_SMOKE_PASS')
    for shard in range(4):
        add(str(shard), 'eval', evaluation.output_directory(arm,interval,seed_index,shard)/'rows'/row/'summary.json',
            [row+'_smoke'], verdict='EVAL_COMPLETE', episodes=125,
            action_noise_seed_base=c['manifest']['action_noise_seed_base']+seed_index-1)
        result[-1]['cmd'] += ['--shard',str(shard)]
    add('merge', 'merge', training.OUTPUT/'rows'/row/'summary.json',
        [row+'_'+str(shard) for shard in range(4)], verdict='EVAL_COMPLETE', episodes=500)
    return result


def study_jobs(c, arm):
    out = training.OUTPUT
    row = evaluation.row_name(arm,8,1)
    plan = [dict(id='train_smoke_'+arm,
        cmd=[sys.executable,'-u','-m',MODULE,'train-smoke','--arm',arm], deps=[],
        summary=str(out/'train_smoke'/arm/'summary.json'),
        completion=dict(identity=c['identity'],verdict='TRAIN_SMOKE_PASS',steps=2)),
        dict(id='runtime_'+arm,
        cmd=[sys.executable,'-u','-m',MODULE,'runtime-smoke','--arm',arm],deps=['train_smoke_'+arm],
        summary=str(out/'pretraining_runtime'/row/'part0/eval_smoke'/row/'summary.json'),
        completion=dict(identity=c['identity'],verdict='EVAL_SMOKE_PASS')),
        dict(id='train_'+arm,
        cmd=[sys.executable,'-u','-m',MODULE,'train','--arm',arm],deps=['runtime_'+arm],
        summary=str(out/'train'/arm/'summary.json'),
        completion=dict(identity=c['identity'],verdict='TRAIN_COMPLETE',steps=10000))]
    deps = ['train_'+arm]
    for interval in ((4,8) if arm=='fresh_joint' else (8,)):
        row_jobs = evaluation_jobs(c,arm,interval,1,deps)
        plan += row_jobs
        deps = [row_jobs[-1]['id']]
    return plan


def validate_jobs(plan):
    ids = {j['id'] for j in plan}
    if len(ids)!=len(plan) or len({j['summary'] for j in plan})!=len(plan):
        raise RuntimeError('Duplicate jobs or output paths')
    done=set()
    while len(done)<len(ids):
        ready={j['id'] for j in plan if set(j['deps']).issubset(done)}-done
        if not ready:
            raise RuntimeError('Cyclic or missing dependency')
        done.update(ready)


def select_short(rows):
    complete={name:r for name,r in rows.items()
        if r.get('verdict')=='EVAL_COMPLETE' and r.get('episodes')==500 and r.get('interval')==8}
    if not complete:
        raise RuntimeError('No completed short K8 checkpoint')
    return min(complete,key=lambda name:(-complete[name]['successes'],complete[name]['policy_ms_per_action'],name))


def summarize():
    c=training.checked()
    rows={}
    for path in sorted((training.OUTPUT/'rows').glob('*/summary.json')):
        r=read_json(path)
        if r['identity']!=c['identity'] or r['episodes']!=500:
            raise RuntimeError('Mixed completed row')
        rows[path.parent.name]=r
    references={}
    for name,path in {
        'long_training_seed01':predecessor.OUTPUT/'comparison_summary.json',
        'long_training_seed02':predecessor.OUTPUT/'confirmation/comparison_summary.json',
    }.items():
        if path.is_file():
            references[name]=dict(path=str(path),sha256=sha(path),result=read_json(path))
    result=dict(identity=c['identity'],rows=rows,completed_long_training_references=references,
        training={arm:read_json(training.OUTPUT/'train'/arm/'summary.json') for arm in training.ARMS
                  if (training.OUTPUT/'train'/arm/'summary.json').is_file()},
        hardware='sd1 RTX3090 eager',updated_unix=time.time(),
        interpretation='Short training changes optimization and removes learned-reader distillation targets. It is not an initialization-only ablation of the70K training history. No Latent Bridge superiority inferred from unmatched hardware.')
    write_json(training.OUTPUT/'comparison_summary.json',result)
    return result


def reference_integrity():
    from tools.simvla import geometry_target_confirmation as confirmation
    path = confirmation.OUTPUT/'contract.json'
    if not path.exists():
        return dict(complete=False,reason='Missing reference confirmation contract')
    c=confirmation.checked()
    expected={(r['task_id'],r['trial_id']) for r in c['manifest']['episodes']}
    if len(expected)!=500:
        raise RuntimeError('Reference manifest is not Long500')
    for arm in confirmation.ARMS:
        root=confirmation.OUTPUT/'rows'/arm
        if not (root/'summary.json').exists() or not (root/'episode_outcomes.json').exists():
            return dict(complete=False,reason='Missing reference row '+arm)
        summary,records=read_json(root/'summary.json'),read_json(root/'episode_outcomes.json')
        checksum=c['sources'][arm]['checkpoint_sha256']
        ids=[(r['task_id'],r['trial_id']) for r in records]
        if (summary.get('verdict')!='EVAL_COMPLETE' or summary.get('episodes')!=500
                or summary.get('identity')!=c['identity'] or summary.get('model_sha256')!=checksum
                or len(ids)!=500 or set(ids)!=expected or len(set(ids))!=500
                or any(r['identity']!=c['identity'] or r['model_sha256']!=checksum
                       or r['action_noise_seed_base']!=c['manifest']['action_noise_seed_base'] for r in records)
                or sum(r['success'] for r in records)!=summary['successes']):
            raise RuntimeError('Reference checkpoint/seed/episode outcome mismatch: '+arm)
    return dict(complete=True,identity=c['identity'],action_noise_seed_base=c['manifest']['action_noise_seed_base'])


def wait_predecessor():
    start=time.monotonic()
    spec=dict(path=str(predecessor.OUTPUT),lock='queue.lock')
    while True:
        path=predecessor.OUTPUT/'queue_status.json'
        state=read_json(path) if path.exists() else {}
        unlocked = not queue.predecessor_pending(spec)
        subqueues = list((predecessor.OUTPUT/'queues').glob('*/queue.lock'))
        subqueues += list((predecessor.OUTPUT/'confirmation/queues').glob('*/queue.lock'))
        drained = unlocked and all(not queue.predecessor_pending(dict(path=str(p.parent),lock=p.name))
                                   for p in subqueues)
        if drained and state.get('phase') in ('complete','finished_with_failures'):
            return
        if drained and (predecessor.OUTPUT/'queue_error.json').exists():
            pid = state.get('pid')
            try:
                if pid:os.kill(pid,0)
                alive=bool(pid)
            except ProcessLookupError:
                alive=False
            if not alive:
                write_json(training.OUTPUT/'predecessor_exception.json',dict(
                    original_status=state,error=read_json(predecessor.OUTPUT/'queue_error.json'),
                    all_queue_leases_released=True,updated_unix=time.time()))
                return
        if time.monotonic()-start>48*3600:
            raise TimeoutError('Predecessor has not terminated after48h')
        write_json(training.OUTPUT/'queue_status.json',dict(phase='waiting_for_predecessor',
            predecessor=str(predecessor.OUTPUT),predecessor_phase=state.get('active_phase',state.get('phase')),
            planned_studies=['fresh_joint','protected_short','selected_second_noise'],
            pid=os.getpid(),updated_unix=time.time()))
        print('WAIT existing geometry training and full confirmation; new studies are queued',flush=True)
        time.sleep(30)


def run():
    owner=queue.acquire_lock(training.OUTPUT/'queue.lock')
    if owner is None:
        raise RuntimeError('Short-training campaign already has an owner')
    try:
        c=training.prepare()
        plan=sum([study_jobs(c,arm) for arm in training.ARMS],[])
        validate_jobs(plan)
        write_json(training.OUTPUT/'planned_jobs.json',plan)
        write_json(training.OUTPUT/'followup_plan.json',dict(
            studies=3,first_two_jobs=len(plan),confirmation_jobs=6,
            confirmation='After both K8 rows complete, select highestSR; ties lowerlatency thenname; evaluate500 on next action-noise seed. Reuse completed70K seed02 comparator.',
            reuse='Frozen early protected K4 equals completed fresh10K K4 exactly; no duplicate K4 rollout.',
            technical_failures='Retry once, preserve logs, continue independent study. No SR/MSE stop.'))
        wait_predecessor()
        done,failed=set(),{}
        def phase(name,local):
            write_json(training.OUTPUT/'queue_status.json',dict(phase='running',active_phase=name,
                detail=str(training.OUTPUT/'queues'/name/'queue_status.json'),
                completed=sorted(done),failed=failed,pid=os.getpid(),updated_unix=time.time()))
            queue.run_queue(training.OUTPUT/'queues'/name,local,gpus=(4,5,6,7),predecessor=[],
                environment=training.prior.environment,cwd=ROOT,timeout=24*3600)
            state=read_json(training.OUTPUT/'queues'/name/'queue_status.json')
            done.update(state['completed']);failed.update(state['failed'])
            for key in done:failed.pop(key,None)
            summarize()
        # Repair only missing noise-confirmation jobs; trained models and complete
        # episodes are retained. Its error is independent of the two new studies.
        from tools.simvla import geometry_target_confirmation as confirmation
        reference_contract = confirmation.OUTPUT/'contract.json'
        if reference_contract.exists():
            repair = confirmation.jobs(read_json(reference_contract))
            for job in repair:
                job['cmd'][job['cmd'].index('-m')+1] = 'tools.simvla.short_visual_reference_repair'
            plan = repair + plan
            write_json(training.OUTPUT/'planned_jobs.json',plan)
            for arm in confirmation.ARMS:
                phase('reference_repair_'+arm,[j for j in repair if j['arm']==arm])
            confirmation.summarize()
        independent_training=[j for arm in training.ARMS for j in study_jobs(c,arm)
                              if j['id'].startswith(('train_smoke_','runtime_','train_'))]
        phase('short_training',independent_training)
        for arm in training.ARMS:
            phase(arm,study_jobs(c,arm))
        result=summarize()
        candidates={arm:result['rows'][evaluation.row_name(arm,8,1)] for arm in training.ARMS
                    if evaluation.row_name(arm,8,1) in result['rows']}
        if candidates:
            selected=select_short(candidates)
            selection=dict(arm=selected,seed01_result=candidates[selected],
                checkpoint_sha256=result['training'][selected]['checkpoint_sha256'])
            destination=training.OUTPUT/'selection.json'
            if destination.exists() and read_json(destination)!=selection:
                raise RuntimeError('Confirmation selection changed on resume')
            write_json(destination,selection)
            second=evaluation_jobs(c,selected,8,2)
            plan+=second
            write_json(training.OUTPUT/'planned_jobs.json',plan)
            phase('selected_second_noise',second)
        else:
            failed['selected_second_noise']=dict(reason='Both source training/evaluation studies failed technically')
        summarize()
        reference = reference_integrity()
        reference_ready = reference['complete']
        short_failed={k:v for k,v in failed.items() if not k.startswith('confirm_')}
        write_json(training.OUTPUT/'comparison_status.json',dict(
            short_jobs_complete=not bool(short_failed),long_training_seed02_complete=reference_ready,
            verdict='COMPARISON_READY' if not short_failed and reference_ready else 'COMPARISON_INCOMPLETE',
            reference=reference))
        write_json(training.OUTPUT/'queue_status.json',dict(
            phase='finished_with_failures' if failed else 'complete',completed=sorted(done),failed=failed,
            active={},pending=[],total_jobs=len(plan),updated_unix=time.time()))
        return int(bool(failed))
    except Exception as exc:
        write_json(training.OUTPUT/'queue_error.json',dict(error=repr(exc),updated_unix=time.time()))
        raise
    finally:
        owner.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',nargs='?',default='all',choices=(
        'all','preflight','summarize','train-smoke','runtime-smoke','train','eval-smoke','eval','merge'))
    p.add_argument('--arm',choices=training.ARMS)
    p.add_argument('--interval',type=int,choices=(4,8),default=8)
    p.add_argument('--seed-index',type=int,choices=(1,2),default=1)
    p.add_argument('--shard',type=int,choices=range(4),default=0)
    args=p.parse_args()
    if training.prior.IS_RB2:
        p.error('This campaign is assigned to sd1')
    if args.command=='all':return run()
    if args.command=='preflight':
        c=training.prepare()
        plan=sum([study_jobs(c,a) for a in training.ARMS],[])
        validate_jobs(plan)
        print('PREFLIGHT_PASS',len(plan),'initial jobs +6 confirmation jobs',training.OUTPUT,flush=True)
        return 0
    if args.command=='summarize':
        print(summarize());return 0
    if not args.arm:p.error('--arm required')
    if args.command in ('train','train-smoke'):
        training.train(args.arm,smoke=args.command=='train-smoke')
    elif args.command=='merge':
        evaluation.merge(args.arm,args.interval,args.seed_index)
    else:
        evaluation.evaluate(args.arm,args.interval,args.seed_index,args.shard,
            smoke=args.command!='eval',training_smoke=args.command=='runtime-smoke',
            bounded=args.command=='runtime-smoke')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
