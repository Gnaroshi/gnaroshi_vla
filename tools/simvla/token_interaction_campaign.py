"""Matched token-interaction training, Long500 tests, and rb2 handoff."""
import argparse
import copy
from contextlib import contextmanager
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

from tools.simvla import short_visual_train as base
from tools.simvla import short_visual_eval as evaluation
from tools.simvla import short_visual_campaign as campaign
from tools.simvla import geometry_comparison_rb2 as portable
from tools.simvla import gpu_followup_queue as queue
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha
from methods.latentloop.modules.token_interaction import install, gradient_coverage

OUTPUT = base.prior.STORAGE / 'results/simvla/token_interaction/matched_nfe1_v1'
MODULE = 'tools.simvla.token_interaction_campaign'
ARMS = ('warm_attention', 'warm_mlp', 'fresh_attention', 'fresh_mlp')
FILES = ('methods/latentloop/modules/token_interaction.py',
         'tools/simvla/token_interaction_audit.py', 'tools/simvla/token_interaction_campaign.py',
         'tools/simvla/token_interaction_rb2.py', 'tests/test_token_interaction.py',
         'architectures/simvla/wrappers/run_token_interaction.sh')
TRAIN, PAYLOAD = base.train, base.model_payload
REMOTE_BUNDLE = '/home/mingyujung/private/gnaroshi_vla_storage/transfers/simvla_token_interaction'


def parent_ledger(c, bundle):
    root = base.prior.STORAGE / 'results/simvla'
    paths = [Path(c['input_sources']['fresh10k_training_summary']['path'])]
    paths += [root / name / 'summary.json' for name in (
        'visual_midpoint/current_vision_nfe1_v1/train/late_every',
        'visual_readout_distillation/preserved_reader_nfe1_v1/train/warm_retain',
        'visual_attention_alignment/teacher_readout_recursive_v1/train/adapt_fresh',
        'visual_attention_alignment/teacher_readout_recursive_v1/train/attention_distill',
        'attention_block_isolation/matched_k8_nfe1_v1/train/visual_predictor_only')]
    paths += [Path(bundle['sources']['selected']['summary']['checkpoint']).parent / 'summary.json']
    records = [dict(path=str(p), sha256=sha(p), summary=read_json(p)) for p in paths]
    if len(records) != 7 or any(r['summary']['steps'] != 10000 for r in records):
        raise RuntimeError('Expected verified seven-stage70K ancestry')
    return records


def prepare():
    if base.prior.IS_RB2:
        raise RuntimeError('Training preparation belongs to sd1')
    source = base.checked()
    bundle = portable.validate_bundle()
    c = copy.deepcopy(source)
    c['bundle'] = bundle
    c['warm_checkpoint'] = str(portable.BUNDLE / 'selected.pt')
    c['warm_checkpoint_sha256'] = bundle['sources']['selected']['checkpoint_sha256']
    c['arms'] = list(ARMS)
    c['source_sha256'].update({f: sha(ROOT / f) for f in FILES})
    c['parent_ledger'] = parent_ledger(c, bundle)
    seconds = sum(r['summary']['training_seconds'] for r in c['parent_ledger'])
    c['initialization'] = {}
    for arm in ARMS:
        warm = arm.startswith('warm_')
        c['initialization'][arm] = dict(
            parent_fresh_sha256=c['warm_checkpoint_sha256'] if warm else c['model']['sha256'],
            parent_training_steps=70000 if warm else 10000,
            parent_training_seconds=seconds if warm else c['parent_ledger'][0]['summary']['training_seconds'],
            total_training_steps=75000 if warm else 20000,
            early='frozen verified fresh10K K4',
            late='frozen70K plus trainable16384 mixer' if warm else 'fresh10K copy plus trainable16384 mixer',
            visual='frozen70K' if warm else 'new seeded visual predictor', kind=arm.split('_')[1])
    c['step_counts'] = {arm: 5000 if arm.startswith('warm_') else 10000 for arm in ARMS}
    c['training_ages'] = [4, 5, 6, 7]
    c['objective'] = 'Warm: terminal first5 originalNFE10 action L1 plus0.05 mean scaled Condition MSE; fresh: mean first5 action L1 at ages4..7 plus same Condition MSE. Student NFE1; identical targets within each pair.'
    c['gradient_routing'] = 'Predicted z_t enters the next predictor and residual addition without detach. Warm trains only mixer; fresh trains late+visual+mixer. Early and original SimVLA remain frozen.'
    c['gradient_audit'] = 'After two optimizer updates verify all trainable blocks; no extra optimizer update. Frozen weights have no gradients.'
    c['protocol'] = 'Long500,H10/R5,NFE1,K8; matched attention/MLP16384 parameters; early K4 unchanged.'
    c.pop('identity', None)
    c['identity'] = base.prior.digest(c)
    path = OUTPUT / 'contract.json'
    if path.exists() and read_json(path) != c:
        raise RuntimeError('Token-interaction contract changed')
    write_json(path, c)
    write_json(OUTPUT / 'source.json', dict(commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), files=c['source_sha256']))
    return c


def checked():
    c = read_json(OUTPUT / 'contract.json')
    if base.prior.digest({k:v for k,v in c.items() if k != 'identity'}) != c['identity']:
        raise RuntimeError('Contract identity mismatch')
    for f,h in c['source_sha256'].items():
        if sha(ROOT / f) != h:
            raise RuntimeError('Source changed: ' + f)
    for record in c['input_sources'].values():
        base._verify_record(record)
    for record in c['parent_ledger']:
        base._verify_record(record)
    for path,h in c['frozen_original_sha256'].items():
        if sha(path) != h:
            raise RuntimeError('Frozen original changed: ' + path)
    if sha(c['warm_checkpoint']) != c['warm_checkpoint_sha256']:
        raise RuntimeError('Warm parent changed')
    return c


def arm_contract(c, arm):
    if arm not in ARMS:
        raise ValueError(arm)
    local = copy.deepcopy(c)
    local['steps'] = c['step_counts'][arm]
    local['warmup'] = local['steps'] // 20
    return local


def initial_model(c, arm, native):
    import torch
    warm = arm.startswith('warm_')
    if warm:
        model = portable.construct(native)
        model.load_state_dict(torch.load(c['warm_checkpoint'], map_location='cpu', weights_only=False)['model'], strict=True)
    else:
        model = base.build_model(native, 'protected_short', seed=c['seed'],
                                 early_state=base._protected_payload(c)['model'])
    return install(model, arm.split('_')[1], warm=warm, seed=c['seed'])


def loss_for(c, model, action, seq, age, features):
    from tools.simvla.condition_output_split_train import unroll, action_prediction
    from tools.simvla.rollout_state_repair import sample_from_sequence
    from methods.latentloop.modules.trend_condition import scaled_mse
    from methods.latentloop.modules.action_aligned_joint import action_loss
    features.bind(seq)
    _, predictions, _ = unroll(model, seq, age, 8)
    ages = [age] if model.interaction_warm else list(range(4, age + 1))
    losses = []
    for j in ages:
        sample = sample_from_sequence(seq, j)
        prediction = action_prediction(action, predictions[j-1], sample, steps=1)
        losses.append(action_loss(prediction, action.action_space.normalize_action(sample['target_action'])))
    loss_a = sum(losses) / len(losses)
    sample = sample_from_sequence(seq, age)
    loss_c = sum(scaled_mse(z, seq['teacher_conditions'][:,j], sample['anchor'], sample['valid'])
                 for j,z in enumerate(predictions)) / len(predictions)
    return loss_a + c['condition_weight'] * loss_c, loss_a, loss_c


def sampling(c, arm, step, smoke=False):
    age = (5,7)[step-1] if step <= 2 else (4+(step-3)%4 if arm.startswith('warm_') else 7)
    return random.Random(c['seed'] * 100000 + step), age


@contextmanager
def scope(arm):
    c = arm_contract(checked(), arm)
    replacements = dict(OUTPUT=OUTPUT, ARMS=ARMS, checked=lambda:c, initial_model=initial_model,
        loss_for=loss_for, gradient_coverage=gradient_coverage,
        sampling_step=lambda c,n,smoke=False:sampling(c,arm,n,smoke))
    old = {k:getattr(base,k) for k in replacements}
    old_module = campaign.MODULE
    try:
        for k,v in replacements.items():
            setattr(base,k,v)
        campaign.MODULE = MODULE
        yield c
    finally:
        for k,v in old.items():
            setattr(base,k,v)
        campaign.MODULE = old_module


def train(arm, smoke=False):
    with scope(arm):
        TRAIN(arm, smoke=smoke)
    path = OUTPUT / ('train_smoke' if smoke else 'train') / arm / 'summary.json'
    summary = read_json(path)
    if summary['early_sha256_before'] != summary['early_sha256_after']:
        raise RuntimeError('Protected early weights changed')
    summary['protected_early_unchanged'] = True
    write_json(path, summary)


def jobs(c):
    result = []
    for arm in ARMS:
        with scope(arm) as local:
            plan = campaign.study_jobs(local, arm)
        next(j for j in plan if j['id'] == 'train_'+arm)['completion']['steps'] = c['step_counts'][arm]
        result += plan
    campaign.validate_jobs(result)
    return result


def summarize():
    c = checked()
    rows = {p.parent.name:read_json(p) for p in (OUTPUT/'rows').glob('*/summary.json')}
    for row in rows.values():
        if row['identity'] != c['identity'] or row['episodes'] != 500:
            raise RuntimeError('Incomplete or mixed row')
    result = dict(identity=c['identity'], rows=rows, training={a:read_json(OUTPUT/'train'/a/'summary.json')
        for a in ARMS if (OUTPUT/'train'/a/'summary.json').is_file()}, updated_unix=time.time())
    write_json(OUTPUT/'comparison_summary.json', result)
    return result


def row_name(arm, interval=8, seed_index=1):
    if arm not in ARMS or interval not in (4, 8) or seed_index not in (1, 2):
        raise ValueError((arm, interval, seed_index))
    return f'{arm}_k{interval}_seed{seed_index}'


def select_candidate(result):
    candidates = {a:result['rows'][row_name(a)] for a in ARMS if row_name(a) in result['rows']}
    if not candidates:
        raise RuntimeError('No completed token-interaction candidate')
    for arm, row in candidates.items():
        if row['episodes'] != 500 or row['policy_ms_per_action'] is None:
            raise RuntimeError('Incomplete selection evidence: ' + arm)
    chosen = min(candidates, key=lambda a:(-candidates[a]['successes'], candidates[a]['policy_ms_per_action'],
                   result['training'][a]['total_training_steps'], a))
    return chosen, candidates


def select_and_export():
    c = checked()
    result = summarize()
    chosen, candidates = select_candidate(result)
    selection = dict(arm=chosen, candidates=candidates, training=result['training'], identity=c['identity'])
    path = OUTPUT/'selection.json'
    if path.exists() and read_json(path) != selection:
        raise RuntimeError('Selection changed on resume')
    write_json(path, selection)
    bundle = OUTPUT/'transfer'
    bundle.mkdir(parents=True,exist_ok=True)
    summary = result['training'][chosen]
    shutil.copy2(summary['checkpoint'],bundle/'selected.pt')
    # The fresh parent permits strict reconstruction on rb2 without sd1 cache paths.
    shutil.copy2(c['model']['checkpoint'],bundle/'fresh_parent.pt')
    shutil.copy2(c['warm_checkpoint'],bundle/'warm_parent.pt')
    write_json(bundle/'origin.json',c)
    payload = dict(selection=selection, origin_sha256=sha(bundle/'origin.json'),
        checkpoint_sha256=sha(bundle/'selected.pt'), fresh_sha256=sha(bundle/'fresh_parent.pt'),
        warm_sha256=sha(bundle/'warm_parent.pt'), manifest=c['manifest'],
        checkpoint_revision=c['model']['config']['checkpoint_revision'],
        norm_sha256=sha(c['model']['config']['norm_stats']), source_sha256=c['source_sha256'],
        sd1_runtime_parity=read_json(OUTPUT/'shards'/row_name(chosen)/'part0/eval_smoke'/
                                    row_name(chosen)/'runtime.json'))
    write_json(bundle/'bundle.json',payload)
    write_json(bundle/'READY.json',dict(sha256=sha(bundle/'bundle.json')))
    for attempt in range(3):
        try:
            subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','rb2',
                            'mkdir','-p',REMOTE_BUNDLE],check=True,timeout=60)
            subprocess.run(['rsync','-a','--exclude=READY.json',str(bundle)+'/', 'rb2:'+REMOTE_BUNDLE+'/'],
                           check=True,timeout=300)
            subprocess.run(['rsync','-a',str(bundle/'READY.json'),'rb2:'+REMOTE_BUNDLE+'/READY.json'],
                           check=True,timeout=60)
            write_json(OUTPUT/'transfer_status.json',dict(verdict='TRANSFER_COMPLETE',arm=chosen,
                                                        checkpoint_sha256=payload['checkpoint_sha256']))
            return
        except (subprocess.SubprocessError,OSError):
            if attempt == 2:
                raise
            time.sleep(20)


def run():
    owner = queue.acquire_lock(OUTPUT/'campaign.lock')
    if owner is None:
        raise RuntimeError('Campaign already running')
    try:
        c = prepare()
        if read_json(OUTPUT/'audit/control_parity.json')['verdict'] != 'GPU_CONTROL_PARITY_PASS':
            raise RuntimeError('Frozen GPU control parity missing')
        zero = read_json(OUTPUT/'audit/zero_initialization.json')
        if zero['verdict'] != 'ZERO_PARITY_PASS' or zero['identity'] != c['identity']:
            raise RuntimeError('Actual-cache zero initialization parity missing')
        plan = jobs(c)
        write_json(OUTPUT/'planned_jobs.json',plan)
        # Independent equal-budget pairs can train concurrently. Evaluation uses
        # free leased GPUs as each checkpoint becomes ready.
        rc = queue.run_queue(OUTPUT/'primary',plan,gpus=(4,5,6,7),predecessor=[],
            environment=base.prior.environment,cwd=ROOT,timeout=24*3600)
        summarize()
        select_and_export()
        state = read_json(OUTPUT/'primary/queue_status.json')
        write_json(OUTPUT/'queue_status.json',dict(phase='complete' if rc==0 else 'finished_with_failures',
            failed=state['failed'],updated_unix=time.time(),transfer=read_json(OUTPUT/'transfer_status.json')))
        return rc
    except Exception as exc:
        write_json(OUTPUT/'queue_error.json',dict(error=repr(exc),updated_unix=time.time()))
        raise
    finally:
        owner.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('all','preflight','train','train-smoke','runtime-smoke',
                                    'eval-smoke','eval','merge','summarize','export'))
    p.add_argument('--arm',choices=ARMS)
    p.add_argument('--interval',type=int,choices=(4,8),default=8)
    p.add_argument('--seed-index',type=int,choices=(1,2),default=1)
    p.add_argument('--shard',type=int,choices=range(4),default=0)
    a = p.parse_args()
    if base.prior.IS_RB2:
        p.error('Training/eager evaluation belongs to sd1')
    if a.command=='all':return run()
    if a.command=='preflight':
        c=prepare();print('PREFLIGHT_PASS',len(jobs(c)),'jobs');return 0
    if a.command=='summarize':summarize();return 0
    if a.command=='export':select_and_export();return 0
    if not a.arm:p.error('--arm required')
    if a.command in ('train','train-smoke'):
        train(a.arm,smoke=a.command=='train-smoke')
    else:
        with scope(a.arm):
            if a.command=='merge':evaluation.merge(a.arm,a.interval,a.seed_index)
            else:evaluation.evaluate(a.arm,a.interval,a.seed_index,a.shard,
                smoke=a.command!='eval',training_smoke=a.command=='runtime-smoke',bounded=a.command=='runtime-smoke')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
