"""Missing matched baselines and retained-vision mechanism controls."""
import argparse
from pathlib import Path
import subprocess
import sys

from tools.simvla import visual_memory as parent
from tools.simvla import visual_midpoint as base
from tools.simvla import visual_reanchor as anchor
from tools.simvla import failure_interventions as prior
from tools.simvla.compile_benchmark import ROOT, read_json, write_json, sha
from tools.simvla.gpu_followup_queue import run_queue

OUTPUT = prior.STORAGE/'results/simvla/visual_memory_followup/reference_age_and_controls_v1'
INCOMING = Path('/home/mingyujung/private/gnaroshi_vla_storage/incoming/simvla_visual_memory_followup')
TRAIN_ARMS = ('refresh_memory', 'refresh_spatial')
CONTROLS = ('midpoint_memory_frozen_rgb', 'midpoint_spatial_frozen_rgb',
    'refresh_memory_frozen_rgb', 'refresh_spatial_frozen_rgb',
    'midpoint_spatial_permuted_grid', 'refresh_spatial_permuted_grid',
    'midpoint_memory_mean_vision', 'refresh_memory_mean_vision',
    'midpoint_spatial_mean_vision', 'refresh_spatial_mean_vision')
BRIDGES = ('bridge_f8', 'bridge_f6', 'bridge_f2', 'bridge_f3')
FILES = ('methods/latentloop/modules/visual_memory_controls.py',
    'tools/simvla/visual_memory_followup.py', 'tools/simvla/visual_memory_followup_eval.py',
    'architectures/simvla/wrappers/run_visual_memory_followup.sh')


def row_spec(row):
    if row in BRIDGES:
        return dict(kind='bridge', interval=int(row.removeprefix('bridge_f')))
    if row == 'baseline_nfe1':
        return dict(kind='baseline', interval=1)
    pieces = row.split('_', 2)
    source, model = pieces[:2]
    control = pieces[2] if len(pieces) == 3 else 'none'
    if source not in ('refresh', 'midpoint') or model not in ('memory', 'spatial'):
        raise ValueError(row)
    if row not in (*TRAIN_ARMS, *CONTROLS):
        raise ValueError(row)
    return dict(kind='ours', interval=8, reference_age=0 if source=='refresh' else 4,
        spatial=model=='spatial', control=control, model_arm=source+'_'+model,
        parent_arm=model)


def prepare():
    old = parent.checked()
    c = dict(old)
    c.update(parent_contract=str(parent.OUTPUT/'contract.json'), parent_sha256=sha(parent.OUTPUT/'contract.json'),
        source_sha256={f:sha(ROOT/f) for f in FILES}, arms=list(TRAIN_ARMS),
        change='Capture original connector features at refresh without an additional vision call. Retain through ages4..7. Train global and spatial readers with the same K4 parent,10K objective and data. Controls freeze late RGB, reverse spatial positions within each camera, or average vision tokens within each camera.',
        comparison='Existing parent rows reused. New Long500 controls only; matched independent-env baseline NFE1 and Bridge f2/f3/f6/f8 NFE1 on rb2. Existing matched Bridge f4 reused.',
        missing_comparators=list(BRIDGES)+['baseline_nfe1'])
    c.pop('identity', None); c['identity'] = prior.digest(c)
    path = OUTPUT/'contract.json'
    if path.exists() and read_json(path) != c:
        raise RuntimeError('Followup contract changed')
    write_json(path, c)
    write_json(OUTPUT/'source.json', dict(git_head=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),files=c['source_sha256']))
    return c


def checked():
    c = read_json(OUTPUT/'contract.json'); parent.checked()
    if sha(c['parent_contract']) != c['parent_sha256']:
        raise RuntimeError('Parent changed')
    for f, expected in c['source_sha256'].items():
        if sha(ROOT/f) != expected:
            raise RuntimeError('Source changed: '+f)
    return c


def initial_model(c, row, native):
    import torch
    from methods.latentloop.modules.visual_memory_controls import ReferenceMemoryCondition
    spec = row_spec(row)
    short = parent.PARENT_INITIAL(c, 'late_every', native).early
    torch.manual_seed(c['seed'])
    return ReferenceMemoryCondition(short, spatial=spec['spatial'],
        reference_age=spec['reference_age'], control=spec['control'])


def configure_worker():
    from tools.simvla import visual_midpoint_features as features
    original = features.CachedFeatures
    original_c, anchor_c = parent.parent_checked(), anchor.checked()
    class Features(original):
        def __call__(self, age, images):
            import torch
            values = []
            for ids in self.sequence['query_ids']:
                query = ids[age]
                path = features.feature_path(parent.PARENT_OUTPUT/'features', query)
                identity = original_c['identity']
                if not path.exists():
                    path = features.feature_path(anchor.OUTPUT/'features', query)
                    identity = anchor_c['identity']
                value = torch.load(path, map_location='cpu', weights_only=False)
                if value['identity'] != identity or value['query_id'] != query:
                    raise RuntimeError('Original vision cache mismatch')
                values.append(value['features'])
            return torch.cat(values).to(images.device)
    features.CachedFeatures = Features
    base.OUTPUT, base.INCOMING, base.checked, base.initial_model = OUTPUT, INCOMING, checked, initial_model


def model_payload(c, row):
    spec = row_spec(row)
    if spec['reference_age'] == 4:
        return parent.model_payload(parent.checked(), spec['parent_arm'])
    old = base.OUTPUT, base.INCOMING
    try:
        base.OUTPUT, base.INCOMING = OUTPUT, INCOMING
        return base.model_payload(c, spec['model_arm'])
    finally:
        base.OUTPUT, base.INCOMING = old


def jobs(c):
    plan = []
    def add(name, args, path, verdict, deps=(), **extra):
        plan.append(dict(id=name,cmd=[sys.executable,'-u','-m','tools.simvla.visual_memory_followup']+args,
            deps=list(deps),summary=str(path),completion=dict(identity=c['identity'],verdict=verdict),**extra))
    if not prior.IS_RB2:
        for arm in TRAIN_ARMS:
            add('train_smoke_'+arm,['train-smoke','--row',arm],OUTPUT/'train_smoke'/arm/'summary.json','TRAIN_SMOKE_PASS')
            add('train_'+arm,['train','--row',arm],OUTPUT/'train'/arm/'summary.json','TRAIN_COMPLETE',['train_smoke_'+arm])
            add('export_'+arm,['export','--row',arm],OUTPUT/'exports'/f'{arm}.json','EXPORTED',['train_'+arm])
    rows = (*BRIDGES, 'baseline_nfe1', *TRAIN_ARMS) if prior.IS_RB2 else (*TRAIN_ARMS, *CONTROLS)
    last = None
    for row in rows:
        spec = row_spec(row); deps = []
        if prior.IS_RB2 and last: deps.append('eval_'+last)
        extra = {}
        if spec['kind']=='ours' and spec['reference_age']==0:
            if prior.IS_RB2: extra['ready_file'] = str(INCOMING/spec['model_arm']/'READY.json')
            else: deps.append('train_'+spec['model_arm'])
        add('eval_smoke_'+row,['eval-smoke','--row',row],OUTPUT/'eval_smoke'/row/'summary.json','EVAL_SMOKE_PASS',deps,**extra)
        add('eval_'+row,['eval','--row',row],OUTPUT/'rows'/row/'summary.json','EVAL_COMPLETE',['eval_smoke_'+row],**extra)
        last = row
    return plan


def main():
    p=argparse.ArgumentParser(); p.add_argument('command',nargs='?',default='all',
        choices=('all','preflight','train-smoke','train','export','eval-smoke','eval'))
    p.add_argument('--row'); a=p.parse_args()
    if a.command in ('train-smoke','train','export'):
        if a.row not in TRAIN_ARMS: raise ValueError('Only new refresh models train')
        configure_worker()
        if a.command=='export': base.export(a.row)
        else: base.train(a.row,a.command=='train-smoke')
        return 0
    if a.command in ('eval','eval-smoke'):
        from tools.simvla.visual_memory_followup_eval import evaluate
        evaluate(a.row, a.command=='eval-smoke'); return 0
    c=prepare(); plan=jobs(c); write_json(OUTPUT/'planned_jobs.json',plan)
    print('PREFLIGHT_PASS',len(plan),'jobs',OUTPUT,flush=True)
    if a.command=='preflight': return 0
    return run_queue(OUTPUT,plan,gpus=(0,) if prior.IS_RB2 else (4,5,6,7),
        predecessor=[dict(path=str(parent.OUTPUT),lock='queue.lock',allow_when_all_assigned=True)],
        environment=prior.environment,cwd=ROOT,timeout=24*3600)


if __name__=='__main__': raise SystemExit(main())
