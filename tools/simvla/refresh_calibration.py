"""Queued matched refresh-calibration training on sd1 and compiled rb2 evaluation."""
import argparse
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from tools.simvla.error_compensation_common import ROOT, CONFIG, configure, environment, identity, read_json, write_json, sha
from tools.simvla.gpu_followup_queue import run_queue

VARIANTS = ('fixed', 'anchor_input', 'ridge')
CELLS = tuple((v, k) for k in (4, 8) for v in VARIANTS)
SD_STORAGE = Path('/home/mingyujung/shared/nvme1/mingyujung/robotics/gnaroshi_vla')
RB_STORAGE = Path('/home/mingyujung/private/gnaroshi_vla_storage')
SD_OUTPUT = SD_STORAGE/'results/simvla/refresh_calibration/long_k4_k8_seed01_v1'
RB_OUTPUT = RB_STORAGE/'results/simvla/refresh_calibration/long_k4_k8_compiled_seed01_v1'
INCOMING = RB_STORAGE/'incoming/simvla_refresh_calibration'
MODULE = 'tools.simvla.refresh_calibration'


def key_for(variant, k):
    return f'{variant}_k{k}'


def configuration(host):
    if host == 'sd1':
        return dict(read_json(CONFIG), output=str(SD_OUTPUT), steps=3000, bootstrap_steps=3000,
            smoke_steps=2, warmup_steps=150, save_interval=500, condition_weight=.05,
            training_k_c=8, training_condition_ages=list(range(1, 8)), evaluation_condition_intervals=[4, 8],
            smoke_policy_actions=41, model=dict(dim=960, width=64, ridge_lambda=.01),
            extra_source_files=['methods/refresh_calibration/model.py',
                'methods/refresh_calibration/__init__.py',
                'architectures/simvla/wrappers/run_refresh_calibration.sh'],
            student_condition_description='Refresh-calibrated observation-to-condition function: fixed / anchor input / ridge; no Generation Loop.',
            training_description='Shared3K Condition bootstrap then six independent3K continuations (three arms x K4/K8), NFE3 for every query, original NFE10 same-noise targets. All use identical feature architecture/common initialization, batch2, AdamW1e-4 cosine. No SR gate.',
            predecessors=[dict(path=str(SD_STORAGE/'results/simvla/condition_nfe/native_matched_k2_seed01_v1'),lock='queue.lock')])
    if host != 'rb2': raise ValueError(host)
    c = read_json(ROOT/'architectures/simvla/configs/compile_benchmark_rb2.json')
    c.update(read_json(ROOT/'architectures/simvla/configs/compile_campaign_rb2.json'))
    c.update(output=str(RB_OUTPUT), steps=3000, long_rows=list(VARIANTS), other_rows=[], other_suites=[],
        seeds=['seed01'], smoke_episodes=1, smoke_actions=41, warmup_actions=40, videos_per_row=2,
        campaign_module=MODULE, new_training=True,
        extra_source_files=['tools/simvla/refresh_calibration.py','tools/simvla/gpu_followup_queue.py',
            'architectures/simvla/wrappers/run_refresh_calibration.sh'],
        profile_observations=str(RB_STORAGE/'results/simvla/compiled_paper/three_seed_v2/rows/libero_10/seed01/baseline/observations.pt'),
        predecessors=[dict(path=str(RB_STORAGE/p),lock='queue.lock') for p in (
            'results/simvla/condition_nfe/compiled_seed01_v1',
            'results/simvla/libero_pro/long_position_task_seed01_v1')],
        scope='Development seed01, matched 500-episode LIBERO-Long per row; three arms x K4/K8. NFE3/H10/R5/EGL. No Generation module. No success-based stopping.')
    return c


def rb_environment(c, gpu):
    if gpu != 0: raise ValueError('rb2 GPU0 only')
    e = dict(os.environ)
    for key in ('GALLIUM_DRIVER','LIBGL_ALWAYS_SOFTWARE','LP_NUM_THREADS','EGL_DEVICE_ID'):
        e.pop(key, None)
    e.update(CUDA_VISIBLE_DEVICES='0', MUJOCO_EGL_DEVICE_ID='0', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
        HF_HOME=c['hf_home'], HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', USE_TF='0',
        TOKENIZERS_PARALLELISM='false', SIMVLA_UPSTREAM_ROOT=c['upstream'],
        CUBLAS_WORKSPACE_CONFIG=':4096:8', CUDA_DEVICE_MAX_CONNECTIONS='1', NVIDIA_TF32_OVERRIDE='0',
        PYTHONHASHSEED='20260815', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
        LIBERO_CONFIG_PATH=c['libero_config'], PYTHONPATH=str(ROOT), PYTHONUNBUFFERED='1',
        NUMBA_CACHE_DIR=str(RB_OUTPUT/'runtime/numba'), MPLCONFIGDIR=str(RB_OUTPUT/'runtime/matplotlib'),
        TORCHINDUCTOR_CACHE_DIR=str(RB_OUTPUT/'runtime/inductor'), TRITON_CACHE_DIR=str(RB_OUTPUT/'runtime/triton'))
    return e


def child(c, command, variant, k, *, smoke=False, output=None):
    args = [c['python'], '-u', '-m', MODULE, command, '--host', 'sd1' if socket.gethostname()=='jbrserver1' else 'rb2',
            '--variant', variant, '--k', str(k)]
    if smoke: args.append('--smoke')
    if output: args += ['--output', str(output)]
    lease = os.environ.get('GNAROSHI_GPU_LEASE_FD')
    subprocess.run(args, cwd=ROOT, check=True, pass_fds=() if lease is None else (int(lease),))


def export_candidate(c, variant, k):
    from architectures.simvla.adapters.refresh_calibration.train import path_for
    key = key_for(variant, k)
    path = path_for(c, variant, k)
    report = read_json(path.parent/'summary.json')
    if report['checkpoint_sha256'] != sha(path) or report['identity'] != identity(c):
        raise RuntimeError('Export checkpoint identity mismatch')
    bundle = SD_OUTPUT/'export'/key
    bundle.mkdir(parents=True, exist_ok=True)
    target = bundle/'model.pt'
    # Only the current small checkpoint is copied, never the teacher/cache.
    shutil.copy2(path, target)
    shutil.copy2(path.parent/'summary.json', bundle/'training_summary.json')
    shutil.copy2(path.parent/'training_contract.json', bundle/'training_contract.json')
    shutil.copy2(path.parent/'heldout_after.json', bundle/'heldout_after.json')
    manifest = dict(verdict='TRANSFER_READY', variant=variant, k_c=k, steps=c['steps'],
        training_identity=report['identity'], sha256=sha(target), file='model.pt',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    write_json(bundle/'READY.json', manifest)
    destination = str(INCOMING/key)
    remote = ['-e','ssh -o BatchMode=yes -o ConnectTimeout=15']
    for attempt in range(3):
        try:
            subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','rb2','mkdir','-p',destination],check=True,timeout=60)
            subprocess.run(['rsync','-a','--partial',*remote,'--exclude=READY.json',str(bundle)+'/',f'rb2:{destination}/'],check=True,timeout=180)
            subprocess.run(['rsync','-a',*remote,str(bundle/'READY.json'),f'rb2:{destination}/'],check=True,timeout=60)
            write_json(bundle/'export_summary.json', manifest)
            return
        except (subprocess.SubprocessError, OSError):
            if attempt == 2: raise
            time.sleep(15)


def sd_train_cell(c, variant, k):
    from architectures.simvla.adapters.refresh_calibration.train import path_for
    for smoke in (True, False):
        child(c, 'train', variant, k, smoke=smoke)
        if smoke and variant != 'bootstrap': child(c, 'sd-eval', variant, k, smoke=True)
    if variant != 'bootstrap': export_candidate(c, variant, k)
    report = read_json(path_for(c, variant, k).parent/'summary.json')
    write_json(SD_OUTPUT/'completed'/f'train_{key_for(variant,k)}.json',
        dict(verdict='TRAIN_CELL_COMPLETE', identity=identity(c), variant=variant, k_c=k, steps=c['steps'],
             training=report, exported=variant!='bootstrap'))


def sd_eval(c, variant, k, smoke):
    from tools.simvla.error_compensation_eval import run
    from architectures.simvla.adapters.refresh_calibration.policy import make_sd1_policy, check_counts
    run(c, variant, smoke=smoke, k_c=k, policy_factory=make_sd1_policy, counter_checker=check_counts)
    path=Path(c['output'])/('eval_smoke' if smoke else 'online')/f'kc{k}_{variant}'/'summary.json'
    summary=read_json(path)
    summary.update(candidate_training_k_c=k, action_nfe=3, generation_loop=False)
    write_json(path,summary)


def sd_jobs(c):
    run_id = identity(c)
    prefix = [c['python'],'-u','-m',MODULE]
    jobs=[]
    for variant,k in (('bootstrap',8),*CELLS):
        key=key_for(variant,k)
        jobs.append(dict(id='train_'+key, deps=[] if variant=='bootstrap' else ['train_bootstrap_k8'],
            cmd=prefix+['sd-train-cell','--host','sd1','--variant',variant,'--k',str(k)],
            summary=str(SD_OUTPUT/'completed'/f'train_{key}.json'),
            completion=dict(verdict='TRAIN_CELL_COMPLETE',identity=run_id,variant=variant,k_c=k,steps=c['steps'])))
    for variant,k in CELLS:
        jobs.append(dict(id='eval_'+key_for(variant,k),deps=['train_'+key_for(variant,k)],
            cmd=prefix+['sd-eval','--host','sd1','--variant',variant,'--k',str(k)],
            summary=str(SD_OUTPUT/'online'/f'kc{k}_{variant}'/'summary.json'),
            completion=dict(identity=run_id,verdict='EVALUATION_COMPLETE',episodes=500)))
    return jobs


def candidate_spec(c, variant, k):
    directory = INCOMING/key_for(variant,k)
    marker = read_json(directory/'READY.json')
    path = directory/marker['file']
    if marker['verdict']!='TRANSFER_READY' or marker['variant']!=variant or marker['k_c']!=k or marker['steps']!=c['steps'] or sha(path)!=marker['sha256']:
        raise RuntimeError('Transferred checkpoint contract mismatch')
    current = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if marker['source_commit'] != current:
        raise RuntimeError('Training and evaluation source commits differ')
    return {**marker,'path':str(path)}


def recover(c, output, variant, smoke):
    from tools.simvla import compiled_campaign as campaign
    manifest=read_json(output/'manifests/libero_10/seed01/episode_manifest.json')
    specs=sorted(manifest['episodes'],key=lambda s:(-s['task_id'],s['trial_id']))
    if smoke: specs=specs[:c['smoke_episodes']]
    run_id=campaign.digest(dict(campaign=campaign.digest(read_json(output/'campaign_contract.json')),
        suite='libero_10',seed='seed01',row=variant,smoke=smoke))
    return campaign.summarize_cell(output/('smoke' if smoke else 'rows')/'libero_10/seed01'/variant,
        run_id,[(s['task_id'],s['trial_id']) for s in specs])


def rb_cell(c, variant, k):
    from tools.simvla import compiled_campaign as campaign
    output = RB_OUTPUT/'cells'/key_for(variant,k)
    local = dict(c,output=str(output),candidate=candidate_spec(c,variant,k),condition_interval=k,long_rows=[variant])
    dest=output/'runtime_config.json'
    if dest.exists() and read_json(dest)!=local: raise RuntimeError('Row config changed')
    write_json(dest,local)
    campaign.prepare(local,output)
    for smoke in (True,False):
        if recover(local,output,variant,smoke): continue
        directory=output/('smoke' if smoke else 'rows')/'libero_10/seed01'/variant
        if list((directory/'episodes').glob('*.json')):
            archive=output/'failed_attempts'/f'{"smoke" if smoke else "eval"}_{time.time_ns()}'
            archive.parent.mkdir(parents=True,exist_ok=True)
            directory.rename(archive)
        try:
            child(local,'rb-smoke' if smoke else 'rb-eval',variant,k,output=output)
        except subprocess.CalledProcessError:
            if not recover(local,output,variant,smoke): raise
        if not recover(local,output,variant,smoke): raise RuntimeError('Evaluation incomplete')
    # Profiling has its own resumable job; an instrumentation failure cannot lose SR.
    write_json(RB_OUTPUT/'completed'/f'eval_{key_for(variant,k)}.json',
        dict(verdict='EVAL_CELL_COMPLETE',variant=variant,k_c=k,episodes=500,
             result=recover(local,output,variant,False),candidate_sha256=local['candidate']['sha256']))


def rb_profile(c, variant, k):
    from tools.simvla.compiled_profile import profile
    from architectures.simvla.adapters.refresh_calibration import policy as adapter
    output = RB_OUTPUT/'cells'/key_for(variant,k)
    local=read_json(output/'runtime_config.json')
    def instruments(policy,replay,instrument):
        model=policy.native_v0
        instrument(model.features,'forward','observation_encoder')
        instrument(model,'prepare','refresh_preparation_inclusive')
        instrument(model,'predict','condition_generation_inclusive')
        if variant=='ridge': instrument(model,'fit_correction','ridge_fit_inclusive')
        if variant=='anchor_input': instrument(model.anchor_encoder,'forward','anchor_encoding')
    profile(local,output,'libero_10','seed01',variant,replay_factory=adapter.replay_factory,
        policy_factory=adapter.make_rb2_policy,policy_checker=adapter.check_counts,
        reset_checker=adapter.check_reset,compiler_checker=adapter.check_compiler,extra_instruments=instruments)
    result=read_json(output/'latency/libero_10/seed01'/variant/'profile.json')
    write_json(RB_OUTPUT/'completed'/f'profile_{key_for(variant,k)}.json',
        dict(verdict='PROFILE_COMPLETE',variant=variant,k_c=k,timing_valid=result['timing_valid'],result=result))


def rb_jobs(c):
    prefix=[c['python'],'-u','-m',MODULE]
    jobs=[]
    for variant,k in CELLS:
        key=key_for(variant,k)
        jobs.append(dict(id='eval_'+key,cmd=prefix+['rb-cell','--host','rb2','--variant',variant,'--k',str(k)],
            ready_file=str(INCOMING/key/'READY.json'),ready_fields=dict(verdict='TRANSFER_READY',variant=variant,k_c=k,steps=c['steps']),
            upstream_status_file=str(INCOMING/'training_queue_status.json'),
            summary=str(RB_OUTPUT/'completed'/f'eval_{key}.json'),
            completion=dict(verdict='EVAL_CELL_COMPLETE',variant=variant,k_c=k,episodes=500)))
        jobs.append(dict(id='profile_'+key,deps=['eval_'+key],cmd=prefix+['rb-profile','--host','rb2','--variant',variant,'--k',str(k)],
            summary=str(RB_OUTPUT/'completed'/f'profile_{key}.json'),
            completion=dict(verdict='PROFILE_COMPLETE',variant=variant,k_c=k,timing_valid=True)))
    return jobs


def publish_training_status():
    path = SD_OUTPUT/'queue_status.json'
    if not path.is_file():
        return
    try:
        subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','rb2',
            'mkdir','-p',str(INCOMING)],check=True,timeout=45)
        subprocess.run(['rsync','-a','-e','ssh -o BatchMode=yes -o ConnectTimeout=15',
            str(path),f'rb2:{INCOMING}/training_queue_status.json'],check=True,timeout=45)
    except (subprocess.SubprocessError,OSError) as exc:
        print(f'STATUS_TRANSFER_WARNING {exc}',flush=True)


def summarize(c, host):
    output=Path(c['output']); rows={}
    for variant,k in CELLS:
        key=key_for(variant,k)
        path=(output/'online'/f'kc{k}_{variant}'/'summary.json' if host=='sd1'
              else output/'completed'/f'eval_{key}.json')
        if path.exists(): rows[key]=read_json(path)
        profile=output/'completed'/f'profile_{key}.json'
        if host=='rb2' and key in rows and profile.exists(): rows[key]['latency_profile']=read_json(profile)
    write_json(output/'comparison_summary.json',dict(complete=len(rows)==len(CELLS),rows=rows,
        host=host,seed='seed01',episodes_per_row=500,action_nfe=3,generation_loop=False,
        training='Shared3K initialization plus3K per arm/interval; report bootstrap cost separately',
        interpretation='Matched feature architecture and initialization. Independently trained encoders. Anchor-input has additional active parameters, reported in training contracts. No superiority presupposed.'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('all','preflight','sd-train-cell','train','sd-eval','rb-cell','rb-smoke','rb-eval','rb-profile'))
    parser.add_argument('--host',choices=('sd1','rb2'),required=True)
    parser.add_argument('--variant',choices=(*VARIANTS,'bootstrap'))
    parser.add_argument('--k',type=int,choices=(4,8),default=4)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',type=Path)
    a=parser.parse_args()
    expected='jbrserver1' if a.host=='sd1' else 'jbr-TRX50'
    if socket.gethostname()!=expected: raise RuntimeError('Wrong host')
    base=SD_OUTPUT if a.host=='sd1' else RB_OUTPUT
    worker=a.command not in ('all','preflight')
    c=read_json((a.output or base)/'runtime_config.json') if worker else configuration(a.host)
    if a.host=='sd1': configure(c)
    else:
        from tools.simvla.compile_benchmark import configure as configure_rb
        configure_rb(c); sys.path.insert(0,c['libero_root']); os.environ['LIBERO_CONFIG_PATH']=c['libero_config']
    if worker and a.variant is None: parser.error('--variant required')
    if a.command=='train':
        from architectures.simvla.adapters.refresh_calibration.train import train
        train(c,a.variant,a.k,a.smoke); return 0
    if a.command=='sd-train-cell': sd_train_cell(c,a.variant,a.k); return 0
    if a.command=='sd-eval': sd_eval(c,a.variant,a.k,a.smoke); return 0
    if a.command=='rb-cell': rb_cell(c,a.variant,a.k); summarize(c,'rb2'); return 0
    if a.command=='rb-profile': rb_profile(c,a.variant,a.k); summarize(c,'rb2'); return 0
    if a.command in ('rb-smoke','rb-eval'):
        from tools.simvla import compiled_campaign as campaign
        from architectures.simvla.adapters.refresh_calibration import policy as adapter
        campaign.worker(c,a.output,'libero_10','seed01',a.variant,smoke=a.command=='rb-smoke',
            replay_factory=adapter.replay_factory,policy_factory=adapter.make_rb2_policy,
            policy_checker=adapter.check_counts,compiler_checker=adapter.check_compiler,reset_checker=adapter.check_reset)
        return 0
    base.mkdir(parents=True,exist_ok=True)
    if a.host=='sd1':
        from tools.simvla.error_compensation_campaign import prepare
        prepare(c)
    else:
        from tools.simvla.compile_benchmark import preflight
        preflight(c)
        if not Path(c['profile_observations']).is_file(): raise FileNotFoundError(c['profile_observations'])
        from tools.simvla.compiled_campaign import validate_manifest,manifest_path
        validate_manifest(read_json(manifest_path(c,'libero_10','seed01')),'libero_10','seed01')
    for predecessor in c['predecessors']:
        if not (Path(predecessor['path'])/predecessor['lock']).exists():
            raise FileNotFoundError('Missing predecessor: '+str(predecessor))
    config=base/'runtime_config.json'
    if config.exists() and read_json(config)!=c: raise RuntimeError('Saved configuration changed')
    write_json(config,c)
    jobs=sd_jobs(c) if a.host=='sd1' else rb_jobs(c)
    write_json(base/'planned_jobs.json',dict(jobs=jobs,predecessors=c['predecessors']))
    print(f'CPU_PREFLIGHT_PASS host={a.host} jobs={len(jobs)}; GPU smoke runs after predecessor and lease',flush=True)
    if a.command=='preflight': return 0
    try:
        rc=run_queue(base,jobs,gpus=(4,5,6,7) if a.host=='sd1' else (0,),predecessor=c['predecessors'],
            environment=lambda gpu: environment(c,gpu) if a.host=='sd1' else rb_environment(c,gpu),cwd=ROOT,timeout=86400)
    finally:
        summarize(c,a.host)
        if a.host=='sd1': publish_training_status()
    return rc


if __name__=='__main__': raise SystemExit(main())
