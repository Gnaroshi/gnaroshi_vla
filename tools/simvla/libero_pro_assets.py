"""Pinned official LIBERO-PRO assets and numeric-only initial-state loading."""
import argparse
import ast
import hashlib
import io
import os
from pathlib import Path
import pickle
import subprocess
import sys
import zipfile

import numpy as np

from tools.simvla.compile_benchmark import read_json, write_json, sha

CODE_URL = 'https://github.com/Zxy-MLlab/LIBERO-PRO.git'
CODE_COMMIT = 'eafdb809426b13153aa1e4c42d6601844217dfec'
DATA_REPO = 'zhouxueyang/LIBERO-Pro'
DATA_COMMIT = 'c86fc3b8293185a6f373677018ff3e37f8391602'
SUITES = ('libero_10_swap', 'libero_10_task')
STORAGE = Path('/home/mingyujung/private/gnaroshi_vla_storage')
ASSETS = STORAGE/'datasets/LIBERO-PRO'


class NumericStateUnpickler(pickle.Unpickler):
    # Official files wrap NumPy arrays in a torch ZIP with pickle protocol 4.
    # No arbitrary module imports, classes, persistent IDs or object arrays.
    def find_class(self, module, name):
        allowed = {
            ('numpy.core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
            ('numpy._core.multiarray', '_reconstruct'): np.core.multiarray._reconstruct,
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy', 'dtype'): np.dtype,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Forbidden initial-state global: {module}.{name}')
        return allowed[module, name]

    def persistent_load(self, pid):
        raise pickle.UnpicklingError('Persistent IDs are not initial-state data')


def load_numeric_states(path):
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.endswith('/data.pkl')]
        if len(names) != 1 or archive.getinfo(names[0]).file_size > 1024**2:
            raise ValueError('Unexpected initial-state archive')
        value = NumericStateUnpickler(io.BytesIO(archive.read(names[0]))).load()
    if (not isinstance(value, np.ndarray) or value.dtype != np.float64 or value.ndim != 2
            or value.shape[0] < 50 or value.shape[0] > 1000 or value.shape[1] > 1000
            or not np.isfinite(value).all()):
        raise ValueError('Initial states must be a finite float64 matrix with at least 50 rows')
    return value


def official_task_names(root, suite):
    path = Path(root)/'libero/libero/benchmark/libero_suite_task_map.py'
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(n, ast.Name) and n.id == 'libero_task_map' for n in node.targets):
            return ast.literal_eval(node.value)[suite]
    raise ValueError('Official task map missing')


def verify_source(upstream):
    head = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if head != CODE_COMMIT or subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise RuntimeError('LIBERO-PRO must be the unmodified pinned official commit')
    return head


def prepare(root=ASSETS):
    import yaml
    from huggingface_hub import HfApi, snapshot_download
    root = Path(root)
    upstream = root/'upstream'
    head = verify_source(upstream)
    benchmark = root/'benchmark'
    patterns = ['SHA256SUMS.txt', 'README.md'] + [f'{folder}/{suite}/*'
        for suite in SUITES for folder in ('bddl_files', 'init_files')]
    snapshot_download(DATA_REPO, repo_type='dataset', revision=DATA_COMMIT,
        local_dir=str(benchmark), allow_patterns=patterns, max_workers=2)
    info = HfApi().dataset_info(DATA_REPO, revision=DATA_COMMIT, files_metadata=True)
    if info.sha != DATA_COMMIT:
        raise RuntimeError('Dataset revision resolution changed')
    remote = {f.rfilename: f for f in info.siblings}
    sums = {}
    for line in (benchmark/'SHA256SUMS.txt').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        sums[name.lstrip('*').removeprefix('./')] = expected
    cfg = root/'runtime_config'
    cfg.mkdir(parents=True, exist_ok=True)
    paths = dict(benchmark_root=str(upstream/'libero/libero'), assets=str(upstream/'libero/libero/assets'),
        bddl_files=str(benchmark/'bddl_files'), init_states=str(benchmark/'init_files'), datasets=str(benchmark))
    (cfg/'config.yaml').write_text(yaml.safe_dump(paths))
    os.environ['LIBERO_CONFIG_PATH'] = str(cfg)
    sys.path.insert(0, str(upstream))
    from libero.libero.envs.bddl_utils import get_problem_info
    tasks = {}
    hashes = {}
    for suite in SUITES:
        tasks[suite] = []
        names = official_task_names(upstream, suite)
        if len(names) != 10:
            raise ValueError('Expected the official ten Long tasks')
        for index, name in enumerate(names):
            bddl = benchmark/'bddl_files'/suite/f'{name}.bddl'
            state = benchmark/'init_files'/suite/f'{name}.pruned_init'
            for path in (bddl, state):
                value = sha(path)
                relative = str(path.relative_to(benchmark))
                entry = remote[relative]
                raw = path.read_bytes()
                if entry.lfs:
                    verified = value == entry.lfs.sha256
                else:
                    verified = hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest() == entry.blob_id
                if not verified or (relative in sums and sums[relative] != value):
                    raise ValueError(f'Pinned official content identity mismatch: {path}')
                hashes[str(path)] = value
            numeric = load_numeric_states(state)
            target = root/'numeric_states'/suite/f'{name}.npy'
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, numeric, allow_pickle=False)
            hashes[str(target)] = sha(target)
            info = get_problem_info(str(bddl))
            prompt = info['language_instruction']
            if not prompt.strip():
                raise RuntimeError('BDDL prompt missing')
            tasks[suite].append(dict(task_id=index, name=name, language=prompt,
                bddl=str(bddl), init_file=str(state), numeric_states=str(target),
                state_count=len(numeric), unique_states=len(np.unique(numeric[:50], axis=0)),
                state_dimension=numeric.shape[1]))
    report = dict(verdict='PRO_ASSETS_VERIFIED', code_url=CODE_URL, code_commit=head,
        data_repo=DATA_REPO, data_revision=DATA_COMMIT, tasks=tasks, file_hashes=hashes,
        prompt_source='BDDL :language; never the unchanged filename',
        download_verification='HF pinned-revision Git blob/LFS content IDs; SHA256SUMS additionally checked for entries it covers',
        init_loading='Restricted NumPy-only unpickler for preparation; allow_pickle=False NumPy loading in evaluation')
    write_json(root/'asset_contract.json', report)
    print('PRO_ASSETS_VERIFIED', {k: len(v) for k, v in tasks.items()}, flush=True)
    return report


def verify(root=ASSETS):
    verify_source(Path(root)/'upstream')
    report = read_json(Path(root)/'asset_contract.json')
    if report['code_commit'] != CODE_COMMIT or report['data_revision'] != DATA_COMMIT:
        raise RuntimeError('Unexpected official asset revision')
    for path, value in report['file_hashes'].items():
        if sha(path) != value:
            raise RuntimeError('PRO asset changed: ' + path)
    return report


def check_environments(root=ASSETS):
    root = Path(root)
    report = verify(root)
    os.environ['LIBERO_CONFIG_PATH'] = str(root/'runtime_config')
    sys.path.insert(0, str(root/'upstream'))
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    checked = []
    for suite in SUITES:
        official = benchmark.get_benchmark_dict()[suite]()
        for spec in report['tasks'][suite]:
            task = official.get_task(spec['task_id'])
            if task.name != spec['name'] or task.problem_folder != suite:
                raise RuntimeError('Official task mapping differs from asset manifest')
            env = OffScreenRenderEnv(bddl_file_name=spec['bddl'], camera_heights=256, camera_widths=256)
            try:
                env.seed(7)
                env.reset()
                states = np.load(spec['numeric_states'], allow_pickle=False)
                env.set_init_state(states[0])
                for _ in range(10):
                    obs, _, _, _ = env.step([0., 0., 0., 0., 0., 0., -1.])
                for key in ('agentview_image', 'robot0_eye_in_hand_image'):
                    if obs[key].shape != (256, 256, 3) or obs[key].std() == 0:
                        raise RuntimeError('Blank/malformed PRO camera: '+key)
                checked.append(dict(suite=suite, task_id=spec['task_id'], state_dimension=states.shape[1],
                    initial_success=bool(env.check_success())))
                print(f'ENV_OK {suite}/{spec["task_id"]}', flush=True)
            finally:
                env.close()
    write_json(root/'environment_check.json', dict(verdict='PRO_ENV_RESET_WAIT_RENDER_PASS',
        renderer=os.environ.get('MUJOCO_GL'), gpu_policy_inference=False, checked=checked))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ASSETS)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--check-environments', action='store_true')
    args = parser.parse_args()
    if args.check_environments:
        check_environments(args.root)
    elif args.verify:
        verify(args.root)
    else:
        prepare(args.root)
